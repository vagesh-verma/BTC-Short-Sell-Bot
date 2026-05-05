import * as tf from '@tensorflow/tfjs';
import { logger } from './loggerService';

// Define the possible actions
export enum Action {
  LONG = 0,
  SHORT = 1,
  NEUTRAL = 2
}

// PPO Trajectory Data
interface Experience {
  state: number[];
  action: Action;
  logProb: number;
  reward: number;
  value: number;
  done: boolean;
}

export interface TrainingProgress {
  episode: number;
  reward: number;
  loss: number;
  epsilon: number;
  accuracy: number;
}

export class DRLService {
  private actorModel: tf.LayersModel | null = null;
  private criticModel: tf.LayersModel | null = null;
  private memory: Experience[] = [];
  
  // PPO Hyperparameters
  private gamma: number = 0.99; // Discount factor
  private gaeLambda: number = 0.95; // GAE parameter
  private clipRatio: number = 0.2; // PPO clip range
  private ppoEpochs: number = 10; // Number of training epochs per batch
  private batchSize: number = 64;
  private actorLR: number = 0.0003;
  private criticLR: number = 0.001;
  private entropyCoef: number = 0.01; // Entropy bonus for exploration
  
  private featureCount: number;
  private windowSize: number;

  constructor(windowSize: number = 20, featureCount: number = 24) {
    this.windowSize = windowSize;
    this.featureCount = featureCount;
    this.initModels();
  }

  public initialize(windowSize: number, featureCount: number) {
    if (this.windowSize === windowSize && this.featureCount === featureCount && this.actorModel) {
      return; // Already initialized with these parameters
    }
    
    logger.info(`Initializing PPO Model: Window=${windowSize}, Features=${featureCount}`);
    this.windowSize = windowSize;
    this.featureCount = featureCount;
    this.initModels();
  }

  private initModels() {
    const inputShape = [this.windowSize * this.featureCount + 3]; // +3 for current position state
    
    // Actor: Outputs probabilities for each action
    this.actorModel = this.createActorModel(inputShape);
    // Critic: Outputs scalar value for the state
    this.criticModel = this.createCriticModel(inputShape);
  }

  private createActorModel(inputShape: number[]): tf.LayersModel {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 128, activation: 'relu', inputShape }));
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' })); // Output: Probabilities

    model.compile({
      optimizer: tf.train.adam(this.actorLR),
      loss: 'categoricalCrossentropy' // We will use a custom training loop for PPO
    });
    return model;
  }

  private createCriticModel(inputShape: number[]): tf.LayersModel {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 128, activation: 'relu', inputShape }));
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'linear' })); // Output: Value

    model.compile({
      optimizer: tf.train.adam(this.criticLR),
      loss: 'meanSquaredError'
    });
    return model;
  }

  /**
   * Selects an action based on the current policy.
   * During training, we sample from the distribution.
   * During inference, we take the argmax.
   */
  public act(state: number[], train: boolean = true): { action: Action, logProb: number, value: number } {
    return tf.tidy(() => {
      const input = tf.tensor2d([state]);
      const probs = this.actorModel!.predict(input) as tf.Tensor;
      const value = this.criticModel!.predict(input) as tf.Tensor;
      const probsData = probs.dataSync();
      const valData = value.dataSync()[0];

      let action: Action;
      let logProb: number;

      if (train) {
        // Sample from distribution
        const r = Math.random();
        let cumulative = 0;
        action = Action.NEUTRAL;
        for (let i = 0; i < probsData.length; i++) {
          cumulative += probsData[i];
          if (r <= cumulative) {
            action = i as Action;
            break;
          }
        }
        logProb = Math.log(probsData[action] + 1e-10);
      } else {
        // Deterministic argmax
        action = probs.argMax(1).dataSync()[0] as Action;
        logProb = Math.log(probsData[action] + 1e-10);
      }

      return { action, logProb, value: valData };
    });
  }

  public async train(
    marketData: number[][], 
    prices: number[],
    episodes: number = 10,
    onProgress?: (progress: TrainingProgress) => void
  ) {
    logger.info('Starting DRL Training (PPO)...');

    for (let e = 0; e < episodes; e++) {
      let totalReward = 0;
      let currentPosition: Action = Action.NEUTRAL;
      let entryPrice = 0;
      this.memory = [];

      // 1. Collect trajectories (one episode)
      for (let i = 0; i < marketData.length - 1; i++) {
        const state = [...marketData[i], ...this.encodePosition(currentPosition)];
        const { action, logProb, value } = this.act(state, true);
        
        let reward = 0;
        const currentPrice = prices[i];
        const nextPrice = prices[i + 1];
        
        // Reward logic (Scalable, based on price movement)
        if (action === Action.LONG) {
          if (currentPosition !== Action.LONG) {
            entryPrice = currentPrice;
            currentPosition = Action.LONG;
          }
          reward = (nextPrice - currentPrice) / currentPrice;
        } else if (action === Action.SHORT) {
          if (currentPosition !== Action.SHORT) {
            entryPrice = currentPrice;
            currentPosition = Action.SHORT;
          }
          reward = (currentPrice - nextPrice) / currentPrice;
        } else {
          currentPosition = Action.NEUTRAL;
          reward = 0;
        }

        const done = i === marketData.length - 2;
        this.memory.push({ state, action, logProb, reward, value, done });
        totalReward += reward;
      }

      // 2. Compute Advantages and Returns using GAE
      const { advantages, returns } = this.computeGAE();

      // 3. PPO Update Epochs
      const loss = await this.updatePolicy(advantages, returns);
      
      if (onProgress) {
        onProgress({
          episode: e + 1,
          reward: totalReward,
          loss: loss,
          epsilon: 0, // PPO doesn't use epsilon-greedy
          accuracy: 0
        });
      }

      logger.info(`Episode ${e+1}/${episodes} completed. Reward: ${totalReward.toFixed(4)}, Loss: ${loss.toFixed(6)}`);
    }

    logger.success('PPO Training complete.');
  }

  private computeGAE() {
    const advantages = new Array(this.memory.length).fill(0);
    const returns = new Array(this.memory.length).fill(0);
    let lastAdvantage = 0;
    let lastValue = 0;

    for (let i = this.memory.length - 1; i >= 0; i--) {
      const exp = this.memory[i];
      const delta = exp.reward + (exp.done ? 0 : this.gamma * (this.memory[i + 1]?.value || 0)) - exp.value;
      advantages[i] = delta + this.gamma * this.gaeLambda * (exp.done ? 0 : lastAdvantage);
      lastAdvantage = advantages[i];
      returns[i] = advantages[i] + exp.value;
    }

    // Normalize advantages for stability
    const mean = advantages.reduce((a, b) => a + b) / advantages.length;
    const std = Math.sqrt(advantages.reduce((a, b) => a + Math.pow(b - mean, 2)) / advantages.length) + 1e-8;
    const normalizedAdvantages = advantages.map(a => (a - mean) / std);

    return { advantages: normalizedAdvantages, returns };
  }

  private async updatePolicy(advantages: number[], returns: number[]): Promise<number> {
    const states = this.memory.map(m => m.state);
    const actions = this.memory.map(m => m.action);
    const oldLogProbs = this.memory.map(m => m.logProb);

    const tensorStates = tf.tensor2d(states);
    const tensorActions = tf.tensor1d(actions, 'int32');
    const tensorOldLogProbs = tf.tensor1d(oldLogProbs);
    const tensorAdvantages = tf.tensor1d(advantages);
    const tensorReturns = tf.tensor2d(returns.map(r => [r]));

    let totalLoss = 0;

    for (let epoch = 0; epoch < this.ppoEpochs; epoch++) {
      // 1. Actor Update (Policy Gradient with Clipping)
      const actorLoss = await this.trainActorBatch(tensorStates, tensorActions, tensorOldLogProbs, tensorAdvantages);
      
      // 2. Critic Update (Value Function)
      const criticResult = await this.criticModel!.fit(tensorStates, tensorReturns, {
        epochs: 1,
        batchSize: this.batchSize,
        verbose: 0
      });

      totalLoss += actorLoss + (criticResult.history.loss[0] as number);
    }

    tf.dispose([tensorStates, tensorActions, tensorOldLogProbs, tensorAdvantages, tensorReturns]);
    return totalLoss / this.ppoEpochs;
  }

  private async trainActorBatch(states: tf.Tensor2D, actions: tf.Tensor1D, oldLogProbs: tf.Tensor1D, advantages: tf.Tensor1D): Promise<number> {
    const optimizer = this.actorModel!.optimizer as tf.AdamOptimizer;
    
    const lossFunc = () => {
      return tf.tidy(() => {
        const probs = this.actorModel!.predict(states) as tf.Tensor2D;
        
        // Gather probabilities of actions taken
        const actionMask = tf.oneHot(actions, 3);
        const actionProbs = tf.sum(tf.mul(probs, actionMask), 1);
        const newLogProbs = tf.log(tf.add(actionProbs, 1e-10));

        // PPO Clip Objective
        const ratio = tf.exp(tf.sub(newLogProbs, oldLogProbs));
        const surr1 = tf.mul(ratio, advantages);
        const surr2 = tf.mul(tf.clipByValue(ratio, 1 - this.clipRatio, 1 + this.clipRatio), advantages);
        
        const policyLoss = tf.neg(tf.mean(tf.minimum(surr1, surr2)));

        // Entropy Bonus (encourages exploration)
        const entropy = tf.neg(tf.sum(tf.mul(probs, tf.log(tf.add(probs, 1e-10))), 1));
        const entropyLoss = tf.neg(tf.mean(entropy));

        return tf.add(policyLoss, tf.mul(this.entropyCoef, entropyLoss)) as tf.Scalar;
      });
    };

    const cost = optimizer.minimize(lossFunc, true);
    const lossValue = cost ? (await cost.data())[0] : 0;
    tf.dispose(cost);
    return lossValue;
  }

  private encodePosition(pos: Action): number[] {
    if (pos === Action.LONG) return [1, 0, 0];
    if (pos === Action.SHORT) return [0, 1, 0];
    return [0, 0, 1];
  }

  public predict(state: number[]): Action {
    if (!this.actorModel) return Action.NEUTRAL;
    const { action } = this.act([...state, ...this.encodePosition(Action.NEUTRAL)], false);
    return action;
  }

  public async saveToLocalStorage(name: string) {
    if (this.actorModel && this.criticModel) {
      await this.actorModel.save(`indexeddb://ppo-actor-${name}`);
      await this.criticModel.save(`indexeddb://ppo-critic-${name}`);
      localStorage.setItem(`ppo-meta-${name}`, JSON.stringify({
        windowSize: this.windowSize,
        featureCount: this.featureCount
      }));
      logger.success(`PPO Model "${name}" saved.`);
    }
  }

  public async loadFromLocalStorage(name: string) {
    try {
      this.actorModel = await tf.loadLayersModel(`indexeddb://ppo-actor-${name}`);
      this.criticModel = await tf.loadLayersModel(`indexeddb://ppo-critic-${name}`);
      
      this.actorModel.compile({ optimizer: tf.train.adam(this.actorLR), loss: 'categoricalCrossentropy' });
      this.criticModel.compile({ optimizer: tf.train.adam(this.criticLR), loss: 'meanSquaredError' });

      const meta = localStorage.getItem(`ppo-meta-${name}`);
      if (meta) {
        const { windowSize, featureCount } = JSON.parse(meta);
        this.windowSize = windowSize;
        this.featureCount = featureCount;
      }
      logger.success(`PPO Model "${name}" loaded.`);
      return true;
    } catch (e) {
      logger.error(`Failed to load PPO model "${name}".`);
      return false;
    }
  }

  public async getModelArtifacts() {
    if (!this.actorModel) return null;
    let savedModel: any = null;
    await this.actorModel.save(tf.io.withSaveHandler(async (artifacts) => {
      savedModel = artifacts;
      return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON' } };
    }));
    return savedModel;
  }
}

export const drlService = new DRLService();
