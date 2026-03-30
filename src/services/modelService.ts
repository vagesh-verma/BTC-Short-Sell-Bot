import * as tf from '@tensorflow/tfjs';
import { logger } from './loggerService';

export class GRUModel {
  private model: tf.LayersModel | null = null;
  private windowSize: number;
  private featureCount: number;

  constructor(windowSize: number = 20, featureCount: number = 12) {
    this.windowSize = windowSize;
    this.featureCount = featureCount;
  }

  public async buildModel(
    units: number = 128, 
    dropout: number = 0.2, 
    learningRate: number = 0.001
  ) {
    logger.info(`Building GRU Model (Window: ${this.windowSize}, Features: ${this.featureCount}, Units: ${units}, Dropout: ${dropout}, LR: ${learningRate})...`);
    const model = tf.sequential();
    
    // GRU Layer
    model.add(tf.layers.gru({
      units: units,
      inputShape: [this.windowSize, this.featureCount],
      returnSequences: true,
    }));
    
    model.add(tf.layers.dropout({ rate: dropout }));

    model.add(tf.layers.gru({
      units: Math.floor(units / 2),
      returnSequences: false,
    }));
    
    model.add(tf.layers.dropout({ rate: dropout }));
    
    // Output Layer: Predict if the price will go DOWN (Short opportunity)
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    model.compile({
      optimizer: tf.train.adam(learningRate),
      loss: 'binaryCrossentropy',
      metrics: ['accuracy'],
    });

    this.model = model;
    logger.success('Model architecture compiled.');
  }

  public async train(
    data: number[], 
    labels: number[], 
    epochs: number = 10, 
    units: number = 128,
    dropout: number = 0.2,
    learningRate: number = 0.001,
    onEpochEnd?: (epoch: number, logs?: tf.Logs) => void
  ) {
    if (!this.model) await this.buildModel(units, dropout, learningRate);
    
    const numSamples = data.length / (this.windowSize * this.featureCount);
    logger.info(`Starting training with ${numSamples} samples for ${epochs} epochs...`);
    
    const xs = tf.tensor3d(data, [numSamples, this.windowSize, this.featureCount]);
    const ys = tf.tensor2d(labels, [labels.length, 1]);

    await this.model!.fit(xs, ys, {
      epochs,
      batchSize: 32,
      shuffle: true,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          if (logs) {
            logger.info(`Epoch ${epoch + 1}/${epochs} - loss: ${logs.loss.toFixed(4)}, acc: ${logs.acc.toFixed(4)}`);
          }
          if (onEpochEnd) onEpochEnd(epoch, logs);
        }
      }
    });

    logger.success('Training completed.');
    xs.dispose();
    ys.dispose();
  }

  public predict(input: number[]): number {
    if (!this.model) return 0;
    
    const inputTensor = tf.tensor3d(input, [1, this.windowSize, this.featureCount]);
    const prediction = this.model.predict(inputTensor) as tf.Tensor;
    const result = prediction.dataSync()[0];
    
    inputTensor.dispose();
    prediction.dispose();
    
    return result;
  }

  public async save(name: string) {
    if (!this.model) throw new Error('No model to save');
    await this.model.save(`indexeddb://${name}`);
    localStorage.setItem(`${name}_metadata`, JSON.stringify({
      windowSize: this.windowSize,
      featureCount: this.featureCount
    }));
  }

  public async getArtifacts(): Promise<tf.io.ModelArtifacts> {
    if (!this.model) throw new Error('No model to get artifacts from');
    let artifacts: tf.io.ModelArtifacts | null = null;
    await this.model.save({
      save: (art) => {
        artifacts = art;
        return Promise.resolve({ modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON' } });
      }
    });
    if (!artifacts) throw new Error('Failed to capture model artifacts');
    return artifacts;
  }

  public static async load(name: string): Promise<GRUModel> {
    const metadataStr = localStorage.getItem(`${name}_metadata`);
    if (!metadataStr) throw new Error('Model metadata not found');
    const metadata = JSON.parse(metadataStr);
    
    const gru = new GRUModel(metadata.windowSize, metadata.featureCount);
    try {
      gru.model = await tf.loadLayersModel(`indexeddb://${name}`);
    } catch (err) {
      logger.info(`Model ${name} not found in IndexedDB, trying LocalStorage...`);
      gru.model = await tf.loadLayersModel(`localstorage://${name}`);
    }
    
    gru.model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'binaryCrossentropy',
      metrics: ['accuracy'],
    });
    return gru;
  }

  public static async loadFromArtifacts(artifacts: tf.io.ModelArtifacts, metadata: { windowSize: number, featureCount: number }): Promise<GRUModel> {
    const gru = new GRUModel(metadata.windowSize, metadata.featureCount);
    gru.model = await tf.loadLayersModel({
      load: () => Promise.resolve(artifacts)
    });
    gru.model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'binaryCrossentropy',
      metrics: ['accuracy'],
    });
    return gru;
  }

  public static async remove(name: string) {
    try {
      await tf.io.removeModel(`indexeddb://${name}`);
    } catch (err) {
      // Ignore if not found in IndexedDB
    }
    try {
      await tf.io.removeModel(`localstorage://${name}`);
    } catch (err) {
      // Ignore if not found in LocalStorage
    }
    localStorage.removeItem(`${name}_metadata`);
  }
}

export function prepareData(
  prices: number[], 
  rsi: number[], 
  ema: number[], 
  bbUpper: number[], 
  bbLower: number[], 
  windowSize: number,
  secondaryPredictions?: number[],
  macdHistogram?: number[],
  stochRsi?: number[],
  atr?: number[],
  dropThreshold: number = 0.5,
  ema9?: number[],
  emaBelow?: number[],
  emaCross?: number[],
  obv?: number[],
  mfi?: number[],
  volatility?: number[],
  hourOfDay?: number[],
  dayOfWeek?: number[],
  bearishHarami?: number[],
  marubozu?: number[],
  engulfing?: number[]
) {
  const samples: { x: number[], y: number }[] = [];
  
  for (let i = windowSize; i < prices.length - 1; i++) {
    const windowIndices = Array.from({ length: windowSize }, (_, k) => i - windowSize + k);
    
    const priceWindow = windowIndices.map(idx => prices[idx]);
    const rsiWindow = windowIndices.map(idx => rsi[idx]);
    const emaWindow = windowIndices.map(idx => ema[idx]);
    const bbUpperWindow = windowIndices.map(idx => bbUpper[idx]);
    const bbLowerWindow = windowIndices.map(idx => bbLower[idx]);
    const secondaryWindow = secondaryPredictions ? windowIndices.map(idx => secondaryPredictions[idx]) : [];
    const macdWindow = macdHistogram ? windowIndices.map(idx => macdHistogram[idx]) : [];
    const stochRsiWindow = stochRsi ? windowIndices.map(idx => stochRsi[idx]) : [];
    const atrWindow = atr ? windowIndices.map(idx => atr[idx]) : [];
    const ema9Window = ema9 ? windowIndices.map(idx => ema9[idx]) : [];
    const emaBelowWindow = emaBelow ? windowIndices.map(idx => emaBelow[idx]) : [];
    const emaCrossWindow = emaCross ? windowIndices.map(idx => emaCross[idx]) : [];
    const obvWindow = obv ? windowIndices.map(idx => obv[idx]) : [];
    const mfiWindow = mfi ? windowIndices.map(idx => mfi[idx]) : [];
    const volWindow = volatility ? windowIndices.map(idx => volatility[idx]) : [];
    const hourWindow = hourOfDay ? windowIndices.map(idx => hourOfDay[idx]) : [];
    const dayWindow = dayOfWeek ? windowIndices.map(idx => dayOfWeek[idx]) : [];
    const haramiWindow = bearishHarami ? windowIndices.map(idx => bearishHarami[idx]) : [];
    const marubozuWindow = marubozu ? windowIndices.map(idx => marubozu[idx]) : [];
    const engulfingWindow = engulfing ? windowIndices.map(idx => engulfing[idx]) : [];

    const normalize = (arr: number[]) => {
      const min = Math.min(...arr);
      const max = Math.max(...arr);
      return arr.map(v => (max === min ? 0 : (v - min) / (max - min)));
    };

    const normPrice = normalize(priceWindow);
    const normRsi = rsiWindow.map(v => v / 100);
    const normEma = normalize(emaWindow);
    const normBbUpper = normalize(bbUpperWindow);
    const normBbLower = normalize(bbLowerWindow);
    const normMacd = macdHistogram ? normalize(macdWindow) : [];
    const normAtr = atr ? normalize(atrWindow) : [];
    const normEma9 = ema9 ? normalize(ema9Window) : [];
    const normObv = obv ? normalize(obvWindow) : [];
    const normVol = volatility ? normalize(volWindow) : [];

    const x: number[] = [];
    for (let j = 0; j < windowSize; j++) {
      x.push(normPrice[j], normRsi[j], normEma[j], normBbUpper[j], normBbLower[j]);
      if (macdHistogram) x.push(normMacd[j]);
      if (stochRsi) x.push(stochRsiWindow[j]);
      if (atr) x.push(normAtr[j]);
      if (secondaryPredictions) x.push(secondaryWindow[j]);
      if (ema9) x.push(normEma9[j]);
      if (emaBelow) x.push(emaBelowWindow[j]);
      if (emaCross) x.push(emaCrossWindow[j]);
      if (obv) x.push(normObv[j]);
      if (mfi) x.push(mfiWindow[j] / 100);
      if (volatility) x.push(normVol[j]);
      if (hourOfDay) {
        const h = hourWindow[j];
        x.push(h / 24);
        // Asia: 0-9 UTC
        x.push(h >= 0 && h <= 9 ? 1 : 0);
        // London: 8-17 UTC
        x.push(h >= 8 && h <= 17 ? 1 : 0);
        // NY: 13-22 UTC
        x.push(h >= 13 && h <= 22 ? 1 : 0);
      }
      if (dayOfWeek) x.push(dayWindow[j] / 7);
      if (bearishHarami) x.push(haramiWindow[j]);
      if (marubozu) x.push(marubozuWindow[j]);
      if (engulfing) x.push(engulfingWindow[j]);
    }
    
    const currentPrice = prices[i];
    const nextPrice = prices[i + 1];
    // dropThreshold is percentage, e.g. 0.5 means 0.5% drop
    const multiplier = 1 - (dropThreshold / 100);
    const y = nextPrice < currentPrice * multiplier ? 1 : 0;
    
    samples.push({ x, y });
  }

  const positiveSamples = samples.filter(s => s.y === 1);
  const negativeSamples = samples.filter(s => s.y === 0);
  
  logger.info(`Data prepared: ${samples.length} total samples.`);
  logger.info(`Initial balance: ${positiveSamples.length} positive (drops), ${negativeSamples.length} negative.`);

  const balancedSamples = [...negativeSamples];
  if (positiveSamples.length > 0) {
    // Aim for roughly 20% positive samples if possible
    const targetPositiveCount = Math.floor(negativeSamples.length * 0.25);
    const multiplier = Math.ceil(targetPositiveCount / positiveSamples.length);
    
    logger.info(`Oversampling positive class ${multiplier}x to balance training.`);
    for (let m = 0; m < multiplier; m++) {
      balancedSamples.push(...positiveSamples);
    }
  }

  logger.info(`Final balanced dataset: ${balancedSamples.length} samples.`);

  // Shuffle balanced samples
  for (let i = balancedSamples.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [balancedSamples[i], balancedSamples[j]] = [balancedSamples[j], balancedSamples[i]];
  }

  const xs = balancedSamples.flatMap(s => s.x);
  const ys = balancedSamples.map(s => s.y);

  return { 
    xs, 
    ys, 
    stats: { 
      total: samples.length, 
      positive: positiveSamples.length, 
      negative: negativeSamples.length 
    } 
  };
}
