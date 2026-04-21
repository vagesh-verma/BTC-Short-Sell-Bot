import * as tf from '@tensorflow/tfjs';
import { logger } from './loggerService';

export interface DataPreparationOptions {
  prices: number[];
  highs?: number[];
  lows?: number[];
  rsi: number[];
  ema: number[];
  bbUpper: number[];
  bbLower: number[];
  windowSize: number;
  secondaryPredictions?: number[];
  macdHistogram?: number[];
  stochRsi?: number[];
  atr?: number[];
  dropThreshold?: number;
  longThreshold?: number;
  maxLookahead?: number;
  ema9?: number[];
  emaBelow?: number[];
  emaCross?: number[];
  obv?: number[];
  mfi?: number[];
  volatility?: number[];
  hourOfDay?: number[];
  dayOfWeek?: number[];
  bearishHarami?: number[];
  marubozu?: number[];
  engulfing?: number[];
  startIndex?: number;
}

export class GRUModel {
  private model: tf.LayersModel | null = null;
  private windowSize: number;
  private featureCount: number;
  public name: string = 'unnamed';

  constructor(windowSize: number = 20, featureCount: number = 12, name: string = 'unnamed') {
    this.windowSize = windowSize;
    this.featureCount = featureCount;
    this.name = name;
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
    
    // Output Layer: Predict Short, Long, or Sideways
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));

    model.compile({
      optimizer: tf.train.adam(learningRate),
      loss: 'categoricalCrossentropy',
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
    const ys = tf.tensor2d(labels, [labels.length / 3, 3]);

    await this.model!.fit(xs, ys, {
      epochs,
      batchSize: 32,
      shuffle: true,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          if (logs) {
            logger.info(`Epoch ${epoch + 1}/${epochs} - loss: ${(logs.loss || 0).toFixed(4)}, acc: ${(logs.acc || 0).toFixed(4)}`);
          }
          if (onEpochEnd) onEpochEnd(epoch, logs);
        }
      }
    });

    logger.success('Training completed.');
    xs.dispose();
    ys.dispose();
  }

  public predict(input: number[]): number[] {
    if (!this.model) return [0, 0, 1];
    
    const expectedSize = this.windowSize * this.featureCount;
    let finalInput = input;
    if (input.length > expectedSize) {
      finalInput = input.slice(0, expectedSize);
    } else if (input.length < expectedSize) {
      finalInput = [...input, ...new Array(expectedSize - input.length).fill(0)];
    }

    return tf.tidy(() => {
      const inputTensor = tf.tensor3d(finalInput, [1, this.windowSize, this.featureCount]);
      const prediction = this.model!.predict(inputTensor) as tf.Tensor;
      return Array.from(prediction.dataSync());
    });
  }

  public predictMultiple(input: number[], passes: number = 10): { mean: number[], std: number[] } {
    if (!this.model) return { mean: [0, 0, 1], std: [0, 0, 0] };

    const expectedSize = this.windowSize * this.featureCount;
    let finalInput = input;
    if (input.length > expectedSize) {
      finalInput = input.slice(0, expectedSize);
    } else if (input.length < expectedSize) {
      finalInput = [...input, ...new Array(expectedSize - input.length).fill(0)];
    }

    return tf.tidy(() => {
      const inputTensor = tf.tensor3d(finalInput, [1, this.windowSize, this.featureCount]);
      const predictions: number[][] = [];
      
      for (let i = 0; i < passes; i++) {
        // Monte Carlo Dropout - this depends on model being compiled with dropout
        const pred = this.model!.predict(inputTensor) as tf.Tensor;
        predictions.push(Array.from(pred.dataSync()));
      }

      const numClasses = 3;
      const means: number[] = new Array(numClasses).fill(0);
      const stds: number[] = new Array(numClasses).fill(0);

      for (let c = 0; c < numClasses; c++) {
        const classProbs = predictions.map(p => p[c]);
        const mean = classProbs.reduce((a, b) => a + b, 0) / passes;
        const variance = classProbs.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / passes;
        means[c] = mean;
        stds[c] = Math.sqrt(variance);
      }

      return { mean: means, std: stds };
    });
  }

  public async save(name?: string) {
    if (!this.model) return;
    const saveName = name || this.name;
    await this.model.save(`indexeddb://${saveName}`);
    localStorage.setItem(`${saveName}_metadata`, JSON.stringify({
      windowSize: this.windowSize,
      featureCount: this.featureCount
    }));
  }

  public async getArtifacts(): Promise<tf.io.ModelArtifacts> {
    if (!this.model) throw new Error('Model not initialized');
    return new Promise((resolve) => {
      this.model!.save(tf.io.withSaveHandler(async (artifacts) => {
        resolve(artifacts);
        return {
          modelArtifactsInfo: {
            dateSaved: new Date(),
            modelTopologyType: 'JSON',
          } as any
        };
      }));
    });
  }

  public static async load(name: string): Promise<GRUModel | null> {
    try {
      const metadataStr = localStorage.getItem(`${name}_metadata`);
      if (!metadataStr) return null;
      const metadata = JSON.parse(metadataStr);
      const gru = new GRUModel(metadata.windowSize, metadata.featureCount, name);
      gru.model = await tf.loadLayersModel(`indexeddb://${name}`);
      gru.model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
      });
      return gru;
    } catch (err) {
      console.error('Error loading model:', err);
      return null;
    }
  }

  public static async loadFromArtifacts(artifacts: tf.io.ModelArtifacts, metadata: { windowSize: number, featureCount: number }, name: string = 'unnamed'): Promise<GRUModel> {
    const gru = new GRUModel(metadata.windowSize, metadata.featureCount, name);
    gru.model = await tf.loadLayersModel({
      load: () => Promise.resolve(artifacts)
    });
    gru.model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });
    return gru;
  }

  public static async remove(name: string) {
    try {
      await tf.io.removeModel(`indexeddb://${name}`);
    } catch (err) {}
    try {
      await tf.io.removeModel(`localstorage://${name}`);
    } catch (err) {}
    localStorage.removeItem(`${name}_metadata`);
  }
}

export function prepareData(options: DataPreparationOptions) {
    const {
    prices, highs, lows, rsi, ema, bbUpper, bbLower, windowSize,
    secondaryPredictions, macdHistogram, stochRsi, atr,
    dropThreshold = 0.5, longThreshold = 0.5, maxLookahead = 12,
    ema9, emaBelow, emaCross, obv, mfi, volatility,
    hourOfDay, dayOfWeek, bearishHarami, marubozu, engulfing,
    startIndex = 0
  } = options;

  const samples: { x: number[], y: number[] }[] = [];
  const actualStart = Math.max(windowSize, startIndex);
  
  for (let i = actualStart; i < prices.length - 1; i++) {
    const windowIndices = Array.from({ length: windowSize }, (_, k) => i - windowSize + k);
    
    const priceWindow = windowIndices.map(idx => prices[idx]);
    const rsiWindow = windowIndices.map(idx => rsi[idx]);
    const emaWindow = windowIndices.map(idx => ema[idx]);
    const bbUpperWindow = windowIndices.map(idx => bbUpper[idx]);
    const bbLowerWindow = windowIndices.map(idx => bbLower[idx]);

    const normalize = (arr: number[]) => {
      const min = Math.min(...arr);
      const max = Math.max(...arr);
      return arr.map(v => (max === min ? 0.5 : (v - min) / (max - min)));
    };

    const normPrice = normalize(priceWindow);
    const normRsi = rsiWindow.map(v => v / 100);
    const normEma = normalize(emaWindow);
    const normBbUpper = normalize(bbUpperWindow);
    const normBbLower = normalize(bbLowerWindow);

    const x: number[] = [];
    for (let j = 0; j < windowSize; j++) {
      x.push(normPrice[j], normRsi[j], normEma[j], normBbUpper[j], normBbLower[j]);
      
      if (secondaryPredictions) x.push(secondaryPredictions[i - windowSize + j + 1] || 0.5);
      if (macdHistogram) x.push(normalize(windowIndices.map(idx => macdHistogram[idx]))[j]);
      if (stochRsi) x.push(stochRsi[windowIndices[j]]);
      if (atr) x.push(normalize(windowIndices.map(idx => atr[idx]))[j]);
      if (ema9) x.push(normalize(windowIndices.map(idx => ema9[idx]))[j]);
      if (emaBelow) x.push(emaBelow[windowIndices[j]]);
      if (emaCross) x.push(emaCross[windowIndices[j]]);
      if (obv) x.push(normalize(windowIndices.map(idx => obv[idx]))[j]);
      if (mfi) x.push(mfi[windowIndices[j]] / 100);
      if (volatility) x.push(normalize(windowIndices.map(idx => volatility[idx]))[j]);
      
      if (hourOfDay) {
        const h = hourOfDay[windowIndices[j]];
        x.push(h / 24);
        x.push(h >= 0 && h <= 9 ? 1 : 0);
        x.push(h >= 8 && h <= 17 ? 1 : 0);
        x.push(h >= 13 && h <= 22 ? 1 : 0);
      }
      if (dayOfWeek) x.push(dayOfWeek[windowIndices[j]] / 7);
      if (bearishHarami) x.push(bearishHarami[windowIndices[j]]);
      if (marubozu) x.push(marubozu[windowIndices[j]]);
      if (engulfing) x.push(engulfing[windowIndices[j]]);
    }
    
    const currentPrice = prices[i];
    
    const shortTP = currentPrice * (1 - (dropThreshold / 100));
    const shortSL = currentPrice * (1 + (dropThreshold / 100));
    
    const longTP = currentPrice * (1 + (longThreshold / 100));
    const longSL = currentPrice * (1 - (longThreshold / 100));

    let isShort = false;
    let isLong = false;

    // Look ahead to check if target achieved before opposite movement
    const lookLimit = Math.min(i + maxLookahead, prices.length - 1);
    
    // Check for SHORT signal
    for (let j = i + 1; j <= lookLimit; j++) {
      const high = highs ? highs[j] : prices[j];
      const low = lows ? lows[j] : prices[j];
      
      if (low <= shortTP && high < shortSL) {
        isShort = true;
        break;
      }
      if (high >= shortSL) break; // Hit SL first or in same bar (conservative)
    }

    // Check for LONG signal (if not already short, or check separately for multi-label, but here we pick one)
    if (!isShort) {
      for (let j = i + 1; j <= lookLimit; j++) {
        const high = highs ? highs[j] : prices[j];
        const low = lows ? lows[j] : prices[j];
        
        if (high >= longTP && low > longSL) {
          isLong = true;
          break;
        }
        if (low <= longSL) break; // Hit SL first or in same bar
      }
    }

    let classification: number[];
    if (isShort) {
      classification = [1, 0, 0];
    } else if (isLong) {
      classification = [0, 1, 0];
    } else {
      classification = [0, 0, 1];
    }
    
    samples.push({ x, y: classification });
  }

  const shortSamples = samples.filter(s => s.y[0] === 1);
  const longSamples = samples.filter(s => s.y[1] === 1);
  const sidewaysSamples = samples.filter(s => s.y[2] === 1);
  
  const balancedSamples: { x: number[], y: number[] }[] = [];
  const maxClassCount = Math.max(shortSamples.length, longSamples.length, sidewaysSamples.length);
  const oversample = (arr: {x: number[], y: number[]}[], target: number) => {
    if (arr.length === 0) return [];
    const result = [...arr];
    while (result.length < target) {
      result.push(arr[Math.floor(Math.random() * arr.length)]);
    }
    return result;
  };

  const targetCount = Math.min(maxClassCount, sidewaysSamples.length * 2);
  balancedSamples.push(...oversample(shortSamples, targetCount));
  balancedSamples.push(...oversample(longSamples, targetCount));
  balancedSamples.push(...sidewaysSamples);

  for (let i = balancedSamples.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [balancedSamples[i], balancedSamples[j]] = [balancedSamples[j], balancedSamples[i]];
  }

  const xs = balancedSamples.flatMap(s => s.x);
  const ys = balancedSamples.flatMap(s => s.y);

  return { 
    xs, 
    ys, 
    stats: { 
      total: samples.length, 
      short: shortSamples.length,
      long: longSamples.length,
      sideways: sidewaysSamples.length
    } 
  };
}
