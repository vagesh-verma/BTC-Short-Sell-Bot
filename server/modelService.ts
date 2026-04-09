import * as tf from '@tensorflow/tfjs';

export class GRUModel {
  private model: tf.LayersModel | null = null;
  private windowSize: number;
  private featureCount: number;

  constructor(windowSize: number = 20, featureCount: number = 12) {
    this.windowSize = windowSize;
    this.featureCount = featureCount;
  }

  public predict(input: number[]): number {
    return this.predictMultiple(input, 1).mean;
  }

  public predictMultiple(input: number[], passes: number = 1): { mean: number, std: number } {
    if (!this.model) return { mean: 0, std: 0 };
    
    // Slice input to match expected shape if necessary
    const expectedSize = this.windowSize * this.featureCount;
    let finalInput = input;
    if (input.length > expectedSize) {
      finalInput = input.slice(0, expectedSize);
    } else if (input.length < expectedSize) {
      // Pad with zeros if too small
      finalInput = [...input, ...new Array(expectedSize - input.length).fill(0)];
    }

    const inputTensor = tf.tensor3d(finalInput, [1, this.windowSize, this.featureCount]);
    const predictions: number[] = [];

    for (let i = 0; i < passes; i++) {
      // Use training: true to enable dropout during inference (MC Dropout)
      const pred = this.model.apply(inputTensor, { training: passes > 1 }) as tf.Tensor;
      predictions.push(pred.dataSync()[0]);
      pred.dispose();
    }
    
    inputTensor.dispose();

    const mean = predictions.reduce((a, b) => a + b, 0) / predictions.length;
    let std = 0;
    if (predictions.length > 1) {
      const variance = predictions.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / predictions.length;
      std = Math.sqrt(variance);
    }
    
    return { mean, std };
  }

  public getWindowSize(): number {
    return this.windowSize;
  }

  public getFeatureCount(): number {
    return this.featureCount;
  }

  public static async loadFromArtifacts(artifacts: any, metadata: { windowSize: number, featureCount: number }): Promise<GRUModel> {
    const gru = new GRUModel(metadata.windowSize, metadata.featureCount);
    
    // If weightData is a string (base64 from client), convert it back to ArrayBuffer
    if (typeof artifacts.weightData === 'string') {
      const buffer = Buffer.from(artifacts.weightData, 'base64');
      artifacts.weightData = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
    }

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
}
