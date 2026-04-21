import * as tf from '@tensorflow/tfjs';

export class GRUModel {
  private model: tf.LayersModel | null = null;
  private windowSize: number;
  private featureCount: number;

  constructor(windowSize: number = 20, featureCount: number = 12) {
    this.windowSize = windowSize;
    this.featureCount = featureCount;
  }

  public predict(input: number[]): number[] {
    return this.predictMultiple(input, 1).mean;
  }

  public predictMultiple(input: number[], passes: number = 1): { mean: number[], std: number[] } {
    if (!this.model) return { mean: [0, 0, 1], std: [0, 0, 0] };
    
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
    const allPredictions: number[][] = [];

    for (let i = 0; i < passes; i++) {
      // Use training: true to enable dropout during inference (MC Dropout)
      const pred = this.model.apply(inputTensor, { training: passes > 1 }) as tf.Tensor;
      allPredictions.push(Array.from(pred.dataSync() as Float32Array));
      pred.dispose();
    }
    
    inputTensor.dispose();

    const mean = [0, 0, 0];
    const std = [0, 0, 0];

    for (let c = 0; c < 3; c++) {
      const classPredictions = allPredictions.map(p => p[c]);
      mean[c] = classPredictions.reduce((a, b) => a + b, 0) / passes;
      if (passes > 1) {
        const variance = classPredictions.reduce((a, b) => a + Math.pow(b - mean[c], 2), 0) / passes;
        std[c] = Math.sqrt(variance);
      }
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
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });
    return gru;
  }
}
