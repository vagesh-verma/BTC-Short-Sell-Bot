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
    if (!this.model) return 0;
    
    const inputTensor = tf.tensor3d(input, [1, this.windowSize, this.featureCount]);
    const prediction = this.model.predict(inputTensor) as tf.Tensor;
    const result = prediction.dataSync()[0];
    
    inputTensor.dispose();
    prediction.dispose();
    
    return result;
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
