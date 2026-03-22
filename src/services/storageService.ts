import { GRUModel } from './modelService';
import { logger } from './loggerService';

export interface ModelPair {
  name: string;
  timestamp: number;
}

const STORAGE_KEY = 'saved_model_pairs';

export function getSavedModelPairs(): ModelPair[] {
  const saved = localStorage.getItem(STORAGE_KEY);
  return saved ? JSON.parse(saved) : [];
}

export async function saveModelPair(name: string, model1h: GRUModel, model4h: GRUModel) {
  try {
    const id = name.replace(/\s+/g, '_').toLowerCase();
    await model1h.save(`model_pair_${id}_1h`);
    await model4h.save(`model_pair_${id}_4h`);

    const pairs = getSavedModelPairs();
    const newPair: ModelPair = { name, timestamp: Date.now() };
    
    // Replace if exists
    const existingIdx = pairs.findIndex(p => p.name === name);
    if (existingIdx !== -1) {
      pairs[existingIdx] = newPair;
    } else {
      pairs.push(newPair);
    }

    localStorage.setItem(STORAGE_KEY, JSON.stringify(pairs));
    logger.success(`Model pair "${name}" saved successfully.`);
  } catch (err) {
    logger.error(`Failed to save model pair: ${err}`);
    throw err;
  }
}

export async function loadModelPair(name: string): Promise<{ model1h: GRUModel, model4h: GRUModel }> {
  try {
    const id = name.replace(/\s+/g, '_').toLowerCase();
    const model1h = await GRUModel.load(`model_pair_${id}_1h`);
    const model4h = await GRUModel.load(`model_pair_${id}_4h`);
    logger.success(`Model pair "${name}" loaded successfully.`);
    return { model1h, model4h };
  } catch (err) {
    logger.error(`Failed to load model pair: ${err}`);
    throw err;
  }
}

export function deleteModelPair(name: string) {
  const id = name.replace(/\s+/g, '_').toLowerCase();
  localStorage.removeItem(`model_pair_${id}_1h`);
  localStorage.removeItem(`model_pair_${id}_1h_metadata`);
  localStorage.removeItem(`model_pair_${id}_4h`);
  localStorage.removeItem(`model_pair_${id}_4h_metadata`);
  
  // Also remove the tfjs keys from localStorage
  // tfjs uses keys like 'tensorflowjs_models/model_pair_.../model_topology'
  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);
    if (key && key.includes(`model_pair_${id}`)) {
      localStorage.removeItem(key);
      i--; // Adjust index after removal
    }
  }

  const pairs = getSavedModelPairs().filter(p => p.name !== name);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(pairs));
  logger.info(`Model pair "${name}" deleted.`);
}
