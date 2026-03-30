import { GRUModel } from './modelService';
import { logger } from './loggerService';
import { GitHubConfig, uploadToGitHub, deleteFromGitHub, fetchFromGitHub, listFromGitHub } from './githubService';

export interface ModelPair {
  name: string;
  timestamp: number;
  onGitHub?: boolean;
}

const STORAGE_KEY = 'saved_model_pairs';

export function getSavedModelPairs(): ModelPair[] {
  const saved = localStorage.getItem(STORAGE_KEY);
  return saved ? JSON.parse(saved) : [];
}

export async function syncModelsFromGitHub(config: GitHubConfig): Promise<ModelPair[]> {
  try {
    const contents = await listFromGitHub(config);
    const githubDirs = contents.filter(item => item.type === 'dir');
    
    const localPairs = getSavedModelPairs();
    const updatedPairs = [...localPairs];

    for (const dir of githubDirs) {
      const id = dir.name;
      // Check if this ID matches any local pair
      const existingIdx = updatedPairs.findIndex(p => p.name.replace(/\s+/g, '_').toLowerCase() === id);
      
      if (existingIdx !== -1) {
        updatedPairs[existingIdx].onGitHub = true;
      } else {
        // Create a new entry for the GitHub-only model
        updatedPairs.push({
          name: id.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' '),
          timestamp: Date.now(), // We don't know the original timestamp easily
          onGitHub: true
        });
      }
    }

    localStorage.setItem(STORAGE_KEY, JSON.stringify(updatedPairs));
    logger.success(`Synced ${githubDirs.length} models from GitHub.`);
    return updatedPairs;
  } catch (err) {
    logger.error(`Failed to sync models from GitHub: ${err}`);
    throw err;
  }
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
      pairs[existingIdx] = { ...pairs[existingIdx], ...newPair };
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

export async function uploadModelPairToGitHub(name: string, config: GitHubConfig) {
  try {
    const id = name.replace(/\s+/g, '_').toLowerCase();
    const modelConfig = { ...config, path: `${config.path}/${id}` };
    
    const { model1h, model4h } = await loadModelPair(name);

    const artifacts1h = await model1h.getArtifacts();
    const artifacts4h = await model4h.getArtifacts();

    const metadata1h = localStorage.getItem(`model_pair_${id}_1h_metadata`);
    const metadata4h = localStorage.getItem(`model_pair_${id}_4h_metadata`);

    await uploadToGitHub(modelConfig, `1h.json`, JSON.stringify(artifacts1h), `Upload model 1h: ${name}`);
    await uploadToGitHub(modelConfig, `4h.json`, JSON.stringify(artifacts4h), `Upload model 4h: ${name}`);
    if (metadata1h) await uploadToGitHub(modelConfig, `1h_metadata.json`, metadata1h, `Upload metadata 1h: ${name}`);
    if (metadata4h) await uploadToGitHub(modelConfig, `4h_metadata.json`, metadata4h, `Upload metadata 4h: ${name}`);

    const pairs = getSavedModelPairs();
    const idx = pairs.findIndex(p => p.name === name);
    if (idx !== -1) {
      pairs[idx].onGitHub = true;
      localStorage.setItem(STORAGE_KEY, JSON.stringify(pairs));
    }

    logger.success(`Model pair "${name}" uploaded to GitHub.`);
  } catch (err) {
    logger.error(`Failed to upload model pair to GitHub: ${err}`);
    throw err;
  }
}

export async function deleteModelPairFromGitHub(name: string, config: GitHubConfig) {
  try {
    const id = name.replace(/\s+/g, '_').toLowerCase();
    const modelConfig = { ...config, path: `${config.path}/${id}` };
    
    await deleteFromGitHub(modelConfig, `1h.json`, `Delete model 1h: ${name}`);
    await deleteFromGitHub(modelConfig, `4h.json`, `Delete model 4h: ${name}`);
    await deleteFromGitHub(modelConfig, `1h_metadata.json`, `Delete metadata 1h: ${name}`);
    await deleteFromGitHub(modelConfig, `4h_metadata.json`, `Delete metadata 4h: ${name}`);

    const pairs = getSavedModelPairs();
    const idx = pairs.findIndex(p => p.name === name);
    if (idx !== -1) {
      pairs[idx].onGitHub = false;
      localStorage.setItem(STORAGE_KEY, JSON.stringify(pairs));
    }

    logger.success(`Model pair "${name}" deleted from GitHub.`);
  } catch (err) {
    logger.error(`Failed to delete model pair from GitHub: ${err}`);
    throw err;
  }
}

export async function loadModelPairFromGitHub(name: string, config: GitHubConfig): Promise<{ model1h: GRUModel, model4h: GRUModel }> {
  try {
    const id = name.replace(/\s+/g, '_').toLowerCase();
    const modelConfig = { ...config, path: `${config.path}/${id}` };

    const artifacts1hStr = await fetchFromGitHub(modelConfig, `1h.json`);
    const artifacts4hStr = await fetchFromGitHub(modelConfig, `4h.json`);
    const metadata1hStr = await fetchFromGitHub(modelConfig, `1h_metadata.json`);
    const metadata4hStr = await fetchFromGitHub(modelConfig, `4h_metadata.json`);

    const artifacts1h = JSON.parse(artifacts1hStr);
    const artifacts4h = JSON.parse(artifacts4hStr);
    const metadata1h = JSON.parse(metadata1hStr);
    const metadata4h = JSON.parse(metadata4hStr);

    const model1h = await GRUModel.loadFromArtifacts(artifacts1h, metadata1h);
    const model4h = await GRUModel.loadFromArtifacts(artifacts4h, metadata4h);

    logger.success(`Model pair "${name}" loaded from GitHub.`);
    return { model1h, model4h };
  } catch (err) {
    logger.error(`Failed to load model pair from GitHub: ${err}`);
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

export async function deleteModelPair(name: string) {
  const id = name.replace(/\s+/g, '_').toLowerCase();
  
  // Remove from IndexedDB
  await GRUModel.remove(`model_pair_${id}_1h`);
  await GRUModel.remove(`model_pair_${id}_4h`);
  
  // Also clean up any legacy localStorage entries
  localStorage.removeItem(`model_pair_${id}_1h`);
  localStorage.removeItem(`model_pair_${id}_1h_metadata`);
  localStorage.removeItem(`model_pair_${id}_4h`);
  localStorage.removeItem(`model_pair_${id}_4h_metadata`);
  
  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);
    if (key && key.includes(`model_pair_${id}`)) {
      localStorage.removeItem(key);
      i--;
    }
  }

  const pairs = getSavedModelPairs().filter(p => p.name !== name);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(pairs));
  logger.info(`Model pair "${name}" deleted.`);
}
