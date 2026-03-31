import * as tf from '@tensorflow/tfjs';
import { GRUModel } from './modelService';
import { logger } from './loggerService';
import { GitHubConfig, uploadToGitHub, deleteFromGitHub, fetchFromGitHub, listFromGitHub } from './githubService';

export interface ModelPair {
  name: string;
  githubId?: string; // Store the exact ID/folder name found on GitHub
  timestamp: number;
  onGitHub?: boolean;
}

const STORAGE_KEY = 'saved_model_pairs';

function nameToId(name: string): string {
  return name.trim().replace(/\s+/g, '_').toLowerCase();
}

function serializeArtifacts(artifacts: tf.io.ModelArtifacts): string {
  const serialized = { ...artifacts } as any;
  if (artifacts.weightData) {
    const weightData = artifacts.weightData;
    if (Array.isArray(weightData)) {
      serialized.weightData = weightData.map(buffer => {
        const bytes = new Uint8Array(buffer);
        let binary = '';
        for (let i = 0; i < bytes.byteLength; i++) {
          binary += String.fromCharCode(bytes[i]);
        }
        return btoa(binary);
      });
      serialized.isWeightDataArray = true;
    } else {
      const bytes = new Uint8Array(weightData);
      let binary = '';
      for (let i = 0; i < bytes.byteLength; i++) {
        binary += String.fromCharCode(bytes[i]);
      }
      serialized.weightData = btoa(binary);
      serialized.isWeightDataArray = false;
    }
  }
  return JSON.stringify(serialized);
}

function deserializeArtifacts(jsonStr: string): tf.io.ModelArtifacts {
  const parsed = JSON.parse(jsonStr);
  if (parsed.weightData) {
    if (parsed.isWeightDataArray && Array.isArray(parsed.weightData)) {
      parsed.weightData = parsed.weightData.map((base64: string) => {
        const binary = atob(base64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {
          bytes[i] = binary.charCodeAt(i);
        }
        return bytes.buffer;
      });
    } else if (typeof parsed.weightData === 'string') {
      const binary = atob(parsed.weightData);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) {
        bytes[i] = binary.charCodeAt(i);
      }
      parsed.weightData = bytes.buffer;
    }
  }
  return parsed as tf.io.ModelArtifacts;
}

export function getSavedModelPairs(): ModelPair[] {
  const saved = localStorage.getItem(STORAGE_KEY);
  return saved ? JSON.parse(saved) : [];
}

export async function syncModelsFromGitHub(config: GitHubConfig): Promise<ModelPair[]> {
  try {
    const contents = await listFromGitHub(config);
    
    const localPairs = getSavedModelPairs();
    const updatedPairs = [...localPairs];

    // Track discovered IDs to avoid duplicates
    const discoveredIds = new Set<string>();

    // 1. Check if the current directory IS a model folder (New Structure)
    // If it contains 1h.json and 1h_metadata.json, the current path is a model
    const has1h = contents.some(item => item.name === '1h.json');
    const has1hMeta = contents.some(item => item.name === '1h_metadata.json');
    if (has1h && has1hMeta) {
      const pathSegments = config.path.split('/').filter(Boolean);
      if (pathSegments.length > 0) {
        const id = pathSegments[pathSegments.length - 1];
        discoveredIds.add(id);
      }
    }

    // 2. Check for directories (New Structure - Subfolders)
    const githubDirs = contents.filter(item => item.type === 'dir');
    for (const dir of githubDirs) {
      discoveredIds.add(dir.name);
    }

    // 3. Check for files (Old Structure - Flat)
    // Look for [id]_1h_metadata.json
    const githubFiles = contents.filter(item => item.type === 'file');
    for (const file of githubFiles) {
      if (file.name.endsWith('_1h_metadata.json')) {
        const id = file.name.replace('_1h_metadata.json', '');
        discoveredIds.add(id);
      }
    }

    for (const id of discoveredIds) {
      // Check if this ID matches any local pair
      const existingIdx = updatedPairs.findIndex(p => nameToId(p.name) === id.toLowerCase());
      
      if (existingIdx !== -1) {
        updatedPairs[existingIdx].onGitHub = true;
        updatedPairs[existingIdx].githubId = id; // Store the exact case-sensitive ID
      } else {
        // Check if we already added this from GitHub in a previous sync
        const alreadyAdded = updatedPairs.some(p => p.githubId === id);
        if (!alreadyAdded) {
          // Reconstruct a readable name from the ID
          const readableName = id.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
          updatedPairs.push({
            name: readableName,
            githubId: id,
            timestamp: Date.now(),
            onGitHub: true
          });
        }
      }
    }

    localStorage.setItem(STORAGE_KEY, JSON.stringify(updatedPairs));
    logger.success(`Synced ${discoveredIds.size} models from GitHub.`);
    return updatedPairs;
  } catch (err) {
    logger.error(`Failed to sync models from GitHub: ${err}`);
    throw err;
  }
}

export async function saveModelPair(name: string, model1h: GRUModel, model4h: GRUModel) {
  try {
    const id = nameToId(name);
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
    const pairs = getSavedModelPairs();
    const pair = pairs.find(p => p.name === name);
    const id = pair?.githubId || nameToId(name);
    const modelConfig = { ...config, path: config.path ? `${config.path}/${id}` : id };
    
    const { model1h, model4h } = await loadModelPair(name);

    const artifacts1h = await model1h.getArtifacts();
    const artifacts4h = await model4h.getArtifacts();

    const metadata1h = localStorage.getItem(`model_pair_${id}_1h_metadata`);
    const metadata4h = localStorage.getItem(`model_pair_${id}_4h_metadata`);

    await uploadToGitHub(modelConfig, `1h.json`, serializeArtifacts(artifacts1h), `Upload model 1h: ${name}`);
    await uploadToGitHub(modelConfig, `4h.json`, serializeArtifacts(artifacts4h), `Upload model 4h: ${name}`);
    if (metadata1h) await uploadToGitHub(modelConfig, `1h_metadata.json`, metadata1h, `Upload metadata 1h: ${name}`);
    if (metadata4h) await uploadToGitHub(modelConfig, `4h_metadata.json`, metadata4h, `Upload metadata 4h: ${name}`);

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
    const pairs = getSavedModelPairs();
    const pair = pairs.find(p => p.name === name);
    const id = pair?.githubId || nameToId(name);
    const modelConfig = { ...config, path: config.path ? `${config.path}/${id}` : id };
    
    await deleteFromGitHub(modelConfig, `1h.json`, `Delete model 1h: ${name}`);
    await deleteFromGitHub(modelConfig, `4h.json`, `Delete model 4h: ${name}`);
    await deleteFromGitHub(modelConfig, `1h_metadata.json`, `Delete metadata 1h: ${name}`);
    await deleteFromGitHub(modelConfig, `4h_metadata.json`, `Delete metadata 4h: ${name}`);

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
    const pairs = getSavedModelPairs();
    const pair = pairs.find(p => p.name === name);
    const id = pair?.githubId || nameToId(name);
    
    // 1. Try Subfolder Structure (path/id/1h.json)
    const subfolderConfig = { ...config, path: config.path ? `${config.path}/${id}` : id };
    try {
      logger.info(`Trying Step 1: Subfolder structure at "${subfolderConfig.path}/1h.json"`);
      const artifacts1hStr = await fetchFromGitHub(subfolderConfig, `1h.json`);
      const artifacts4hStr = await fetchFromGitHub(subfolderConfig, `4h.json`);
      const metadata1hStr = await fetchFromGitHub(subfolderConfig, `1h_metadata.json`);
      const metadata4hStr = await fetchFromGitHub(subfolderConfig, `4h_metadata.json`);

      const artifacts1h = deserializeArtifacts(artifacts1hStr);
      const artifacts4h = deserializeArtifacts(artifacts4hStr);
      const metadata1h = JSON.parse(metadata1hStr);
      const metadata4h = JSON.parse(metadata4hStr);

      const model1h = await GRUModel.loadFromArtifacts(artifacts1h, metadata1h, `model_pair_${id}_1h`);
      const model4h = await GRUModel.loadFromArtifacts(artifacts4h, metadata4h, `model_pair_${id}_4h`);

      logger.success(`Model pair "${name}" loaded from GitHub (Subfolder).`);
      return { model1h, model4h };
    } catch (err) {
      logger.info(`Step 1 failed: ${err}`);
      // 2. Try Direct Path Structure (path/1h.json)
      // This handles cases where the user points directly to a model folder
      try {
        logger.info(`Trying Step 2: Direct path structure at "${config.path}/1h.json"`);
        const artifacts1hStr = await fetchFromGitHub(config, `1h.json`);
        const artifacts4hStr = await fetchFromGitHub(config, `4h.json`);
        const metadata1hStr = await fetchFromGitHub(config, `1h_metadata.json`);
        const metadata4hStr = await fetchFromGitHub(config, `4h_metadata.json`);

        const artifacts1h = deserializeArtifacts(artifacts1hStr);
        const artifacts4h = deserializeArtifacts(artifacts4hStr);
        const metadata1h = JSON.parse(metadata1hStr);
        const metadata4h = JSON.parse(metadata4hStr);

        const model1h = await GRUModel.loadFromArtifacts(artifacts1h, metadata1h, `model_pair_${id}_1h`);
        const model4h = await GRUModel.loadFromArtifacts(artifacts4h, metadata4h, `model_pair_${id}_4h`);

        logger.success(`Model pair "${name}" loaded from GitHub (Direct Path).`);
        return { model1h, model4h };
      } catch (err2) {
        logger.info(`Step 2 failed: ${err2}`);
        // 3. Try Flat Structure (path/id_1h.json)
        logger.info(`Trying Step 3: Flat structure at "${config.path}/${id}_1h.json"`);
        
        const artifacts1hStr = await fetchFromGitHub(config, `${id}_1h.json`);
        const artifacts4hStr = await fetchFromGitHub(config, `${id}_4h.json`);
        const metadata1hStr = await fetchFromGitHub(config, `${id}_1h_metadata.json`);
        const metadata4hStr = await fetchFromGitHub(config, `${id}_4h_metadata.json`);

        const artifacts1h = deserializeArtifacts(artifacts1hStr);
        const artifacts4h = deserializeArtifacts(artifacts4hStr);
        const metadata1h = JSON.parse(metadata1hStr);
        const metadata4h = JSON.parse(metadata4hStr);

        const model1h = await GRUModel.loadFromArtifacts(artifacts1h, metadata1h, `model_pair_${id}_1h`);
        const model4h = await GRUModel.loadFromArtifacts(artifacts4h, metadata4h, `model_pair_${id}_4h`);

        logger.success(`Model pair "${name}" loaded from GitHub (Flat).`);
        return { model1h, model4h };
      }
    }
  } catch (err) {
    logger.error(`Failed to load model pair from GitHub: ${err}`);
    throw err;
  }
}

export async function loadModelPair(name: string): Promise<{ model1h: GRUModel, model4h: GRUModel }> {
  try {
    const id = nameToId(name);
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
  const id = nameToId(name);
  
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
