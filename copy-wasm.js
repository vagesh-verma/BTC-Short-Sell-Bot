import fs from 'fs';
import path from 'path';

const source = path.resolve('node_modules', 'ml-xgboost', 'dist', 'wasm', 'xgboost.wasm');
const destDir = path.resolve('public');
const dest = path.resolve(destDir, 'xgboost.wasm');

if (!fs.existsSync(destDir)) {
  fs.mkdirSync(destDir, { recursive: true });
}

if (fs.existsSync(source)) {
  fs.copyFileSync(source, dest);
  console.log('Successfully copied xgboost.wasm to public/');
} else {
  console.error('Source file not found:', source);
}
