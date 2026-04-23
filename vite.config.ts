import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import fs from 'fs';
import {defineConfig, loadEnv} from 'vite';

export default defineConfig(({mode}) => {
  const env = loadEnv(mode, '.', '');
  
  // Custom plugin to copy xgboost.wasm to public directory
  const copyXGBoostWasm = () => ({
    name: 'copy-xgboost-wasm',
    buildStart() {
      const source = path.resolve('node_modules', 'ml-xgboost', 'dist', 'wasm', 'xgboost.wasm');
      const destDir = path.resolve('public');
      const dest = path.resolve(destDir, 'xgboost.wasm');

      if (!fs.existsSync(destDir)) {
        fs.mkdirSync(destDir, { recursive: true });
      }

      if (fs.existsSync(source)) {
        try {
          fs.copyFileSync(source, dest);
          console.log('Successfully copied xgboost.wasm to public/');
        } catch (err) {
          console.error('Failed to copy xgboost.wasm:', err);
        }
      } else {
        console.warn('XGBoost WASM source not found at:', source);
      }
    }
  });

  return {
    plugins: [react(), tailwindcss(), copyXGBoostWasm()],
    define: {
      'process.env.GEMINI_API_KEY': JSON.stringify(env.GEMINI_API_KEY),
      'global': 'window',
      'process.env': '{}',
    },
    build: {
      commonjsOptions: {
        transformMixedEsModules: true,
      },
    },
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      },
    },
    server: {
      // HMR is disabled in AI Studio via DISABLE_HMR env var.
      // Do not modifyâfile watching is disabled to prevent flickering during agent edits.
      hmr: process.env.DISABLE_HMR !== 'true',
    },
  };
});
