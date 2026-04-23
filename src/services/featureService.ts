import { Candle } from './dataService';
import { 
  calculateEMA, 
  calculateRSI, 
  calculateBollingerBands, 
  calculateMACD, 
  calculateStochasticRSI, 
  calculateATR, 
  calculateEMACross, 
  calculateOBV, 
  calculateMFI, 
  calculateVolatility, 
  calculateROC, 
  calculateBearishHarami, 
  calculateMarubozu, 
  calculateEngulfing 
} from './indicatorService';

export const INDICATOR_BUFFER_SIZE = 500; // Increased for better stability

export interface FeatureSettings {
  rsi: number;
  ema: number;
  ema9: number;
  bb: number;
  mfi: number;
  volatility: number;
}

export function generateFeatureVector(
  candles: Candle[], 
  settings: FeatureSettings, 
  windowSize: number
): number[] {
  // Use a fixed buffer for all calculations to ensure training, backtest and live alignment
  const buffer = candles.slice(-INDICATOR_BUFFER_SIZE);
  const p = buffer.map(c => c.close);
  const h = buffer.map(c => c.high);
  const l = buffer.map(c => c.low);
  const v = buffer.map(c => c.volume);
  const o = buffer.map(c => c.open);
  const hr = buffer.map(c => new Date(c.time).getHours());
  const d = buffer.map(c => new Date(c.time).getDay());

  const rsi = calculateRSI(p, settings.rsi);
  const ema = calculateEMA(p, settings.ema);
  const ema9 = calculateEMA(p, settings.ema9);
  const cross = calculateEMACross(ema9, ema);
  const bb = calculateBollingerBands(p, settings.bb);
  const macd = calculateMACD(p);
  const stoch = calculateStochasticRSI(rsi);
  const atr = calculateATR(h, l, p);
  const obv = calculateOBV(p, v);
  const mfi = calculateMFI(h, l, p, v, settings.mfi);
  const vol = calculateVolatility(p, settings.volatility);
  const roc = calculateROC(p, 12);
  const harami = calculateBearishHarami(o, p);
  const marubozu = calculateMarubozu(o, h, l, p);
  const engulfing = calculateEngulfing(o, p);

  const normalize = (arr: number[]) => {
    const min = Math.min(...arr);
    const max = Math.max(...arr);
    return arr.map(v => (max === min ? 0.5 : (v - min) / (max - min)));
  };

  const winIdx = Array.from({ length: windowSize }, (_, k) => buffer.length - windowSize + k);
  
  const np = normalize(winIdx.map(idx => p[idx]));
  const nr = winIdx.map(idx => rsi[idx] / 100);
  const ne = normalize(winIdx.map(idx => ema[idx]));
  const nu = normalize(winIdx.map(idx => bb.upper[idx]));
  const nl = normalize(winIdx.map(idx => bb.lower[idx]));
  const nm = normalize(winIdx.map(idx => macd.histogram[idx]));
  const nml = normalize(winIdx.map(idx => macd.macdLine[idx]));
  const nroc = normalize(winIdx.map(idx => roc[idx]));
  const nstoch = winIdx.map(idx => stoch[idx]);
  const na = normalize(winIdx.map(idx => atr[idx]));
  const ne9 = normalize(winIdx.map(idx => ema9[idx]));
  const nobv = normalize(winIdx.map(idx => obv[idx]));
  const nvol = normalize(winIdx.map(idx => vol[idx]));

  const features: number[] = [];
  for (let j = 0; j < windowSize; j++) {
    const hour = hr[winIdx[j]];
    features.push(
      np[j], 
      nr[j], 
      ne[j], 
      nu[j], 
      nl[j],
      nm[j], 
      nml[j], 
      nroc[j], 
      nstoch[j], 
      na[j],
      ne9[j], 
      cross.isBelow[winIdx[j]], 
      cross.isCross[winIdx[j]],
      nobv[j], 
      mfi[winIdx[j]] / 100, 
      nvol[j],
      hour / 24, 
      hour >= 0 && hour <= 9 ? 1 : 0,  // Asia
      hour >= 8 && hour <= 17 ? 1 : 0, // London
      hour >= 13 && hour <= 22 ? 1 : 0, // NY
      d[winIdx[j]] / 7,
      harami[winIdx[j]], 
      marubozu[winIdx[j]], 
      engulfing[winIdx[j]]
    );
  }

  return features;
}

/**
 * Generates a sequence of feature vectors for a range of candles.
 * This ensures that EVERY sample (whether for training, backtesting, or live)
 * is generated using the EXACT same lookback buffer logic.
 */
export function generateFeatureSequence(
  allCandles: Candle[],
  settings: FeatureSettings,
  windowSize: number,
  startIdx: number,
  endIdx: number
): number[][] {
  const sequence: number[][] = [];
  
  for (let i = startIdx; i <= endIdx; i++) {
    // Each sample gets exactly INDICATOR_BUFFER_SIZE candles ending at index i
    const buffer = allCandles.slice(Math.max(0, i - INDICATOR_BUFFER_SIZE + 1), i + 1);
    
    // If we don't have enough data yet, we can't generate a valid vector
    if (buffer.length < Math.max(windowSize, 50)) {
      // Push empty or placeholder if needed, but usually we just skip or expect startIdx to be safe
      continue;
    }
    
    sequence.push(generateFeatureVector(buffer, settings, windowSize));
  }
  
  return sequence;
}
