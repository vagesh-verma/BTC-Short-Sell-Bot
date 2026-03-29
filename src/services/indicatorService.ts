export function calculateEMA(prices: number[], period: number): number[] {
  if (prices.length === 0) return [];
  const k = 2 / (period + 1);
  const ema: number[] = [prices[0]];
  for (let i = 1; i < prices.length; i++) {
    ema.push(prices[i] * k + ema[i - 1] * (1 - k));
  }
  return ema;
}

export function calculateRSI(prices: number[], period: number = 14): number[] {
  if (prices.length <= period) return new Array(prices.length).fill(50);
  const rsi: number[] = new Array(prices.length).fill(50);
  let gains = 0;
  let losses = 0;

  for (let i = 1; i <= period; i++) {
    const diff = prices[i] - prices[i - 1];
    if (diff >= 0) gains += diff;
    else losses -= diff;
  }

  let avgGain = gains / period;
  let avgLoss = losses / period;

  for (let i = period + 1; i < prices.length; i++) {
    const diff = prices[i] - prices[i - 1];
    const gain = diff >= 0 ? diff : 0;
    const loss = diff < 0 ? -diff : 0;

    avgGain = (avgGain * (period - 1) + gain) / period;
    avgLoss = (avgLoss * (period - 1) + loss) / period;

    const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
    rsi[i] = 100 - 100 / (1 + rs);
  }

  return rsi;
}

export function calculateBollingerBands(prices: number[], period: number = 20, stdDev: number = 2) {
  const upper: number[] = [];
  const middle: number[] = [];
  const lower: number[] = [];

  for (let i = 0; i < prices.length; i++) {
    if (i < period - 1) {
      upper.push(prices[i]);
      middle.push(prices[i]);
      lower.push(prices[i]);
      continue;
    }

    const slice = prices.slice(i - period + 1, i + 1);
    const avg = slice.reduce((a, b) => a + b, 0) / period;
    const squareDiffs = slice.map(p => Math.pow(p - avg, 2));
    const variance = squareDiffs.reduce((a, b) => a + b, 0) / period;
    const sd = Math.sqrt(variance);

    middle.push(avg);
    upper.push(avg + stdDev * sd);
    lower.push(avg - stdDev * sd);
  }

  return { upper, middle, lower };
}

export function calculateMACD(prices: number[], fast: number = 12, slow: number = 26, signal: number = 9) {
  const fastEMA = calculateEMA(prices, fast);
  const slowEMA = calculateEMA(prices, slow);
  const macdLine = fastEMA.map((f, i) => f - slowEMA[i]);
  const signalLine = calculateEMA(macdLine, signal);
  const histogram = macdLine.map((m, i) => m - signalLine[i]);
  
  return { macdLine, signalLine, histogram };
}

export function calculateStochasticRSI(rsi: number[], period: number = 14) {
  const stochRsi: number[] = new Array(rsi.length).fill(0.5);
  
  for (let i = period; i < rsi.length; i++) {
    const slice = rsi.slice(i - period + 1, i + 1);
    const minRsi = Math.min(...slice);
    const maxRsi = Math.max(...slice);
    
    if (maxRsi === minRsi) {
      stochRsi[i] = 0.5;
    } else {
      stochRsi[i] = (rsi[i] - minRsi) / (maxRsi - minRsi);
    }
  }
  
  return stochRsi;
}

export function calculateATR(highs: number[], lows: number[], closes: number[], period: number = 14): number[] {
  const tr: number[] = [highs[0] - lows[0]];
  for (let i = 1; i < highs.length; i++) {
    const hl = highs[i] - lows[i];
    const hpc = Math.abs(highs[i] - closes[i - 1]);
    const lpc = Math.abs(lows[i] - closes[i - 1]);
    tr.push(Math.max(hl, hpc, lpc));
  }
  
  const atr: number[] = new Array(tr.length).fill(tr[0]);
  for (let i = 1; i < tr.length; i++) {
    if (i < period) {
      atr[i] = tr.slice(0, i + 1).reduce((a, b) => a + b, 0) / (i + 1);
    } else {
      atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period;
    }
  }
  
  return atr;
}

export function calculateEMACross(shortEMA: number[], longEMA: number[]) {
  const isBelow: number[] = [];
  const isCross: number[] = [];

  for (let i = 0; i < shortEMA.length; i++) {
    const below = shortEMA[i] < longEMA[i] ? 1 : 0;
    isBelow.push(below);

    if (i === 0) {
      isCross.push(0);
    } else {
      const prevBelow = shortEMA[i - 1] < longEMA[i - 1] ? 1 : 0;
      // Cross occurs if the "below" state changed
      isCross.push(below !== prevBelow ? 1 : 0);
    }
  }

  return { isBelow, isCross };
}

export function calculateOBV(prices: number[], volumes: number[]): number[] {
  if (prices.length === 0) return [];
  const obv: number[] = [0];
  for (let i = 1; i < prices.length; i++) {
    if (prices[i] > prices[i - 1]) {
      obv.push(obv[i - 1] + volumes[i]);
    } else if (prices[i] < prices[i - 1]) {
      obv.push(obv[i - 1] - volumes[i]);
    } else {
      obv.push(obv[i - 1]);
    }
  }
  return obv;
}

export function calculateMFI(highs: number[], lows: number[], closes: number[], volumes: number[], period: number = 14): number[] {
  const typicalPrices = highs.map((h, i) => (h + lows[i] + closes[i]) / 3);
  const moneyFlow = typicalPrices.map((tp, i) => tp * volumes[i]);
  const mfi: number[] = new Array(closes.length).fill(50);

  for (let i = 1; i < closes.length; i++) {
    if (i < period) continue;

    let posFlow = 0;
    let negFlow = 0;

    for (let j = i - period + 1; j <= i; j++) {
      if (typicalPrices[j] > typicalPrices[j - 1]) {
        posFlow += moneyFlow[j];
      } else if (typicalPrices[j] < typicalPrices[j - 1]) {
        negFlow += moneyFlow[j];
      }
    }

    const moneyRatio = negFlow === 0 ? 100 : posFlow / negFlow;
    mfi[i] = 100 - (100 / (1 + moneyRatio));
  }

  return mfi;
}

export function calculateVolatility(prices: number[], period: number = 20): number[] {
  const volatility: number[] = new Array(prices.length).fill(0);
  for (let i = period; i < prices.length; i++) {
    const slice = prices.slice(i - period + 1, i + 1);
    const returns = [];
    for (let j = 1; j < slice.length; j++) {
      returns.push(Math.log(slice[j] / slice[j - 1]));
    }
    const avg = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance = returns.reduce((a, b) => a + Math.pow(b - avg, 2), 0) / returns.length;
    volatility[i] = Math.sqrt(variance);
  }
  return volatility;
}

export function calculateBearishHarami(opens: number[], prices: number[]): number[] {
  const result: number[] = new Array(prices.length).fill(0);
  for (let i = 1; i < prices.length; i++) {
    const prevBullish = prices[i - 1] > opens[i - 1];
    const currBearish = prices[i] < opens[i];
    const isHarami = prevBullish && currBearish && 
                     opens[i] <= prices[i - 1] && 
                     prices[i] >= opens[i - 1];
    result[i] = isHarami ? 1 : 0;
  }
  return result;
}

export function calculateMarubozu(opens: number[], highs: number[], lows: number[], prices: number[]): number[] {
  const result: number[] = new Array(prices.length).fill(0);
  for (let i = 0; i < prices.length; i++) {
    const bodySize = Math.abs(prices[i] - opens[i]);
    const totalSize = highs[i] - lows[i];
    if (totalSize > 0 && (bodySize / totalSize) > 0.95) {
      result[i] = prices[i] > opens[i] ? 1 : -1; // 1 for bullish, -1 for bearish
    }
  }
  return result;
}

export function calculateEngulfing(opens: number[], prices: number[]): number[] {
  const result: number[] = new Array(prices.length).fill(0);
  for (let i = 1; i < prices.length; i++) {
    const prevBearish = prices[i - 1] < opens[i - 1];
    const prevBullish = prices[i - 1] > opens[i - 1];
    const currBearish = prices[i] < opens[i];
    const currBullish = prices[i] > opens[i];

    // Bullish Engulfing
    if (prevBearish && currBullish && opens[i] < prices[i - 1] && prices[i] > opens[i - 1]) {
      result[i] = 1;
    }
    // Bearish Engulfing
    else if (prevBullish && currBearish && opens[i] > prices[i - 1] && prices[i] < opens[i - 1]) {
      result[i] = -1;
    }
  }
  return result;
}
