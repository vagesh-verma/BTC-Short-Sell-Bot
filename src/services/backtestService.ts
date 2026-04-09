import { Candle } from './dataService';

export interface Trade {
  type: 'SHORT';
  entryPrice: number;
  exitPrice: number;
  entryTime: number;
  exitTime: number;
  profit: number;
  profitPct: number;
  exitReason: 'TIME' | 'STOP_LOSS' | 'TAKE_PROFIT' | 'TRAILING_STOP' | 'PREDICTION' | 'MANUAL';
  prediction?: number;
  features?: Record<string, number>;
}

export interface BacktestSettings {
  threshold: number;
  exitThreshold: number;
  stopLoss: number; // e.g., 0.01 for 1%
  takeProfit: number; // e.g., 0.02 for 2%
  trailingStopActivation: number; // e.g., 0.01 for 1% profit
  trailingStopOffset: number; // e.g., 0.005 for 0.5% offset
  maxDurationHours: number;
  quantity: number;
  quantityType: 'USD' | 'BTC' | 'LOTS';
  onlyHighVolumeSessions?: boolean;
  useSessionTrading?: boolean;
  asiaStart: number;
  asiaEnd: number;
  nyStart: number;
  nyEnd: number;
  useOnlyCompletedCandles?: boolean;
  minSignalVelocity?: number;
  mcPasses?: number;
  maxUncertainty?: number;
  strategyType?: 'SHORT_BTC' | 'CALL_SPREAD';
  shortCallDelta?: number;
  longCallDelta?: number;
  dailyProfitLimit?: number;
  dailyLossLimit?: number;
}

export interface BacktestResult {
  trades: Trade[];
  totalProfit: number;
  winRate: number;
  equityCurve: { time: number; balance: number }[];
  maxDrawdown: number;
  avgProfit: number;
  avgLoss: number;
  maxProfit: number;
  maxLoss: number;
  candles: Candle[];
  predictions: number[];
}

export function runBacktest(
  candles: Candle[],
  predictions: number[],
  settings: BacktestSettings,
  initialBalance: number = 10000,
  windowSize: number = 20,
  features?: number[][],
  featureNames?: string[],
  uncertainties?: number[]
): BacktestResult {
  const trades: Trade[] = [];
  let balance = initialBalance;
  let peakBalance = initialBalance;
  let maxDrawdown = 0;
  const equityCurve: { time: number; balance: number }[] = [{ time: candles[0].time, balance }];
  
  let activeTrade: {
    entryPrice: number;
    entryTime: number;
    highestProfitPct: number;
    trailingStopPrice: number | null;
    prediction: number;
    features?: Record<string, number>;
  } | null = null;

  for (let i = 0; i < predictions.length; i++) {
    const candle = candles[i + windowSize];
    if (!candle) break;
    
    const prediction = predictions[i];
    const uncertainty = uncertainties ? uncertainties[i] : 0;
    const now = new Date(candle.time);
    const hour = now.getUTCHours();

    // Check for exit first if in trade
    if (activeTrade) {
      const stopLossPrice = activeTrade.entryPrice * (1 + settings.stopLoss);
      const takeProfitPrice = activeTrade.entryPrice * (1 - settings.takeProfit);
      
      let shouldExit = false;
      let exitReason: Trade['exitReason'] = 'TIME';
      let exitPrice = candle.close;

      // Check SL/TP/Trailing Stop
      if (candle.high >= stopLossPrice) {
        shouldExit = true;
        exitReason = 'STOP_LOSS';
        exitPrice = stopLossPrice;
      } else if (candle.low <= takeProfitPrice) {
        shouldExit = true;
        exitReason = 'TAKE_PROFIT';
        exitPrice = takeProfitPrice;
      } else if (activeTrade.trailingStopPrice !== null && candle.high >= activeTrade.trailingStopPrice) {
        shouldExit = true;
        exitReason = 'TRAILING_STOP';
        exitPrice = activeTrade.trailingStopPrice;
      } else if (prediction < settings.exitThreshold) {
        shouldExit = true;
        exitReason = 'PREDICTION';
        exitPrice = candle.close;
      } else {
        const tradeDuration = (candle.time - activeTrade.entryTime) / (1000 * 60 * 60);
        if (tradeDuration >= settings.maxDurationHours) {
          shouldExit = true;
          exitReason = 'TIME';
          exitPrice = candle.close;
        }
      }

      if (shouldExit) {
        const currentProfitPct = (activeTrade.entryPrice - exitPrice) / activeTrade.entryPrice;
        
        let profit = 0;
        if (settings.quantityType === 'USD') {
          profit = settings.quantity * currentProfitPct;
        } else if (settings.quantityType === 'BTC') {
          // BTC quantity: profit = quantity * (entryPrice - exitPrice)
          profit = settings.quantity * (activeTrade.entryPrice - exitPrice);
        } else {
          // LOTS: 1 lot = 0.001 BTC
          const btcQuantity = settings.quantity * 0.001;
          profit = btcQuantity * (activeTrade.entryPrice - exitPrice);
        }
        
        balance += profit;
        
        trades.push({
          type: 'SHORT',
          entryPrice: activeTrade.entryPrice,
          exitPrice: exitPrice,
          entryTime: activeTrade.entryTime,
          exitTime: candle.time,
          profit,
          profitPct: currentProfitPct * 100,
          exitReason,
          prediction: activeTrade.prediction,
          features: activeTrade.features
        });
        
        activeTrade = null;
      } else {
        // Update Trailing Stop
        const currentProfitPct = (activeTrade.entryPrice - candle.low) / activeTrade.entryPrice;
        activeTrade.highestProfitPct = Math.max(activeTrade.highestProfitPct, currentProfitPct);

        if (activeTrade.highestProfitPct >= settings.trailingStopActivation) {
          const newTrailingStop = candle.low * (1 + settings.trailingStopOffset);
          if (activeTrade.trailingStopPrice === null || newTrailingStop < activeTrade.trailingStopPrice) {
            activeTrade.trailingStopPrice = newTrailingStop;
          }
        }
      }
    }

    // Check for entry if not in trade
    const velocity = i > 0 ? (prediction - predictions[i - 1]) : 0;
    const minVelocity = settings.minSignalVelocity || 0;
    const maxUncertainty = settings.maxUncertainty || 1;
    const mcPasses = settings.mcPasses || 1;

    const passesFilter = mcPasses <= 1 || uncertainty <= maxUncertainty;

    if (!activeTrade && prediction > settings.threshold && velocity >= minVelocity && passesFilter) {
      let canTrade = true;
      const hour = new Date(candle.time).getUTCHours();

      if (settings.useSessionTrading) {
        const isAsia = hour >= settings.asiaStart && hour < settings.asiaEnd;
        const isNY = hour >= settings.nyStart && hour < settings.nyEnd;
        canTrade = isAsia || isNY;
      } else if (settings.onlyHighVolumeSessions) {
        const isAsia = hour >= 0 && hour <= 9;
        const isLondon = hour >= 8 && hour <= 17;
        const isNY = hour >= 13 && hour <= 22;
        canTrade = isAsia || isLondon || isNY;
      }

      if (canTrade) {
        const tradeFeatures: Record<string, number> = {};
        if (features && features[i] && featureNames) {
          featureNames.forEach((name, idx) => {
            tradeFeatures[name] = features[i][idx];
          });
        }

        activeTrade = {
          entryPrice: candle.close,
          entryTime: candle.time,
          highestProfitPct: 0,
          trailingStopPrice: null,
          prediction: prediction,
          features: tradeFeatures
        };
      }
    }
    
    if (balance > peakBalance) peakBalance = balance;
    const drawdown = (peakBalance - balance) / peakBalance;
    if (drawdown > maxDrawdown) maxDrawdown = drawdown;

    equityCurve.push({ time: candle.time, balance });
  }

  const winningTrades = trades.filter(t => t.profit > 0);
  const losingTrades = trades.filter(t => t.profit <= 0);
  
  const wins = winningTrades.length;
  const winRate = trades.length > 0 ? (wins / trades.length) * 100 : 0;

  const avgProfit = winningTrades.length > 0 
    ? winningTrades.reduce((acc, t) => acc + t.profit, 0) / winningTrades.length 
    : 0;
  const avgLoss = losingTrades.length > 0 
    ? losingTrades.reduce((acc, t) => acc + t.profit, 0) / losingTrades.length 
    : 0;

  const maxProfit = trades.length > 0 ? Math.max(...trades.map(t => t.profit)) : 0;
  const maxLoss = trades.length > 0 ? Math.min(...trades.map(t => t.profit)) : 0;

  return {
    trades,
    totalProfit: balance - initialBalance,
    winRate,
    equityCurve,
    maxDrawdown: maxDrawdown * 100,
    avgProfit,
    avgLoss,
    maxProfit,
    maxLoss,
    candles: candles.slice(windowSize),
    predictions
  };
}
