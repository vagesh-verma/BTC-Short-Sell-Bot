import { Candle } from './dataService';

export interface Trade {
  type: 'LONG' | 'SHORT' | 'CALL_SPREAD' | 'SHORT_CALL' | 'PUT_SPREAD' | 'SHORT_PUT';
  entryPrice: number;
  exitPrice: number;
  entryTime: number;
  exitTime: number;
  profit: number;
  profitPct: number;
  exitReason: 'TIME' | 'STOP_LOSS' | 'TAKE_PROFIT' | 'TRAILING_STOP' | 'PREDICTION' | 'MANUAL';
  prediction?: number[];
  features?: Record<string, number>;
}

export interface BacktestSettings {
  shortThreshold: number;
  longThreshold: number;
  xgShortThreshold?: number;
  xgLongThreshold?: number;
  xgShortExitThreshold?: number;
  xgLongExitThreshold?: number;
  shortExitThreshold: number;
  longExitThreshold: number;
  biasThreshold?: number;
  stopLoss: number; // e.g., 0.01 for 1%
  takeProfit: number; // e.g., 0.02 for 2%
  trailingStopActivation: number; // e.g., 0.01 for 1% profit
  trailingStopOffset: number; // e.g., 0.005 for 0.5% offset
  maxDurationHours: number;
  quantity: number;
  quantityType: 'USD' | 'BTC' | 'LOTS';
  labelDropThreshold?: number;
  labelLongThreshold?: number;
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
  strategyType?: 'SHORT_BTC' | 'LONG_BTC' | 'BOTH' | 'CALL_SPREAD' | 'SHORT_CALL' | 'PUT_SPREAD' | 'SHORT_PUT';
  shortCallDelta?: number;
  longCallDelta?: number;
  shortPutDelta?: number;
  longPutDelta?: number;
  dailyProfitLimit?: number;
  dailyLossLimit?: number;
  useDRLConfluence?: boolean;
  useDRLOnly?: boolean;
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
  confusionMatrix?: {
    short: { tp: number; fp: number; fn: number; tn: number; precision: number; recall: number };
    long: { tp: number; fp: number; fn: number; tn: number; precision: number; recall: number };
  };
  candles: Candle[];
  predictions: number[][];
  xgPredictions?: number[][];
  drlActions?: number[];
}

export function runBacktest(
  fullCandles: Candle[],
  predictions: number[][],
  settings: BacktestSettings,
  initialBalance: number = 10000,
  startIdx: number = 0,
  features?: number[][],
  featureNames?: string[],
  uncertainties?: number[][],
  xgPredictions?: number[][],
  drlActions?: number[]
): BacktestResult {
  const trades: Trade[] = [];
  let balance = initialBalance;
  let peakBalance = initialBalance;
  let maxDrawdown = 0;

  // Confusion Matrix Stats
  let shortTP = 0, shortFP = 0, shortFN = 0, shortTN = 0;
  let longTP = 0, longFP = 0, longFN = 0, longTN = 0;
  
  const dropThr = settings.labelDropThreshold || (settings.takeProfit * 100);
  const longThr = settings.labelLongThreshold || (settings.takeProfit * 100);
  const lookahead = settings.maxDurationHours || 12;

  // Actual labels determination (ground truth)
  const actualLabels: number[][] = []; // [isShort, isLong, isSideways]
  for (let i = startIdx; i < fullCandles.length; i++) {
    const currentPrice = fullCandles[i].close;
    const sTP = currentPrice * (1 - (dropThr / 100));
    const sSL = currentPrice * (1 + (dropThr / 100));
    const lTP = currentPrice * (1 + (longThr / 100));
    const lSL = currentPrice * (1 - (longThr / 100));

    let actualShort = false;
    let actualLong = false;
    const lookLimit = Math.min(i + lookahead, fullCandles.length - 1);

    for (let j = i + 1; j <= lookLimit; j++) {
      const h = fullCandles[j].high;
      const l = fullCandles[j].low;
      if (l <= sTP && h < sSL) {
        actualShort = true;
        break;
      }
      if (h >= sSL) break;
    }

    if (!actualShort) {
      for (let j = i + 1; j <= lookLimit; j++) {
        const h = fullCandles[j].high;
        const l = fullCandles[j].low;
        if (h >= lTP && l > lSL) {
          actualLong = true;
          break;
        }
        if (l <= lSL) break;
      }
    }
    actualLabels.push([actualShort ? 1 : 0, actualLong ? 1 : 0]);
  }
  
  // We start from startIdx because predictions[0] corresponds to fullCandles[startIdx]
  const equityCurve: { time: number; balance: number }[] = [{ time: fullCandles[startIdx]?.time || 0, balance }];
  
    let activeTrade: {
      type: Trade['type'];
      entryPrice: number;
      entryTime: number;
      highestProfitPct: number;
      trailingStopPrice: number | null;
      prediction: number[];
      features?: Record<string, number>;
    } | null = null;

  for (let i = startIdx; i < fullCandles.length; i++) {
    const candle = fullCandles[i];
    const predictionIdx = i - startIdx;
    
    // DRL Action if features provided
    let currentDrlAction = 2; // Default NEUTRAL
    if (features && features[predictionIdx]) {
      // Note: In real life we'd need drlService, here we assume third-party provided it in a new arg
      // For now let's skip the DRL confluence calculation during backtest unless we pass it specifically
    }
    
    // If we run out of predictions before candles, stop
    if (predictionIdx >= predictions.length) break;
    
    const predictionFull = predictions[predictionIdx];
    if (!predictionFull) continue;

    const xgPredictionFull = (xgPredictions && xgPredictions[predictionIdx]) ? xgPredictions[predictionIdx] : [1, 1, 0];
    
    const shortProb = predictionFull[0];
    const longProb = predictionFull[1];
    const xgShortProb = xgPredictionFull[0];
    const xgLongProb = xgPredictionFull[1];

    // Confusion Matrix Logic: Requires both to pass thresholds if configured
    const predShort = (shortProb * 100 > settings.shortThreshold) && (xgShortProb * 100 >= (settings.xgShortThreshold ?? 0));
    const predLong = (longProb * 100 > settings.longThreshold) && (xgLongProb * 100 >= (settings.xgLongThreshold ?? 0));
    const groundTruth = actualLabels[predictionIdx];

    if (groundTruth) {
      const isActualShort = groundTruth[0] === 1;
      const isActualLong = groundTruth[1] === 1;

      if (predShort) {
        if (isActualShort) shortTP++;
        else shortFP++;
      } else {
        if (isActualShort) shortFN++;
        else shortTN++;
      }

      if (predLong) {
        if (isActualLong) longTP++;
        else longFP++;
      } else {
        if (isActualLong) longFN++;
        else longTN++;
      }
    }
    
    const uncertaintyFull = (uncertainties && uncertainties[predictionIdx]) ? uncertainties[predictionIdx] : [0, 0, 0];
    const uncertaintyShort = uncertaintyFull[0];
    const uncertaintyLong = uncertaintyFull[1];

    const now = new Date(candle.time);
    const hour = now.getUTCHours();

    // Check for exit first if in trade
    if (activeTrade) {
      const isShort = activeTrade.type === 'SHORT' || activeTrade.type === 'CALL_SPREAD' || activeTrade.type === 'SHORT_CALL';
      const stopLossPrice = isShort 
        ? activeTrade.entryPrice * (1 + settings.stopLoss)
        : activeTrade.entryPrice * (1 - settings.stopLoss);
      const takeProfitPrice = isShort
        ? activeTrade.entryPrice * (1 - settings.takeProfit)
        : activeTrade.entryPrice * (1 + settings.takeProfit);
      
      let shouldExit = false;
      let exitReason: Trade['exitReason'] = 'TIME';
      let exitPrice = candle.close;

      // Check SL/TP/Trailing Stop
      if (isShort) {
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
        } else if (shortProb * 100 < settings.shortExitThreshold || (xgPredictions && xgShortProb * 100 < (settings.xgShortExitThreshold ?? 0))) {
          shouldExit = true;
          exitReason = 'PREDICTION';
          exitPrice = candle.close;
        }
      } else { // LONG
        if (candle.low <= stopLossPrice) {
          shouldExit = true;
          exitReason = 'STOP_LOSS';
          exitPrice = stopLossPrice;
        } else if (candle.high >= takeProfitPrice) {
          shouldExit = true;
          exitReason = 'TAKE_PROFIT';
          exitPrice = takeProfitPrice;
        } else if (activeTrade.trailingStopPrice !== null && candle.low <= activeTrade.trailingStopPrice) {
          shouldExit = true;
          exitReason = 'TRAILING_STOP';
          exitPrice = activeTrade.trailingStopPrice;
        } else if (longProb * 100 < settings.longExitThreshold || (xgPredictions && xgLongProb * 100 < (settings.xgLongExitThreshold ?? 0))) {
          shouldExit = true;
          exitReason = 'PREDICTION';
          exitPrice = candle.close;
        }
      }

      if (!shouldExit) {
        const tradeDuration = (candle.time - activeTrade.entryTime) / (1000 * 60 * 60);
        if (tradeDuration >= settings.maxDurationHours) {
          shouldExit = true;
          exitReason = 'TIME';
          exitPrice = candle.close;
        }
      }

      if (shouldExit) {
        const underlyingProfitPct = isShort
          ? (activeTrade.entryPrice - exitPrice) / activeTrade.entryPrice
          : (exitPrice - activeTrade.entryPrice) / activeTrade.entryPrice;
        let currentProfitPct = underlyingProfitPct;

        // Adjust profit for options strategies based on delta approximation
        if (activeTrade.type === 'SHORT_CALL') {
          const delta = settings.shortCallDelta || 0.3;
          currentProfitPct = underlyingProfitPct * delta;
        } else if (activeTrade.type === 'CALL_SPREAD') {
          const shortDelta = settings.shortCallDelta || 0.3;
          const longDelta = settings.longCallDelta || 0.1;
          const netDelta = Math.max(0, shortDelta - longDelta);
          currentProfitPct = underlyingProfitPct * netDelta;
        } else if (activeTrade.type === 'SHORT_PUT') {
          const delta = settings.shortPutDelta || 0.3;
          currentProfitPct = underlyingProfitPct * delta;
        } else if (activeTrade.type === 'PUT_SPREAD') {
          const shortDelta = settings.shortPutDelta || 0.3;
          const longDelta = settings.longPutDelta || 0.1;
          const netDelta = Math.max(0, shortDelta - longDelta);
          currentProfitPct = underlyingProfitPct * netDelta;
        }
        
        let profit = 0;
        if (settings.quantityType === 'USD') {
          profit = settings.quantity * currentProfitPct;
        } else if (settings.quantityType === 'BTC') {
          profit = settings.quantity * (isShort ? (activeTrade.entryPrice - exitPrice) : (exitPrice - activeTrade.entryPrice));
        } else {
          const btcQuantity = settings.quantity * 0.001;
          profit = btcQuantity * (isShort ? (activeTrade.entryPrice - exitPrice) : (exitPrice - activeTrade.entryPrice));
        }
        
        balance += profit;
        
        trades.push({
          type: activeTrade.type,
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
        if (isShort) {
          const currentProfitPct = (activeTrade.entryPrice - candle.low) / activeTrade.entryPrice;
          activeTrade.highestProfitPct = Math.max(activeTrade.highestProfitPct, currentProfitPct);

          if (activeTrade.highestProfitPct >= settings.trailingStopActivation) {
            const newTrailingStop = candle.low * (1 + settings.trailingStopOffset);
            if (activeTrade.trailingStopPrice === null || newTrailingStop < activeTrade.trailingStopPrice) {
              activeTrade.trailingStopPrice = newTrailingStop;
            }
          }
        } else {
          const currentProfitPct = (candle.high - activeTrade.entryPrice) / activeTrade.entryPrice;
          activeTrade.highestProfitPct = Math.max(activeTrade.highestProfitPct, currentProfitPct);

          if (activeTrade.highestProfitPct >= settings.trailingStopActivation) {
            const newTrailingStop = candle.high * (1 - settings.trailingStopOffset);
            if (activeTrade.trailingStopPrice === null || newTrailingStop > activeTrade.trailingStopPrice) {
              activeTrade.trailingStopPrice = newTrailingStop;
            }
          }
        }
      }
    }

    // Check for entry if not in trade
    const prevPrediction = predictionIdx > 0 ? predictions[predictionIdx - 1] : null;
    const velocityShort = prevPrediction ? (shortProb - prevPrediction[0]) : 0;
    const velocityLong = prevPrediction ? (longProb - prevPrediction[1]) : 0;
    const minVelocity = settings.minSignalVelocity || 0;
    const maxUncertainty = settings.maxUncertainty || 1;
    const mcPasses = settings.mcPasses || 1;

    if (!activeTrade) {
      let entryType: Trade['type'] | null = null;
      
      const sessionFilter = () => {
        const h = new Date(candle.time).getUTCHours();
        if (settings.useSessionTrading) {
          return (h >= settings.asiaStart && h < settings.asiaEnd) || (h >= settings.nyStart && h < settings.nyEnd);
        } else if (settings.onlyHighVolumeSessions) {
          return (h >= 0 && h <= 9) || (h >= 8 && h <= 17) || (h >= 13 && h <= 22);
        }
        return true;
      };

      if (sessionFilter()) {
        const bias = settings.biasThreshold || 0;
        const drlMatchesShort = drlActions && drlActions[predictionIdx] === 1;
        const drlMatchesLong = drlActions && drlActions[predictionIdx] === 0;

        let canShort = false;
        let canLong = false;

        if (settings.useDRLOnly) {
          canShort = !!drlMatchesShort;
          canLong = !!drlMatchesLong;
        } else {
          canShort = (shortProb * 100) > settings.shortThreshold && 
                          ((shortProb - longProb) * 100) >= bias &&
                          velocityShort >= minVelocity && 
                          (mcPasses <= 1 || uncertaintyShort <= maxUncertainty) &&
                          (!xgPredictions || (xgShortProb * 100 >= (settings.xgShortThreshold ?? 0))) &&
                          (!settings.useDRLConfluence || drlMatchesShort);
          canLong = (longProb * 100) > settings.longThreshold && 
                         ((longProb - shortProb) * 100) >= bias &&
                         velocityLong >= minVelocity && 
                         (mcPasses <= 1 || uncertaintyLong <= maxUncertainty) &&
                         (!xgPredictions || (xgLongProb * 100 >= (settings.xgLongThreshold ?? 0))) &&
                         (!settings.useDRLConfluence || drlMatchesLong);
        }

        if (canShort && (!canLong || shortProb > longProb)) {
          if (settings.strategyType === 'SHORT_BTC' || settings.strategyType === 'BOTH') {
            entryType = 'SHORT';
          } else if (settings.strategyType === 'CALL_SPREAD') {
            entryType = 'CALL_SPREAD';
          } else if (settings.strategyType === 'SHORT_CALL') {
            entryType = 'SHORT_CALL';
          }
        } else if (canLong) {
          if (settings.strategyType === 'LONG_BTC' || settings.strategyType === 'BOTH') {
            entryType = 'LONG';
          } else if (settings.strategyType === 'PUT_SPREAD') {
            entryType = 'PUT_SPREAD';
          } else if (settings.strategyType === 'SHORT_PUT') {
            entryType = 'SHORT_PUT';
          }
        }
      }

      if (entryType) {
        const tradeFeatures: Record<string, number> = {};
        if (features && features[predictionIdx] && featureNames) {
          featureNames.forEach((name, idx) => {
            tradeFeatures[name] = features[predictionIdx][idx];
          });
        }

        activeTrade = {
          type: entryType,
          entryPrice: candle.close,
          entryTime: candle.time,
          highestProfitPct: 0,
          trailingStopPrice: null,
          prediction: predictionFull,
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

  const precisionShort = (shortTP / (shortTP + shortFP) || 0) * 100;
  const recallShort = (shortTP / (shortTP + shortFN) || 0) * 100;
  const precisionLong = (longTP / (longTP + longFP) || 0) * 100;
  const recallLong = (longTP / (longTP + longFN) || 0) * 100;

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
    confusionMatrix: {
      short: { tp: shortTP, fp: shortFP, fn: shortFN, tn: shortTN, precision: precisionShort, recall: recallShort },
      long: { tp: longTP, fp: longFP, fn: longFN, tn: longTN, precision: precisionLong, recall: recallLong }
    },
    candles: fullCandles.slice(startIdx, startIdx + predictions.length),
    predictions,
    xgPredictions,
    drlActions
  };
}
