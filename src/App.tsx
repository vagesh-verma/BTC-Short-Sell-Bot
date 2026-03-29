import React, { useState, useEffect, useMemo, useRef } from 'react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  AreaChart, 
  Area 
} from 'recharts';
import { 
  TrendingDown, 
  Activity, 
  BarChart3, 
  Play, 
  RefreshCw, 
  AlertCircle, 
  ArrowDownRight,
  ArrowUpRight,
  Wallet,
  History,
  Settings2,
  ShieldAlert,
  Target,
  Zap
} from 'lucide-react';
import { format } from 'date-fns';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

import { fetchBTCData, Candle } from './services/dataService';
import { GRUModel, prepareData } from './services/modelService';
import { runBacktest, BacktestResult, Trade, BacktestSettings } from './services/backtestService';
import { calculateEMA, calculateRSI, calculateBollingerBands, calculateMACD, calculateStochasticRSI, calculateATR, calculateEMACross } from './services/indicatorService';
import { DeltaSocketService, TickerUpdate } from './services/deltaSocketService';
import { getSavedModelPairs, saveModelPair, loadModelPair, deleteModelPair, ModelPair } from './services/storageService';

import { Terminal } from './components/Terminal';
import { logger } from './services/loggerService';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export default function App() {
  const [candles, setCandles] = useState<Candle[]>([]);
  const [candles4h, setCandles4h] = useState<Candle[]>([]);
  const [loading, setLoading] = useState(true);
  const [training, setTraining] = useState(false);
  const [backtestResult, setBacktestResult] = useState<BacktestResult | null>(null);
  const [status, setStatus] = useState<string>('Ready to fetch data');
  const [error, setError] = useState<string | null>(null);
  const [epochs, setEpochs] = useState<number>(15);
  const [trainingLogs, setTrainingLogs] = useState<{epoch: number, loss: number, acc: number}[]>([]);
  const [secondaryTrainingLogs, setSecondaryTrainingLogs] = useState<{epoch: number, loss: number, acc: number}[]>([]);
  const [predictions, setPredictions] = useState<number[]>([]);
  const [predictionStats, setPredictionStats] = useState<{total: number, aboveThreshold: number} | null>(null);
  
  const model1hRef = useRef<GRUModel | null>(null);
  const model4hRef = useRef<GRUModel | null>(null);

  const [trainingRange, setTrainingRange] = useState({
    start: format(new Date(Date.now() - 30 * 24 * 60 * 60 * 1000), 'yyyy-MM-dd'),
    end: format(new Date(), 'yyyy-MM-dd')
  });
  const [backtestRange, setBacktestRange] = useState({
    start: format(new Date(Date.now() - 7 * 24 * 60 * 60 * 1000), 'yyyy-MM-dd'),
    end: format(new Date(), 'yyyy-MM-dd')
  });
  const [dropThreshold, setDropThreshold] = useState<number>(0.5);
  const [indicatorPeriods, setIndicatorPeriods] = useState({
    rsi: 14,
    ema: 20,
    ema9: 9,
    bb: 20
  });

  // Backtest Settings
  const [settings, setSettings] = useState<BacktestSettings>({
    threshold: 0.15,
    exitThreshold: 0.05,
    stopLoss: 0.015, // 1.5%
    takeProfit: 0.03, // 3%
    trailingStopActivation: 0.01, // 1%
    trailingStopOffset: 0.005, // 0.5%
    maxDurationHours: 12,
    quantity: 1000,
    quantityType: 'USD',
  });

  const [isLiveMode, setIsLiveMode] = useState(false);
  const [isRealTrading, setIsRealTrading] = useState(false);
  const [livePrice, setLivePrice] = useState<number | null>(null);
  const [lastLiveUpdate, setLastLiveUpdate] = useState<Date | null>(null);
  const [livePaperBalance, setLivePaperBalance] = useState(10000);
  const [realBalance, setRealBalance] = useState<number | null>(null);
  const [liveTrades, setLiveTrades] = useState<Trade[]>([]);
  const [serverStatus, setServerStatus] = useState<{apiKey: string, serverIp: string} | null>(null);
  const [activeLiveTrade, setActiveLiveTrade] = useState<{
    entryPrice: number;
    entryTime: number;
    highestProfitPct: number;
    trailingStopPrice: number | null;
  } | null>(null);
  const [livePrediction, setLivePrediction] = useState<number | null>(null);
  const [liveParams, setLiveParams] = useState<{
    rsi: number;
    ema: number;
    ema9: number;
    bbUpper: number;
    bbLower: number;
    macdHist: number;
    stochRsi: number;
    atr: number;
    emaCross: boolean;
    secondaryPrediction: number;
  } | null>(null);

  const [savedModels, setSavedModels] = useState<ModelPair[]>([]);
  const [newModelName, setNewModelName] = useState('');

  const windowSize = 20;

  useEffect(() => {
    setSavedModels(getSavedModelPairs());
  }, []);

  useEffect(() => {
    const fetchServerStatus = async () => {
      try {
        const res = await fetch('/api/status');
        if (res.ok) {
          const data = await res.json();
          setServerStatus(data);
        }
      } catch (e) {
        console.error('Failed to fetch server status');
      }
    };
    fetchServerStatus();
  }, []);

  useEffect(() => {
    loadData();
  }, [trainingRange]);

  useEffect(() => {
    if (isLiveMode && model1hRef.current && model4hRef.current) {
      const socket = new DeltaSocketService('BTCUSD', (update: TickerUpdate) => {
        setLivePrice(update.price);
        processLiveUpdate(update.price);
      });
      socket.connect();
      return () => socket.disconnect();
    }
  }, [isLiveMode]);

  useEffect(() => {
    if (isRealTrading) {
      fetchRealBalance();
    }
  }, [isRealTrading]);

  const fetchRealBalance = async () => {
    try {
      const response = await fetch('/api/wallet');
      const data = await response.json();
      if (data.result) {
        // Find USDT or BTC balance
        const usdt = data.result.find((b: any) => b.asset_symbol === 'USDT');
        if (usdt) setRealBalance(parseFloat(usdt.available_balance));
      }
    } catch (error) {
      console.error('Error fetching real balance:', error);
    }
  };

  const placeRealOrder = async (side: 'buy' | 'sell', size: number) => {
    try {
      const response = await fetch('/api/order', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol: 'BTCUSD',
          side,
          order_type: 'market',
          size
        })
      });
      const data = await response.json();
      if (data.result) {
        logger.success(`Real order placed: ${side} ${size} units`);
        fetchRealBalance();
      } else {
        logger.error(`Real order failed: ${data.error || 'Unknown error'}`);
      }
    } catch (error) {
      logger.error(`Error placing real order: ${error}`);
    }
  };

  const processLiveUpdate = async (price: number) => {
    if (!model1hRef.current || !model4hRef.current || candles.length < windowSize || candles4h.length < windowSize) return;

    // 1. Calculate 4h prediction as a feature
    const last4h = candles4h.slice(-windowSize);
    const p4h = last4h.map(c => c.close);
    const r4h = calculateRSI(p4h, indicatorPeriods.rsi);
    const e4h = calculateEMA(p4h, indicatorPeriods.ema);
    const b4h = calculateBollingerBands(p4h, indicatorPeriods.bb);
    
    const normalize = (arr: number[]) => {
      const min = Math.min(...arr);
      const max = Math.max(...arr);
      return arr.map(v => (max === min ? 0 : (v - min) / (max - min)));
    };

    const np4h = normalize(p4h);
    const nr4h = r4h.map(v => v / 100);
    const ne4h = normalize(e4h);
    const nu4h = normalize(b4h.upper);
    const nl4h = normalize(b4h.lower);

    const x4h: number[] = [];
    for (let j = 0; j < windowSize; j++) {
      x4h.push(np4h[j], nr4h[j], ne4h[j], nu4h[j], nl4h[j]);
    }
    const secondaryPrediction = model4hRef.current.predict(x4h);

    // 2. Calculate 1h prediction
    const last1h = candles.slice(-windowSize + 1);
    const p1h = [...last1h.map(c => c.close), price];
    const h1h = [...last1h.map(c => c.high), price];
    const l1h = [...last1h.map(c => c.low), price];
    
    const r1h = calculateRSI(p1h, indicatorPeriods.rsi);
    const e1h = calculateEMA(p1h, indicatorPeriods.ema);
    const e9_1h = calculateEMA(p1h, indicatorPeriods.ema9);
    const b1h = calculateBollingerBands(p1h, indicatorPeriods.bb);
    const m1h = calculateMACD(p1h);
    const s1h = calculateStochasticRSI(r1h);
    const a1h = calculateATR(h1h, l1h, p1h);
    const c1h = calculateEMACross(e9_1h, e1h);

    const np1h = normalize(p1h);
    const nr1h = r1h.map(v => v / 100);
    const ne1h = normalize(e1h);
    const nu1h = normalize(b1h.upper);
    const nl1h = normalize(b1h.lower);
    const nm1h = normalize(m1h.histogram);
    const na1h = normalize(a1h);
    const ne9_1h = normalize(e9_1h);

    const x1h: number[] = [];
    for (let j = 0; j < windowSize; j++) {
      x1h.push(
        np1h[j], nr1h[j], ne1h[j], nu1h[j], nl1h[j],
        nm1h[j], s1h[j], na1h[j], secondaryPrediction,
        ne9_1h[j], c1h.isBelow[j], c1h.isCross[j]
      );
    }

    const prediction = model1hRef.current.predict(x1h);
    setLivePrediction(prediction);
    setLivePrice(price);
    setLastLiveUpdate(new Date());

    setLiveParams({
      rsi: r1h[r1h.length - 1],
      ema: e1h[e1h.length - 1],
      ema9: e9_1h[e9_1h.length - 1],
      bbUpper: b1h.upper[b1h.upper.length - 1],
      bbLower: b1h.lower[b1h.lower.length - 1],
      macdHist: m1h.histogram[m1h.histogram.length - 1],
      stochRsi: s1h[s1h.length - 1],
      atr: a1h[a1h.length - 1],
      emaCross: c1h.isBelow[c1h.isBelow.length - 1] === 1,
      secondaryPrediction: secondaryPrediction
    });

    // 3. Live Paper Trading Logic
    if (activeLiveTrade) {
      const profitPct = (activeLiveTrade.entryPrice - price) / activeLiveTrade.entryPrice;
      const stopLossPrice = activeLiveTrade.entryPrice * (1 + settings.stopLoss);
      const takeProfitPrice = activeLiveTrade.entryPrice * (1 - settings.takeProfit);

      let shouldExit = false;
      let reason: Trade['exitReason'] = 'TIME';

      if (price >= stopLossPrice) {
        shouldExit = true;
        reason = 'STOP_LOSS';
      } else if (price <= takeProfitPrice) {
        shouldExit = true;
        reason = 'TAKE_PROFIT';
      } else if (prediction < settings.exitThreshold) {
        shouldExit = true;
        reason = 'PREDICTION';
      }

      if (shouldExit) {
        const profit = settings.quantity * profitPct;
        const newTrade: Trade = {
          type: 'SHORT',
          entryPrice: activeLiveTrade.entryPrice,
          exitPrice: price,
          entryTime: activeLiveTrade.entryTime,
          exitTime: Date.now(),
          profit,
          profitPct: profitPct * 100,
          exitReason: reason
        };
        setLiveTrades(prev => [newTrade, ...prev]);
        setLivePaperBalance(prev => prev + profit);
        setActiveLiveTrade(null);
        logger.success(`Live Trade Closed: ${reason} | Profit: $${profit.toFixed(2)}`);
        
        if (isRealTrading) {
          placeRealOrder('buy', settings.quantity); // Close SHORT with BUY
        }
      }
    } else if (prediction > settings.threshold) {
      setActiveLiveTrade({
        entryPrice: price,
        entryTime: Date.now(),
        highestProfitPct: 0,
        trailingStopPrice: null
      });
      logger.info(`Live Trade Opened: SHORT at $${price.toFixed(2)}`);
      
      if (isRealTrading) {
        placeRealOrder('sell', settings.quantity); // Open SHORT with SELL
      }
    }
  };

  const handleSaveModel = async () => {
    if (!newModelName.trim()) {
      logger.error('Please enter a name for the model.');
      return;
    }
    if (!model1hRef.current || !model4hRef.current) {
      logger.error('No trained models to save.');
      return;
    }
    try {
      await saveModelPair(newModelName, model1hRef.current, model4hRef.current);
      setSavedModels(getSavedModelPairs());
      setNewModelName('');
    } catch (err) {
      console.error(err);
    }
  };

  const handleLoadModel = async (name: string) => {
    try {
      setLoading(true);
      setStatus(`Loading model "${name}"...`);
      const { model1h, model4h } = await loadModelPair(name);
      model1hRef.current = model1h;
      model4hRef.current = model4h;
      setStatus(`Model "${name}" loaded!`);
      // Trigger a re-render to update UI buttons
      setPredictions([]); 
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteModel = (name: string) => {
    if (confirm(`Are you sure you want to delete model "${name}"?`)) {
      deleteModelPair(name);
      setSavedModels(getSavedModelPairs());
    }
  };

  const loadData = async () => {
    setLoading(true);
    setError(null);
    logger.info(`--- Initializing Data Load (${trainingRange.start} to ${trainingRange.end}) ---`);
    try {
      const startTs = new Date(trainingRange.start).getTime();
      const endTs = new Date(trainingRange.end).getTime();
      const data = await fetchBTCData(0, '1h', startTs, endTs);
      const data4h = await fetchBTCData(0, '4h', startTs - (windowSize * 4 * 60 * 60 * 1000), endTs);
      
      if (data.length === 0 || data4h.length === 0) {
        logger.error('Failed to fetch initial data.');
        throw new Error('Failed to fetch data');
      }
      setCandles(data);
      setCandles4h(data4h);
      setStatus(`Loaded ${data.length}h and ${data4h.length} 4h candles`);
      logger.success(`Successfully loaded ${data.length} 1h and ${data4h.length} 4h candles.`);
    } catch (err) {
      const msg = 'Error loading BTC data. Please check your connection.';
      setError(msg);
      logger.error(msg);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const getPredictionsForRange = async (candles1h: Candle[], candles4h: Candle[], m1h: GRUModel, m4h: GRUModel) => {
    logger.info('Generating predictions for provided range...');
    const prices4h = candles4h.map(c => c.close);
    const highs4h = candles4h.map(c => c.high);
    const lows4h = candles4h.map(c => c.low);
    const rsi4h = calculateRSI(prices4h, indicatorPeriods.rsi);
    const ema4h = calculateEMA(prices4h, indicatorPeriods.ema);
    const ema9_4h = calculateEMA(prices4h, indicatorPeriods.ema9);
    const cross4h = calculateEMACross(ema9_4h, ema4h);
    const bb4h = calculateBollingerBands(prices4h, indicatorPeriods.bb);
    const macd4h = calculateMACD(prices4h);
    const stochRsi4h = calculateStochasticRSI(rsi4h);
    const atr4h = calculateATR(highs4h, lows4h, prices4h);

    const secondaryPredictionsFor1h: number[] = [];
    const prices1h = candles1h.map(c => c.close);
    
    for (let i = 0; i < candles1h.length; i++) {
      const currentTime = candles1h[i].time;
      let last4hIdx = -1;
      for (let j = candles4h.length - 1; j >= windowSize; j--) {
        if (candles4h[j].time <= currentTime) {
          last4hIdx = j;
          break;
        }
      }

      if (last4hIdx >= windowSize) {
        const windowIndices = Array.from({ length: windowSize }, (_, k) => last4hIdx - windowSize + k);
        const pWindow = windowIndices.map(idx => prices4h[idx]);
        const rWindow = windowIndices.map(idx => rsi4h[idx]);
        const eWindow = windowIndices.map(idx => ema4h[idx]);
        const e9Window = windowIndices.map(idx => ema9_4h[idx]);
        const uWindow = windowIndices.map(idx => bb4h.upper[idx]);
        const lWindow = windowIndices.map(idx => bb4h.lower[idx]);

        const normalize = (arr: number[]) => {
          const min = Math.min(...arr);
          const max = Math.max(...arr);
          return arr.map(v => (max === min ? 0 : (v - min) / (max - min)));
        };

        const input: number[] = [];
        const nP = normalize(pWindow);
        const nR = rWindow.map(v => v / 100);
        const nE = normalize(eWindow);
        const nE9 = normalize(e9Window);
        const nU = normalize(uWindow);
        const nL = normalize(lWindow);
        const nH = normalize(windowIndices.map(idx => macd4h.histogram[idx]));
        const nS = windowIndices.map(idx => stochRsi4h[idx]);
        const nA = normalize(windowIndices.map(idx => atr4h[idx]));
        const nBelow = windowIndices.map(idx => cross4h.isBelow[idx]);
        const nCross = windowIndices.map(idx => cross4h.isCross[idx]);

        for (let j = 0; j < windowSize; j++) {
          input.push(nP[j], nR[j], nE[j], nU[j], nL[j], nH[j], nS[j], nA[j], nE9[j], nBelow[j], nCross[j]);
        }
        secondaryPredictionsFor1h.push(m4h.predict(input));
      } else {
        secondaryPredictionsFor1h.push(0.5);
      }
    }

    const highs1h = candles1h.map(c => c.high);
    const lows1h = candles1h.map(c => c.low);
    const rsi1h = calculateRSI(prices1h, indicatorPeriods.rsi);
    const ema1h = calculateEMA(prices1h, indicatorPeriods.ema);
    const ema9_1h = calculateEMA(prices1h, indicatorPeriods.ema9);
    const cross1h = calculateEMACross(ema9_1h, ema1h);
    const bb1h = calculateBollingerBands(prices1h, indicatorPeriods.bb);
    const macd1h = calculateMACD(prices1h);
    const stochRsi1h = calculateStochasticRSI(rsi1h);
    const atr1h = calculateATR(highs1h, lows1h, prices1h);

    const generatedPredictions: number[] = [];
    for (let i = windowSize; i < prices1h.length; i++) {
      const windowIndices = Array.from({ length: windowSize }, (_, k) => i - windowSize + k);
      
      const priceWindow = windowIndices.map(idx => prices1h[idx]);
      const rsiWindow = windowIndices.map(idx => rsi1h[idx]);
      const emaWindow = windowIndices.map(idx => ema1h[idx]);
      const ema9Window = windowIndices.map(idx => ema9_1h[idx]);
      const bbUpperWindow = windowIndices.map(idx => bb1h.upper[idx]);
      const bbLowerWindow = windowIndices.map(idx => bb1h.lower[idx]);
      const secWindow = windowIndices.map(idx => secondaryPredictionsFor1h[idx]);

      const normalize = (arr: number[]) => {
        const min = Math.min(...arr);
        const max = Math.max(...arr);
        return arr.map(v => (max === min ? 0 : (v - min) / (max - min)));
      };

      const normPrice = normalize(priceWindow);
      const normRsi = rsiWindow.map(v => v / 100);
      const normEma = normalize(emaWindow);
      const normEma9 = normalize(ema9Window);
      const normBbUpper = normalize(bbUpperWindow);
      const normBbLower = normalize(bbLowerWindow);
      const normMacd = normalize(windowIndices.map(idx => macd1h.histogram[idx]));
      const normStochRsi = windowIndices.map(idx => stochRsi1h[idx]);
      const normAtr = normalize(windowIndices.map(idx => atr1h[idx]));
      const normBelow = windowIndices.map(idx => cross1h.isBelow[idx]);
      const normCross = windowIndices.map(idx => cross1h.isCross[idx]);

      const input: number[] = [];
      for (let j = 0; j < windowSize; j++) {
        input.push(
          normPrice[j], normRsi[j], normEma[j], normBbUpper[j], normBbLower[j],
          normMacd[j], normStochRsi[j], normAtr[j], secWindow[j],
          normEma9[j], normBelow[j], normCross[j]
        );
      }
      generatedPredictions.push(m1h.predict(input));
    }
    return generatedPredictions;
  };

  const startTraining = async () => {
    if (candles.length < windowSize + 50) return;
    
    setTraining(true);
    logger.info('--- Starting Training Process ---');
    setPredictions([]);
    setBacktestResult(null);
    setTrainingLogs([]);
    setSecondaryTrainingLogs([]);
    
    try {
      // 1. Fetch 4h data for secondary model
      setStatus('Fetching 4h data for secondary model...');
      logger.info('Fetching 4h historical data for trend analysis...');
      const startTs = new Date(trainingRange.start).getTime();
      const endTs = new Date(trainingRange.end).getTime();
      const candles4h = await fetchBTCData(0, '4h', startTs, endTs);
      
      if (candles4h.length < windowSize + 10) {
        logger.error('Insufficient 4h data for training.');
        throw new Error('Not enough 4h data');
      }

      // 2. Train Secondary 4h Model
      setStatus('Training Secondary 4h Model...');
      logger.info('Training secondary 4h model for trend analysis...');
      const prices4h = candles4h.map(c => c.close);
      const highs4h = candles4h.map(c => c.high);
      const lows4h = candles4h.map(c => c.low);
      const rsi4h = calculateRSI(prices4h, indicatorPeriods.rsi);
      const ema4h = calculateEMA(prices4h, indicatorPeriods.ema);
      const ema9_4h = calculateEMA(prices4h, indicatorPeriods.ema9);
      const cross4h = calculateEMACross(ema9_4h, ema4h);
      const bb4h = calculateBollingerBands(prices4h, indicatorPeriods.bb);
      const macd4h = calculateMACD(prices4h);
      const stochRsi4h = calculateStochasticRSI(rsi4h);
      const atr4h = calculateATR(highs4h, lows4h, prices4h);
      
      const data4h = prepareData(
        prices4h, rsi4h, ema4h, bb4h.upper, bb4h.lower, windowSize, 
        undefined, macd4h.histogram, stochRsi4h, atr4h, dropThreshold,
        ema9_4h, cross4h.isBelow, cross4h.isCross
      );
      const model4h = new GRUModel(windowSize, 11);
      await model4h.buildModel();
      
      await model4h.train(data4h.xs, data4h.ys, 10, (epoch, logs) => {
        if (logs) {
          setSecondaryTrainingLogs(prev => [...prev, { epoch: epoch + 1, loss: logs.loss, acc: logs.acc }]);
        }
      });
      model4hRef.current = model4h;

      // 3. Generate 4h predictions for 1h candles
      setStatus('Generating Secondary Predictions for 1h data...');
      logger.info('Generating secondary predictions for primary model...');
      const secondaryPredictionsFor1h: number[] = [];
      const prices1h = candles.map(c => c.close);
      
      // For each 1h candle, find the most recent completed 4h candle and get its prediction
      for (let i = 0; i < candles.length; i++) {
        const currentTime = candles[i].time;
        // Find the index of the 4h candle that just closed before or at this 1h candle
        let last4hIdx = -1;
        for (let j = candles4h.length - 1; j >= windowSize; j--) {
          if (candles4h[j].time <= currentTime) {
            last4hIdx = j;
            break;
          }
        }

        if (last4hIdx >= windowSize) {
          const windowIndices = Array.from({ length: windowSize }, (_, k) => last4hIdx - windowSize + k);
          const pWindow = windowIndices.map(idx => prices4h[idx]);
          const rWindow = windowIndices.map(idx => rsi4h[idx]);
          const eWindow = windowIndices.map(idx => ema4h[idx]);
          const e9Window = windowIndices.map(idx => ema9_4h[idx]);
          const uWindow = windowIndices.map(idx => bb4h.upper[idx]);
          const lWindow = windowIndices.map(idx => bb4h.lower[idx]);

          const normalize = (arr: number[]) => {
            const min = Math.min(...arr);
            const max = Math.max(...arr);
            return arr.map(v => (max === min ? 0 : (v - min) / (max - min)));
          };

          const input: number[] = [];
          const nP = normalize(pWindow);
          const nR = rWindow.map(v => v / 100);
          const nE = normalize(eWindow);
          const nE9 = normalize(e9Window);
          const nU = normalize(uWindow);
          const nL = normalize(lWindow);
          const nH = normalize(windowIndices.map(idx => macd4h.histogram[idx]));
          const nS = windowIndices.map(idx => stochRsi4h[idx]);
          const nA = normalize(windowIndices.map(idx => atr4h[idx]));
          const nBelow = windowIndices.map(idx => cross4h.isBelow[idx]);
          const nCross = windowIndices.map(idx => cross4h.isCross[idx]);

          for (let j = 0; j < windowSize; j++) {
            input.push(nP[j], nR[j], nE[j], nU[j], nL[j], nH[j], nS[j], nA[j], nE9[j], nBelow[j], nCross[j]);
          }
          secondaryPredictionsFor1h.push(model4h.predict(input));
        } else {
          secondaryPredictionsFor1h.push(0.5); // Neutral if no 4h data yet
        }
      }

      // 4. Train Primary 1h Model with Secondary Feature
      setStatus('Calculating 1h Technical Indicators...');
      logger.info('Calculating 1h technical indicators...');
      const highs1h = candles.map(c => c.high);
      const lows1h = candles.map(c => c.low);
      const rsi1h = calculateRSI(prices1h, indicatorPeriods.rsi);
      const ema1h = calculateEMA(prices1h, indicatorPeriods.ema);
      const ema9_1h = calculateEMA(prices1h, indicatorPeriods.ema9);
      const cross1h = calculateEMACross(ema9_1h, ema1h);
      const bb1h = calculateBollingerBands(prices1h, indicatorPeriods.bb);
      const macd1h = calculateMACD(prices1h);
      const stochRsi1h = calculateStochasticRSI(rsi1h);
      const atr1h = calculateATR(highs1h, lows1h, prices1h);

      setStatus('Preparing 1h Data with Secondary Feature...');
      logger.info('Preparing 1h training data...');
      const data1h = prepareData(
        prices1h, 
        rsi1h, 
        ema1h, 
        bb1h.upper, 
        bb1h.lower, 
        windowSize, 
        secondaryPredictionsFor1h,
        macd1h.histogram,
        stochRsi1h,
        atr1h,
        dropThreshold,
        ema9_1h,
        cross1h.isBelow,
        cross1h.isCross
      );
      
      const model1h = new GRUModel(windowSize, 12);
      setStatus('Building Deep GRU Architecture (12 Features)...');
      await model1h.buildModel();
      
      setStatus(`Training Primary Model (${epochs} Epochs)...`);
      logger.info(`Training primary model for ${epochs} epochs...`);
      await model1h.train(data1h.xs, data1h.ys, epochs, (epoch, logs) => {
        if (logs) {
          setTrainingLogs(prev => [...prev, { 
            epoch: epoch + 1, 
            loss: logs.loss, 
            acc: logs.acc 
          }]);
        }
      });
      model1hRef.current = model1h;
      
      setStatus('Generating Final Predictions...');
      const generatedPredictions = await getPredictionsForRange(candles, candles4h, model1h, model4h);
      
      setPredictions(generatedPredictions);
      const above = generatedPredictions.filter(p => p > settings.threshold).length;
      setPredictionStats({ total: generatedPredictions.length, aboveThreshold: above });
      setStatus('Model trained! Ready for backtest.');
      logger.success(`Predictions generated. ${above} signals above threshold.`);
      
      // Auto-run initial backtest
      logger.info('Running initial backtest...');
      const result = runBacktest(candles.slice(windowSize), generatedPredictions, settings, 10000, 0);
      setBacktestResult(result);
      logger.success('--- Training and Backtest Complete ---');
    } catch (err: any) {
      const msg = err.message || 'Error during training';
      setError(msg);
      logger.error(`Training failed: ${msg}`);
      console.error(err);
    } finally {
      setTraining(false);
    }
  };

  const startBacktest = async () => {
    if (!model1hRef.current || !model4hRef.current) {
      logger.error('Model not trained yet.');
      return;
    }
    setLoading(true);
    setStatus('Preparing backtest data...');
    logger.info('--- Starting High-Resolution Backtest ---');
    
    try {
      const startTs = new Date(backtestRange.start).getTime();
      // Set endTs to the end of the selected day
      const endTs = new Date(backtestRange.end).getTime() + (24 * 60 * 60 * 1000) - 1;
      
      logger.info(`--- Starting Backtest from ${backtestRange.start} to ${backtestRange.end} ---`);
      
      // 1. Fetch 1h and 4h data for the backtest range to generate predictions
      // We need extra data before the start to fill the window
      const fetchStartTs = startTs - (windowSize * 60 * 60 * 1000);
      
      logger.info(`Fetching 1h data from ${format(new Date(fetchStartTs), 'yyyy-MM-dd HH:mm')} to ${format(new Date(endTs), 'yyyy-MM-dd HH:mm')}...`);
      const btCandles1h = await fetchBTCData(0, '1h', fetchStartTs, endTs);
      
      logger.info(`Fetching 4h data for trend context...`);
      const btCandles4h = await fetchBTCData(0, '4h', fetchStartTs - (windowSize * 4 * 60 * 60 * 1000), endTs);

      if (btCandles1h.length < windowSize) {
        throw new Error(`Insufficient 1h data: got ${btCandles1h.length}, need at least ${windowSize}`);
      }

      setStatus('Generating predictions for backtest range...');
      const btPredictions = await getPredictionsForRange(btCandles1h, btCandles4h, model1hRef.current, model4hRef.current);
      logger.info(`Generated ${btPredictions.length} predictions.`);

      // 2. Fetch 5m data for high-res backtest
      setStatus('Fetching 5m data for accurate backtest...');
      logger.info(`Fetching 5m candles for backtest from ${format(new Date(startTs), 'yyyy-MM-dd HH:mm')} to ${format(new Date(endTs), 'yyyy-MM-dd HH:mm')}...`);
      const highResCandles = await fetchBTCData(0, '5m', startTs, endTs);
      
      if (highResCandles.length === 0) {
        throw new Error('Failed to fetch 5m data for the selected range');
      }

      setStatus('Aligning predictions with 5m candles...');
      
      const predictionMap = new Map<number, number>();
      btPredictions.forEach((val, i) => {
        // btPredictions[i] corresponds to btCandles1h[i + windowSize]
        const candleIndex = i + windowSize;
        if (btCandles1h[candleIndex]) {
          const time = btCandles1h[candleIndex].time;
          predictionMap.set(time, val);
        }
      });

      const alignedPredictions: number[] = [];
      const validHighResCandles: Candle[] = [];

      highResCandles.forEach(c => {
        // Round down to the nearest hour to find the corresponding prediction
        const hourTimestamp = Math.floor(c.time / (1000 * 60 * 60)) * (1000 * 60 * 60);
        const prediction = predictionMap.get(hourTimestamp);
        
        if (prediction !== undefined) {
          // Carry the prediction for the entire hour
          alignedPredictions.push(prediction);
          validHighResCandles.push(c);
        }
      });

      if (validHighResCandles.length === 0) {
        logger.error(`Alignment failed. Prediction map size: ${predictionMap.size}, High-res candles: ${highResCandles.length}`);
        if (highResCandles.length > 0) {
          logger.info(`First 5m candle: ${format(new Date(highResCandles[0].time), 'yyyy-MM-dd HH:mm')}`);
          const firstPredTime = Array.from(predictionMap.keys()).sort()[0];
          if (firstPredTime) {
            logger.info(`First prediction time: ${format(new Date(firstPredTime), 'yyyy-MM-dd HH:mm')}`);
          }
        }
        throw new Error('No overlapping data found for backtest period. Ensure the range is within the trained data or fetch more data.');
      }

      setStatus('Running High-Resolution Backtest...');
      logger.info(`Running backtest on ${validHighResCandles.length} candles (${(validHighResCandles.length * 5 / 60).toFixed(1)} hours of data)...`);
      const result = runBacktest(validHighResCandles, alignedPredictions, settings, 10000, 0);
      
      setBacktestResult(result);
      setStatus('Backtest complete!');
      logger.success(`Backtest complete! Profit: ${result.totalProfit.toFixed(2)}%`);
    } catch (err: any) {
      const msg = err.message || 'Error running backtest';
      setError(msg);
      logger.error(`Backtest failed: ${msg}`);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const chartData = useMemo(() => {
    return candles.slice(-100).map(c => ({
      time: format(new Date(c.time), 'HH:mm'),
      price: c.close,
    }));
  }, [candles]);

  const equityData = useMemo(() => {
    if (!backtestResult) return [];
    return backtestResult.equityCurve.map(e => ({
      time: format(new Date(e.time), 'MM/dd HH:mm'),
      balance: parseFloat(e.balance.toFixed(2)),
    }));
  }, [backtestResult]);

  return (
    <div className="min-h-screen bg-[#0A0A0B] text-white font-sans selection:bg-emerald-500/30">
      {/* Server Status Bar */}
      {serverStatus && (
        <div className="bg-white/5 border-b border-white/5 px-6 py-2 flex flex-wrap items-center gap-6 text-[10px] uppercase tracking-widest font-bold text-white/40">
          <div className="flex items-center gap-2">
            <ShieldAlert className="w-3 h-3 text-emerald-500" />
            <span className="text-white/60">Server Status:</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="opacity-60">IP:</span>
            <code className="bg-white/5 px-2 py-0.5 rounded text-emerald-400 font-mono">{serverStatus.serverIp}</code>
          </div>
          <div className="flex items-center gap-2">
            <span className="opacity-60">API Key:</span>
            <code className="bg-white/5 px-2 py-0.5 rounded text-emerald-400 font-mono">{serverStatus.apiKey}</code>
          </div>
          <div className="ml-auto opacity-40 italic normal-case font-medium">
            Whitelist this IP in Delta Exchange API settings
          </div>
        </div>
      )}
      {/* Header */}
      <header className="border-b border-white/5 bg-[#0D0D0E]/80 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-emerald-500 rounded-lg flex items-center justify-center shadow-[0_0_15px_rgba(16,185,129,0.3)]">
              <Activity className="w-5 h-5 text-black" />
            </div>
            <h1 className="text-lg font-semibold tracking-tight">BTC Short-Sell GRU Bot</h1>
          </div>
          
          <div className="flex items-center gap-4">
            <button
              onClick={() => setIsLiveMode(!isLiveMode)}
              disabled={!model1hRef.current}
              className={cn(
                "flex items-center gap-2 px-4 py-2 rounded-full border transition-all font-bold text-[10px] uppercase tracking-widest",
                isLiveMode 
                  ? "bg-emerald-500/20 border-emerald-500/50 text-emerald-400 shadow-[0_0_15px_rgba(16,185,129,0.2)]" 
                  : "bg-white/5 border-white/10 text-white/40 hover:border-white/20"
              )}
            >
              <Zap className={cn("w-3 h-3", isLiveMode && "fill-emerald-400")} />
              {isLiveMode ? 'Live Mode ON' : 'Live Mode OFF'}
            </button>

            {isLiveMode && (
              <button
                onClick={() => setIsRealTrading(!isRealTrading)}
                className={cn(
                  "flex items-center gap-2 px-4 py-2 rounded-full border transition-all font-bold text-[10px] uppercase tracking-widest",
                  isRealTrading 
                    ? "bg-red-500/20 border-red-500/50 text-red-400 shadow-[0_0_15px_rgba(239,68,68,0.2)]" 
                    : "bg-white/5 border-white/10 text-white/40 hover:border-white/20"
                )}
              >
                <ShieldAlert className={cn("w-3 h-3", isRealTrading && "fill-red-400")} />
                {isRealTrading ? 'REAL TRADING ON' : 'PAPER ONLY'}
              </button>
            )}
            <div className="flex items-center gap-2 px-3 py-1.5 bg-white/5 rounded-full border border-white/10">
              <div className={cn(
                "w-2 h-2 rounded-full",
                loading || training ? "bg-amber-500 animate-pulse" : "bg-emerald-500"
              )} />
              <span className="text-xs font-medium text-white/70">{status}</span>
            </div>
            <button 
              onClick={loadData}
              disabled={loading || training}
              className="p-2 hover:bg-white/5 rounded-lg transition-colors disabled:opacity-50"
            >
              <RefreshCw className={cn("w-4 h-4", loading && "animate-spin")} />
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8 space-y-12">
        {/* Section 1: Current Price & Trend */}
        <section className="space-y-6">
          <div className="flex items-center gap-2 border-b border-white/5 pb-2">
            <Activity className="w-5 h-5 text-emerald-500" />
            <h2 className="text-xl font-bold tracking-tight">1. BTC Price & Trend</h2>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
            <div className="lg:col-span-1">
              <StatCard 
                title="Current BTC Price" 
                value={isLiveMode && livePrice ? `$${livePrice.toLocaleString()}` : (candles.length > 0 ? `$${candles[candles.length - 1].close.toLocaleString()}` : '---')}
                icon={<Wallet className="w-4 h-4 text-emerald-400" />}
                trend={isLiveMode && livePrice && candles.length > 0 ? (livePrice > candles[candles.length - 1].close ? 'up' : 'down') : (candles.length > 1 ? (candles[candles.length - 1].close > candles[candles.length - 2].close ? 'up' : 'down') : undefined)}
              />
            </div>
            <div className="lg:col-span-3">
              <div className="bg-[#0D0D0E] border border-white/5 rounded-2xl p-6 shadow-xl">
                <div className="flex items-center justify-between mb-6">
                  <div className="flex items-center gap-2">
                    <BarChart3 className="w-5 h-5 text-emerald-500" />
                    <h2 className="font-medium">BTC/USDT Price (1h)</h2>
                  </div>
                  <span className="px-2 py-1 bg-emerald-500/10 text-emerald-500 text-[10px] font-bold rounded uppercase tracking-wider">Live Feed</span>
                </div>
                <div className="h-[300px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#ffffff05" vertical={false} />
                      <XAxis dataKey="time" stroke="#ffffff30" fontSize={10} tickLine={false} axisLine={false} />
                      <YAxis domain={['auto', 'auto']} stroke="#ffffff30" fontSize={10} tickLine={false} axisLine={false} tickFormatter={(val) => `$${val.toLocaleString()}`} />
                      <Tooltip contentStyle={{ backgroundColor: '#0D0D0E', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }} itemStyle={{ color: '#10b981' }} />
                      <Line type="monotone" dataKey="price" stroke="#10b981" strokeWidth={2} dot={false} animationDuration={1000} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Section 2: Model Training */}
        <section className="space-y-6">
          <div className="flex items-center gap-2 border-b border-white/5 pb-2">
            <Zap className="w-5 h-5 text-purple-500" />
            <h2 className="text-xl font-bold tracking-tight">2. Model Training</h2>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-1">
              <div className="bg-[#0D0D0E] border border-white/5 rounded-2xl p-6 shadow-xl h-full">
                <h3 className="font-medium mb-6 flex items-center gap-2 text-purple-400">
                  <Settings2 className="w-4 h-4" />
                  Training Parameters
                </h3>
                <div className="space-y-6">
                  {predictionStats && (
                    <div className="p-4 bg-purple-500/10 border border-purple-500/20 rounded-xl space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-[10px] text-purple-400 uppercase font-bold">Model Confidence</span>
                        <span className="text-xs font-bold text-purple-300">{(predictionStats.aboveThreshold / predictionStats.total * 100).toFixed(1)}% Active</span>
                      </div>
                      <div className="flex items-center justify-between text-[10px] text-white/40">
                        <span>Signals Triggered:</span>
                        <span className="text-white/70 font-mono">{predictionStats.aboveThreshold} / {predictionStats.total}</span>
                      </div>
                      <div className="w-full h-1 bg-white/5 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-purple-500 transition-all duration-500" 
                          style={{ width: `${Math.min(100, (predictionStats.aboveThreshold / predictionStats.total * 100))}%` }}
                        />
                      </div>
                    </div>
                  )}
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-1.5">
                      <label className="text-[10px] text-white/40 uppercase font-bold">Training Start</label>
                      <input type="date" value={trainingRange.start} onChange={(e) => setTrainingRange({...trainingRange, start: e.target.value})} className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-xs focus:border-purple-500/50 outline-none" />
                    </div>
                    <div className="space-y-1.5">
                      <label className="text-[10px] text-white/40 uppercase font-bold">Training End</label>
                      <input type="date" value={trainingRange.end} onChange={(e) => setTrainingRange({...trainingRange, end: e.target.value})} className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-xs focus:border-purple-500/50 outline-none" />
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-1.5">
                      <label className="text-[10px] text-white/40 uppercase font-bold">Drop Target (%)</label>
                      <input type="number" step="0.1" min="0.1" max="5" value={dropThreshold} onChange={(e) => setDropThreshold(parseFloat(e.target.value) || 0.5)} className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-xs focus:border-purple-500/50 outline-none" />
                    </div>
                    <div className="space-y-1.5">
                      <label className="text-[10px] text-white/40 uppercase font-bold">Epochs</label>
                      <input type="number" min="5" max="100" value={epochs} onChange={(e) => setEpochs(parseInt(e.target.value) || 15)} className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-xs focus:border-purple-500/50 outline-none" />
                    </div>
                  </div>
                  <div className="grid grid-cols-3 gap-3">
                    <div className="space-y-1.5">
                      <label className="text-[10px] text-white/40 uppercase font-bold">EMA 9</label>
                      <input type="number" value={indicatorPeriods.ema9} onChange={(e) => setIndicatorPeriods({...indicatorPeriods, ema9: parseInt(e.target.value) || 9})} className="w-full bg-white/5 border border-white/10 rounded-lg px-2 py-2 text-xs focus:border-purple-500/50 outline-none" />
                    </div>
                    <div className="space-y-1.5">
                      <label className="text-[10px] text-white/40 uppercase font-bold">RSI</label>
                      <input type="number" value={indicatorPeriods.rsi} onChange={(e) => setIndicatorPeriods({...indicatorPeriods, rsi: parseInt(e.target.value) || 14})} className="w-full bg-white/5 border border-white/10 rounded-lg px-2 py-2 text-xs focus:border-purple-500/50 outline-none" />
                    </div>
                    <div className="space-y-1.5">
                      <label className="text-[10px] text-white/40 uppercase font-bold">EMA 20</label>
                      <input type="number" value={indicatorPeriods.ema} onChange={(e) => setIndicatorPeriods({...indicatorPeriods, ema: parseInt(e.target.value) || 20})} className="w-full bg-white/5 border border-white/10 rounded-lg px-2 py-2 text-xs focus:border-purple-500/50 outline-none" />
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-1.5">
                      <label className="text-[10px] text-white/40 uppercase font-bold">BB</label>
                      <input type="number" value={indicatorPeriods.bb} onChange={(e) => setIndicatorPeriods({...indicatorPeriods, bb: parseInt(e.target.value) || 20})} className="w-full bg-white/5 border border-white/10 rounded-lg px-2 py-2 text-xs focus:border-purple-500/50 outline-none" />
                    </div>
                  </div>
                  <button onClick={startTraining} disabled={loading || training || candles.length === 0} className={cn("w-full py-3 rounded-xl font-semibold flex items-center justify-center gap-2 transition-all active:scale-95", training ? "bg-white/5 text-white/50 cursor-not-allowed" : "bg-purple-600 hover:bg-purple-500 text-white shadow-[0_0_20px_rgba(147,51,234,0.2)]")}>
                    {training ? <><RefreshCw className="w-4 h-4 animate-spin" /> Training...</> : <><Play className="w-4 h-4 fill-current" /> Start Training</>}
                  </button>

                  {model1hRef.current && (
                    <div className="pt-4 border-t border-white/5 space-y-3">
                      <h4 className="text-[10px] text-white/40 uppercase font-bold tracking-wider">Save Current Model</h4>
                      <div className="flex gap-2">
                        <input 
                          type="text" 
                          placeholder="Model Name..." 
                          value={newModelName}
                          onChange={(e) => setNewModelName(e.target.value)}
                          className="flex-1 bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-xs focus:border-purple-500/50 outline-none"
                        />
                        <button 
                          onClick={handleSaveModel}
                          className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg text-xs font-bold transition-colors"
                        >
                          Save
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
            <div className="lg:col-span-2">
              <div className="bg-[#0D0D0E] border border-white/5 rounded-2xl p-6 shadow-xl h-full">
                <h3 className="font-medium mb-4 flex items-center gap-2 text-purple-400">
                  <Activity className="w-4 h-4" />
                  Debug Output & Logs
                </h3>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <h4 className="text-[10px] text-white/40 uppercase font-bold tracking-wider">Primary Logs (1h)</h4>
                      <div className="bg-black/40 rounded-xl border border-white/5 p-4 h-[200px] overflow-y-auto font-mono text-[10px] space-y-1 custom-scrollbar">
                        {trainingLogs.length === 0 ? (
                          <div className="h-full flex items-center justify-center text-white/20 italic">Waiting for training...</div>
                        ) : (
                          trainingLogs.map((log, i) => (
                            <div key={i} className="flex justify-between border-b border-white/5 pb-1">
                              <span className="text-emerald-500">Epoch {log.epoch.toString().padStart(2, '0')}</span>
                              <span className="text-white/60">L: {log.loss.toFixed(4)}</span>
                              <span className="text-amber-500">A: {(log.acc * 100).toFixed(1)}%</span>
                            </div>
                          ))
                        )}
                      </div>
                    </div>
                    <div className="space-y-2">
                      <h4 className="text-[10px] text-white/40 uppercase font-bold tracking-wider">Secondary Logs (4h)</h4>
                      <div className="bg-black/40 rounded-xl border border-white/5 p-4 h-[200px] overflow-y-auto font-mono text-[10px] space-y-1 custom-scrollbar">
                        {secondaryTrainingLogs.length === 0 ? (
                          <div className="h-full flex items-center justify-center text-white/20 italic">Waiting for training...</div>
                        ) : (
                          secondaryTrainingLogs.map((log, i) => (
                            <div key={i} className="flex justify-between border-b border-white/5 pb-1">
                              <span className="text-emerald-500">Epoch {log.epoch.toString().padStart(2, '0')}</span>
                              <span className="text-white/60">L: {log.loss.toFixed(4)}</span>
                              <span className="text-amber-500">A: {(log.acc * 100).toFixed(1)}%</span>
                            </div>
                          ))
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Section 3: Backtest Settings & Results */}
        <section className="space-y-6">
          <div className="flex items-center gap-2 border-b border-white/5 pb-2">
            <TrendingDown className="w-5 h-5 text-amber-500" />
            <h2 className="text-xl font-bold tracking-tight">3. Backtest & Results</h2>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
            {/* Settings */}
            <div className="lg:col-span-1 space-y-6">
              <div className="bg-[#0D0D0E] border border-white/5 rounded-2xl p-6 shadow-xl">
                <h3 className="font-medium mb-6 flex items-center gap-2 text-amber-400">
                  <Settings2 className="w-4 h-4" />
                  Execution Settings
                </h3>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-1.5">
                      <label className="text-[10px] text-white/40 uppercase font-bold">Backtest Start</label>
                      <input type="date" value={backtestRange.start} onChange={(e) => setBacktestRange({...backtestRange, start: e.target.value})} className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-xs outline-none focus:border-amber-500/50" />
                    </div>
                    <div className="space-y-1.5">
                      <label className="text-[10px] text-white/40 uppercase font-bold">Backtest End</label>
                      <input type="date" value={backtestRange.end} onChange={(e) => setBacktestRange({...backtestRange, end: e.target.value})} className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-xs outline-none focus:border-amber-500/50" />
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-4 pt-2 border-t border-white/5">
                    <div className="space-y-1.5">
                      <label className="text-[10px] text-white/40 uppercase font-bold tracking-wider">Entry Threshold</label>
                      <input type="number" step="0.01" value={settings.threshold} onChange={(e) => setSettings({...settings, threshold: parseFloat(e.target.value)})} className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-xs outline-none focus:border-amber-500/50" />
                    </div>
                    <div className="space-y-1.5">
                      <label className="text-[10px] text-white/40 uppercase font-bold tracking-wider">Exit Threshold</label>
                      <input type="number" step="0.01" value={settings.exitThreshold} onChange={(e) => setSettings({...settings, exitThreshold: parseFloat(e.target.value)})} className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-xs outline-none focus:border-amber-500/50" />
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-4 pt-2 border-t border-white/5">
                    <div className="space-y-1.5">
                      <label className="text-[10px] text-white/40 uppercase font-bold">Max Duration (H)</label>
                      <input type="number" value={settings.maxDurationHours} onChange={(e) => setSettings({...settings, maxDurationHours: parseInt(e.target.value) || 12})} className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-xs outline-none focus:border-amber-500/50" />
                    </div>
                    <div className="space-y-1.5">
                      <label className="text-[10px] text-white/40 uppercase font-bold">Quantity</label>
                      <input type="number" step="0.01" value={settings.quantity} onChange={(e) => setSettings({...settings, quantity: parseFloat(e.target.value) || 1000})} className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-xs outline-none focus:border-amber-500/50" />
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-4 pt-2 border-t border-white/5">
                    <div className="space-y-1.5">
                      <label className="text-[10px] text-white/40 uppercase font-bold">Unit</label>
                      <select value={settings.quantityType} onChange={(e) => setSettings({...settings, quantityType: e.target.value as 'USD' | 'BTC'})} className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-xs outline-none focus:border-amber-500/50 appearance-none">
                        <option value="USD">USD</option>
                        <option value="BTC">BTC</option>
                      </select>
                    </div>
                    <div className="space-y-1.5">
                      <label className="text-[10px] text-white/40 uppercase font-bold">Stop Loss %</label>
                      <input type="number" step="0.1" value={settings.stopLoss * 100} onChange={(e) => setSettings({...settings, stopLoss: parseFloat(e.target.value) / 100})} className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-xs outline-none focus:border-amber-500/50" />
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-4 pt-2 border-t border-white/5">
                    <div className="space-y-1.5">
                      <label className="text-[10px] text-white/40 uppercase font-bold">Take Profit %</label>
                      <input type="number" step="0.1" value={settings.takeProfit * 100} onChange={(e) => setSettings({...settings, takeProfit: parseFloat(e.target.value) / 100})} className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-xs outline-none focus:border-amber-500/50" />
                    </div>
                    <div className="space-y-1.5">
                      <label className="text-[10px] text-white/40 uppercase font-bold">Trail Trigger %</label>
                      <input type="number" step="0.1" value={settings.trailingStopActivation * 100} onChange={(e) => setSettings({...settings, trailingStopActivation: parseFloat(e.target.value) / 100})} className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-xs outline-none focus:border-amber-500/50" />
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-4 pt-2 border-t border-white/5">
                    <div className="space-y-1.5">
                      <label className="text-[10px] text-white/40 uppercase font-bold">Trail Offset %</label>
                      <input type="number" step="0.1" value={settings.trailingStopOffset * 100} onChange={(e) => setSettings({...settings, trailingStopOffset: parseFloat(e.target.value) / 100})} className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-xs outline-none focus:border-amber-500/50" />
                    </div>
                  </div>
                  <button onClick={startBacktest} disabled={loading || training || !model1hRef.current} className={cn("w-full py-3 rounded-xl font-semibold flex items-center justify-center gap-2 transition-all active:scale-95", !model1hRef.current ? "bg-white/5 text-white/50 cursor-not-allowed" : "bg-amber-500 hover:bg-amber-400 text-black shadow-[0_0_20px_rgba(245,158,11,0.2)]")}>
                    <Activity className="w-4 h-4" /> {!model1hRef.current ? 'Train Model First' : 'Run Backtest'}
                  </button>
                </div>
              </div>
            </div>

            {/* Results & Equity */}
            <div className="lg:col-span-3 space-y-6">
              {/* Detailed Metrics */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <MetricCard title="Total Profit" value={backtestResult ? `$${backtestResult.totalProfit.toLocaleString()}` : '---'} color="text-emerald-400" />
                <MetricCard title="Win Rate" value={backtestResult ? `${backtestResult.winRate.toFixed(1)}%` : '---'} color="text-blue-400" />
                <MetricCard title="Total Trades" value={backtestResult ? backtestResult.trades.length.toString() : '---'} color="text-purple-400" />
                <MetricCard title="Max Drawdown" value={backtestResult ? `${backtestResult.maxDrawdown.toFixed(2)}%` : '---'} color="text-red-400" />
                <MetricCard title="Avg Profit" value={backtestResult ? `$${backtestResult.avgProfit.toFixed(2)}` : '---'} color="text-emerald-500/70" />
                <MetricCard title="Avg Loss" value={backtestResult ? `$${backtestResult.avgLoss.toFixed(2)}` : '---'} color="text-red-500/70" />
                <MetricCard title="Max Profit" value={backtestResult ? `$${backtestResult.maxProfit.toFixed(2)}` : '---'} color="text-emerald-600" />
                <MetricCard title="Max Loss" value={backtestResult ? `$${backtestResult.maxLoss.toFixed(2)}` : '---'} color="text-red-600" />
              </div>

              {/* Equity Curve */}
              {backtestResult && (
                <div className="bg-[#0D0D0E] border border-white/5 rounded-2xl p-6 shadow-xl">
                  <h3 className="font-medium mb-6 flex items-center gap-2 text-amber-400">
                    <TrendingDown className="w-5 h-5" />
                    Equity Curve
                  </h3>
                  <div className="h-[250px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={equityData}>
                        <defs>
                          <linearGradient id="colorBalance" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.3}/>
                            <stop offset="95%" stopColor="#f59e0b" stopOpacity={0}/>
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#ffffff05" vertical={false} />
                        <XAxis dataKey="time" stroke="#ffffff30" fontSize={10} tickLine={false} axisLine={false} />
                        <YAxis domain={['auto', 'auto']} stroke="#ffffff30" fontSize={10} tickLine={false} axisLine={false} />
                        <Tooltip contentStyle={{ backgroundColor: '#0D0D0E', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }} />
                        <Area type="monotone" dataKey="balance" stroke="#f59e0b" fillOpacity={1} fill="url(#colorBalance)" />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              {/* Trade History */}
              <div className="bg-[#0D0D0E] border border-white/5 rounded-2xl p-6 shadow-xl">
                <h3 className="font-medium mb-4 flex items-center gap-2 text-purple-400">
                  <History className="w-4 h-4" />
                  Trade History
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-h-[600px] overflow-y-auto pr-2 custom-scrollbar">
                  {backtestResult && backtestResult.trades.length > 0 ? (
                    backtestResult.trades.slice().reverse().map((trade, idx) => (
                      <div key={idx} className="p-3 bg-white/5 rounded-xl border border-white/5 hover:border-white/10 transition-colors space-y-2">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <span className="text-[10px] font-bold text-amber-500 bg-amber-500/10 px-1.5 py-0.5 rounded">SHORT</span>
                            <span className={cn(
                              "text-[10px] font-bold px-1.5 py-0.5 rounded",
                              trade.exitReason === 'TAKE_PROFIT' ? "text-emerald-500 bg-emerald-500/10" :
                              trade.exitReason === 'STOP_LOSS' ? "text-red-500 bg-red-500/10" :
                              trade.exitReason === 'PREDICTION' ? "text-purple-500 bg-purple-500/10" :
                              "text-blue-500 bg-blue-500/10"
                            )}>
                              {trade.exitReason.replace('_', ' ')}
                            </span>
                          </div>
                          <div className="text-right">
                            <p className={cn("text-sm font-bold", trade.profit > 0 ? "text-emerald-500" : "text-red-500")}>
                              {trade.profitPct > 0 ? '+' : ''}{trade.profitPct.toFixed(2)}%
                            </p>
                          </div>
                        </div>
                        
                        <div className="grid grid-cols-2 gap-4 text-[10px]">
                          <div className="space-y-1">
                            <p className="text-white/40 uppercase font-bold tracking-wider">Entry</p>
                            <p className="text-white/80">{format(new Date(trade.entryTime), 'MM/dd HH:mm')}</p>
                            <p className="text-white font-mono font-medium">${trade.entryPrice.toLocaleString()}</p>
                          </div>
                          <div className="space-y-1 text-right">
                            <p className="text-white/40 uppercase font-bold tracking-wider">Exit</p>
                            <p className="text-white/80">{format(new Date(trade.exitTime), 'MM/dd HH:mm')}</p>
                            <p className="text-white font-mono font-medium">${trade.exitPrice.toLocaleString()}</p>
                          </div>
                        </div>

                        <div className="pt-2 border-t border-white/5 flex justify-between items-center">
                          <span className="text-[9px] text-white/30 italic">Duration: {((trade.exitTime - trade.entryTime) / (1000 * 60 * 60)).toFixed(1)}h</span>
                          <span className={cn("text-xs font-bold", trade.profit > 0 ? "text-emerald-500/70" : "text-red-500/70")}>
                            ${trade.profit.toFixed(2)}
                          </span>
                        </div>
                      </div>
                    ))
                  ) : (
                    <div className="col-span-full py-12 text-center text-white/20 italic">No trades recorded yet</div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </section>
        {/* Section 4: Model Library */}
        <section className="space-y-6">
          <div className="flex items-center gap-2 border-b border-white/5 pb-2">
            <History className="w-5 h-5 text-blue-500" />
            <h2 className="text-xl font-bold tracking-tight">4. Model Library</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {savedModels.length === 0 ? (
              <div className="col-span-full py-12 bg-[#0D0D0E] border border-white/5 rounded-2xl flex flex-col items-center justify-center text-white/20">
                <History className="w-8 h-8 mb-2 opacity-20" />
                <p className="text-sm">No saved models found. Train and save a model to see it here.</p>
              </div>
            ) : (
              savedModels.map((pair, idx) => (
                <div key={idx} className="bg-[#0D0D0E] border border-white/5 rounded-2xl p-5 shadow-xl hover:border-blue-500/30 transition-all group">
                  <div className="flex items-center justify-between mb-4">
                    <div className="w-10 h-10 bg-blue-500/10 rounded-xl flex items-center justify-center group-hover:bg-blue-500/20 transition-colors">
                      <Target className="w-5 h-5 text-blue-400" />
                    </div>
                    <button 
                      onClick={() => handleDeleteModel(pair.name)}
                      className="p-2 hover:bg-red-500/10 text-white/20 hover:text-red-400 rounded-lg transition-all"
                    >
                      <ShieldAlert className="w-4 h-4" />
                    </button>
                  </div>
                  <h4 className="font-bold text-white/80 mb-1 truncate">{pair.name}</h4>
                  <p className="text-[10px] text-white/30 mb-6">Saved: {format(new Date(pair.timestamp), 'MMM dd, HH:mm')}</p>
                  
                  <button 
                    onClick={() => handleLoadModel(pair.name)}
                    disabled={loading || training}
                    className="w-full py-2 bg-blue-600/10 hover:bg-blue-600 text-blue-400 hover:text-white border border-blue-600/20 rounded-lg text-xs font-bold transition-all"
                  >
                    Load Model
                  </button>
                </div>
              ))
            )}
          </div>
        </section>

        {/* Section 5: Live Paper Trading Monitor */}
        {isLiveMode && (
          <section className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-700">
            <div className="flex items-center gap-2 border-b border-emerald-500/20 pb-2">
              <Activity className="w-5 h-5 text-emerald-500" />
              <h2 className="text-xl font-bold tracking-tight">5. Live {isRealTrading ? 'Real' : 'Paper'} Trading Monitor</h2>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
              <div className="lg:col-span-1 space-y-4">
                <div className={cn(
                  "border rounded-2xl p-6 shadow-xl relative overflow-hidden",
                  isRealTrading ? "bg-red-950/20 border-red-500/20" : "bg-[#0D0D0E] border-emerald-500/20"
                )}>
                  <div className="absolute top-0 right-0 p-2">
                    <div className={cn("w-2 h-2 rounded-full animate-ping", isRealTrading ? "bg-red-500" : "bg-emerald-500")} />
                  </div>
                  <h3 className="text-xs font-bold text-white/40 uppercase mb-4">{isRealTrading ? 'Real Wallet Balance (USDT)' : 'Live Paper Balance'}</h3>
                  <div className={cn(
                    "text-3xl font-bold",
                    isRealTrading ? "text-red-400" : "text-emerald-400"
                  )}>
                    ${(isRealTrading ? (realBalance || 0) : livePaperBalance).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                  </div>
                  <div className="mt-2 text-[10px] text-white/30">{isRealTrading ? 'Connected to Delta Exchange' : 'Starting: $10,000.00'}</div>
                </div>

                <div className="bg-[#0D0D0E] border border-white/5 rounded-2xl p-6 shadow-xl">
                  <h3 className="text-xs font-bold text-white/40 uppercase mb-4">Live Signal & Parameters</h3>
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-white/60">Current Price</span>
                      <span className="text-xs font-mono font-bold text-emerald-400">
                        ${livePrice ? livePrice.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : '--'}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-white/60">Model Confidence</span>
                      <span className={cn(
                        "text-xs font-bold",
                        livePrediction && livePrediction > settings.threshold ? "text-amber-400" : "text-white/40"
                      )}>
                        {livePrediction ? (livePrediction * 100).toFixed(1) : '--'}%
                      </span>
                    </div>
                    <div className="w-full bg-white/5 h-1.5 rounded-full overflow-hidden">
                      <div 
                        className={cn(
                          "h-full transition-all duration-500",
                          livePrediction && livePrediction > settings.threshold ? "bg-amber-500" : "bg-white/20"
                        )}
                        style={{ width: `${(livePrediction || 0) * 100}%` }}
                      />
                    </div>

                    {liveParams && (
                      <div className="grid grid-cols-2 gap-x-4 gap-y-2 pt-4 border-t border-white/5">
                        <div className="flex justify-between items-center">
                          <span className="text-[10px] text-white/30 uppercase">RSI</span>
                          <span className="text-[10px] font-mono text-white/60">{liveParams.rsi.toFixed(1)}</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-[10px] text-white/30 uppercase">EMA9/20</span>
                          <span className={cn("text-[10px] font-mono", liveParams.emaCross ? "text-emerald-400" : "text-red-400")}>
                            {liveParams.emaCross ? 'UP' : 'DOWN'}
                          </span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-[10px] text-white/30 uppercase">MACD Hist</span>
                          <span className="text-[10px] font-mono text-white/60">{liveParams.macdHist.toFixed(2)}</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-[10px] text-white/30 uppercase">StochRSI</span>
                          <span className="text-[10px] font-mono text-white/60">{liveParams.stochRsi.toFixed(2)}</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-[10px] text-white/30 uppercase">ATR</span>
                          <span className="text-[10px] font-mono text-white/60">{liveParams.atr.toFixed(1)}</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-[10px] text-white/30 uppercase">4h Pred</span>
                          <span className="text-[10px] font-mono text-white/60">{(liveParams.secondaryPrediction * 100).toFixed(1)}%</span>
                        </div>
                        <div className="col-span-2 flex justify-between items-center pt-2 border-t border-white/5 opacity-40">
                          <span className="text-[9px] uppercase tracking-tighter">Last Updated</span>
                          <span className="text-[9px] font-mono">{lastLiveUpdate ? format(lastLiveUpdate, 'HH:mm:ss') : '--:--:--'}</span>
                        </div>
                      </div>
                    )}

                    <div className="flex items-center justify-between pt-2 border-t border-white/5">
                      <span className="text-xs text-white/60">Last Updated</span>
                      <span className="text-[10px] font-mono text-white/40">
                        {lastLiveUpdate ? format(lastLiveUpdate, 'HH:mm:ss') : '--'}
                      </span>
                    </div>
                    <div className="flex items-center justify-between pt-2 border-t border-white/5">
                      <span className="text-xs text-white/60">Status</span>
                      <span className={cn(
                        "text-[10px] font-bold px-2 py-0.5 rounded uppercase",
                        activeLiveTrade ? "bg-amber-500/10 text-amber-500" : "bg-white/5 text-white/40"
                      )}>
                        {activeLiveTrade ? 'In Trade (SHORT)' : 'Waiting for Signal'}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="lg:col-span-3">
                <div className="bg-[#0D0D0E] border border-white/5 rounded-2xl p-6 shadow-xl h-full">
                  <h3 className="font-medium mb-4 flex items-center gap-2 text-emerald-400">
                    <History className="w-4 h-4" />
                    Live Trade History
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-h-[400px] overflow-y-auto pr-2 custom-scrollbar">
                    {liveTrades.length > 0 ? (
                      liveTrades.map((trade, idx) => (
                        <div key={idx} className="p-3 bg-white/5 rounded-xl border border-white/5 hover:border-white/10 transition-colors space-y-2">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <span className="text-[10px] font-bold text-amber-500 bg-amber-500/10 px-1.5 py-0.5 rounded">SHORT</span>
                              <span className={cn(
                                "text-[10px] font-bold px-1.5 py-0.5 rounded",
                                trade.exitReason === 'TAKE_PROFIT' ? "text-emerald-500 bg-emerald-500/10" :
                                trade.exitReason === 'STOP_LOSS' ? "text-red-500 bg-red-500/10" :
                                trade.exitReason === 'PREDICTION' ? "text-purple-500 bg-purple-500/10" :
                                "text-blue-500 bg-blue-500/10"
                              )}>
                                {trade.exitReason.replace('_', ' ')}
                              </span>
                            </div>
                            <span className={cn("text-xs font-bold", trade.profit >= 0 ? "text-emerald-500" : "text-red-500")}>
                              {trade.profit >= 0 ? '+' : ''}{trade.profit.toFixed(2)}
                            </span>
                          </div>
                          <div className="grid grid-cols-2 gap-4 text-[10px]">
                            <div className="space-y-1">
                              <div className="text-white/30 uppercase">Entry</div>
                              <div className="text-white/70 font-medium">${trade.entryPrice.toLocaleString()}</div>
                            </div>
                            <div className="space-y-1">
                              <div className="text-white/30 uppercase">Exit</div>
                              <div className="text-white/70 font-medium">${trade.exitPrice.toLocaleString()}</div>
                            </div>
                          </div>
                        </div>
                      ))
                    ) : (
                      <div className="col-span-full flex flex-col items-center justify-center py-12 text-white/20">
                        <Activity className="w-8 h-8 mb-2 opacity-20" />
                        <p className="text-sm">No live trades yet</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </section>
        )}
      </main>

      <style dangerouslySetInnerHTML={{ __html: `
        .custom-scrollbar::-webkit-scrollbar {
          width: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(255, 255, 255, 0.1);
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(255, 255, 255, 0.2);
        }
      `}} />
      <Terminal />
    </div>
  );
}

function StatCard({ title, value, icon, trend }: { title: string, value: string, icon: React.ReactNode, trend?: 'up' | 'down' }) {
  return (
    <div className="bg-[#0D0D0E] border border-white/5 rounded-2xl p-5 shadow-lg group hover:border-white/10 transition-all h-full flex flex-col justify-center">
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs font-medium text-white/50">{title}</span>
        <div className="p-2 bg-white/5 rounded-lg group-hover:scale-110 transition-transform">
          {icon}
        </div>
      </div>
      <div className="flex items-end justify-between">
        <span className="text-2xl font-bold tracking-tight">{value}</span>
        {trend && (
          <div className={cn(
            "flex items-center gap-0.5 text-[10px] font-bold px-1.5 py-0.5 rounded",
            trend === 'up' ? "text-emerald-500 bg-emerald-500/10" : "text-red-500 bg-red-500/10"
          )}>
            {trend === 'up' ? <ArrowUpRight className="w-3 h-3" /> : <ArrowDownRight className="w-3 h-3" />}
            {trend === 'up' ? 'UP' : 'DOWN'}
          </div>
        )}
      </div>
    </div>
  );
}

function MetricCard({ title, value, color }: { title: string, value: string, color: string }) {
  return (
    <div className="bg-white/5 border border-white/5 rounded-xl p-4 hover:border-white/10 transition-colors">
      <p className="text-[10px] text-white/40 uppercase font-bold tracking-wider mb-1">{title}</p>
      <p className={cn("text-lg font-bold tracking-tight", color)}>{value}</p>
    </div>
  );
}
