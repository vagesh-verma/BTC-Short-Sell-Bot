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
  Trash2,
  Activity, 
  BarChart3, 
  Play, 
  RefreshCw, 
  AlertCircle, 
  ArrowDownRight,
  ArrowUpRight,
  Wallet,
  History,
  Settings,
  Settings2,
  ShieldAlert,
  Target,
  Zap,
  Save,
  ChevronDown,
  CloudUpload,
  CloudOff,
  Github
} from 'lucide-react';
import { format } from 'date-fns';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

import { fetchBTCData, Candle } from './services/dataService';
import { GRUModel, prepareData } from './services/modelService';
import { runBacktest, BacktestResult, Trade, BacktestSettings } from './services/backtestService';
import { calculateEMA, calculateRSI, calculateBollingerBands, calculateMACD, calculateStochasticRSI, calculateATR, calculateEMACross, calculateOBV, calculateMFI, calculateVolatility, calculateBearishHarami, calculateMarubozu, calculateEngulfing } from './services/indicatorService';
import { DeltaSocketService, TickerUpdate } from './services/deltaSocketService';
import { getSavedModelPairs, saveModelPair, loadModelPair, deleteModelPair, ModelPair, uploadModelPairToGitHub, deleteModelPairFromGitHub, loadModelPairFromGitHub, syncModelsFromGitHub } from './services/storageService';
import { GitHubConfig } from './services/githubService';

import { Terminal } from './components/Terminal';
import { logger } from './services/loggerService';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

const FEATURE_NAMES = [
  'Price', 'RSI', 'EMA', 'BB_Upper', 'BB_Lower', 'MACD_Hist', 'Stoch_RSI', 'ATR', 'Secondary_Pred',
  'EMA9', 'Below', 'Cross', 'OBV', 'MFI', 'Volatility', 'Hour', 'Asia', 'London', 'NY', 'Day',
  'Harami', 'Marubozu', 'Engulfing'
];

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
  const [trainingStats1h, setTrainingStats1h] = useState<{total: number, positive: number, negative: number} | null>(null);
  const [trainingStats4h, setTrainingStats4h] = useState<{total: number, positive: number, negative: number} | null>(null);
  const [predictions, setPredictions] = useState<number[]>([]);
  const [predictionStats, setPredictionStats] = useState<{total: number, aboveThreshold: number} | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<{ name: string, type: 'local' | 'github' | 'complete' } | null>(null);
  
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
  const [dropThreshold4h, setDropThreshold4h] = useState<number>(0.5);
  const [modelHyperparams, setModelHyperparams] = useState({
    units: 128,
    dropout: 0.2,
    learningRate: 0.001
  });
  const [indicatorPeriods, setIndicatorPeriods] = useState({
    rsi: 14,
    ema: 20,
    ema9: 9,
    bb: 20,
    mfi: 14,
    volatility: 20
  });

  // Backtest & Live Settings
  const [settings, setSettings] = useState<BacktestSettings>({
    threshold: 0.2,
    exitThreshold: 0,
    stopLoss: 0.01, // 1%
    takeProfit: 0.02, // 2%
    trailingStopActivation: 0.005, // 0.5%
    trailingStopOffset: 0.003, // 0.3%
    maxDurationHours: 12,
    quantity: 200,
    quantityType: 'LOTS',
    useSessionTrading: false,
    asiaStart: 2,
    asiaEnd: 5,
    nyStart: 13,
    nyEnd: 18,
    useOnlyCompletedCandles: false,
    minSignalVelocity: 0.1,
    mcPasses: 1,
    maxUncertainty: 0.1,
    strategyType: 'SHORT_BTC',
    shortCallDelta: 0.3,
    longCallDelta: 0.1,
    dailyProfitLimit: 0,
    dailyLossLimit: 0,
  });

  const [isRealTrading, setIsRealTrading] = useState(false);
  const [isLiveMode, setIsLiveMode] = useState(false);
  const [isTogglingTrading, setIsTogglingTrading] = useState(false);
  const [serverTradingStatus, setServerTradingStatus] = useState<any>(null);
  const [isSyncing, setIsSyncing] = useState(false);
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
    prediction: number;
    features: Record<string, number>;
    pnl?: number;
    pnlPct?: number;
    size?: number;
  } | null>(null);
  const [livePrediction, setLivePrediction] = useState<number | null>(null);
  const hasSyncedFromServer = useRef(false);
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
    obv: number;
    mfi: number;
    volatility: number;
    harami: number;
    marubozu: number;
    engulfing: number;
    session: string;
    dayOfWeek: number;
  } | null>(null);

  const [savedModels, setSavedModels] = useState<ModelPair[]>([]);
  const [newModelName, setNewModelName] = useState('');
  const [githubConfig, setGithubConfig] = useState<GitHubConfig>({
    owner: '',
    repo: '',
    path: 'models',
    token: ''
  });

  const windowSize = 20;

  useEffect(() => {
    setSavedModels(getSavedModelPairs());
    // Load GitHub config from localStorage if exists
    const savedConfig = localStorage.getItem('github_config');
    if (savedConfig) {
      const config = JSON.parse(savedConfig);
      setGithubConfig(config);
      
      // Auto-sync on load if config is complete
      if (config.owner && config.repo && config.token) {
        syncModelsFromGitHub(config).then(updated => {
          setSavedModels(updated);
        }).catch(err => {
          console.error('Initial GitHub sync failed:', err);
        });
      }
    }
  }, []);

  const handleSaveGithubConfig = async () => {
    localStorage.setItem('github_config', JSON.stringify(githubConfig));
    logger.success('GitHub configuration saved.');
    
    if (githubConfig.owner && githubConfig.repo && githubConfig.token) {
      try {
        setLoading(true);
        setStatus('Syncing models from GitHub...');
        const updated = await syncModelsFromGitHub(githubConfig);
        setSavedModels(updated);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
  };

  const handleUploadToGitHub = async (name: string) => {
    if (!githubConfig.owner || !githubConfig.repo || !githubConfig.token) {
      logger.error('Please configure GitHub settings first.');
      return;
    }
    try {
      setLoading(true);
      setStatus(`Uploading "${name}" to GitHub...`);
      await uploadModelPairToGitHub(name, githubConfig);
      setSavedModels(getSavedModelPairs());
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteFromGitHub = async (name: string) => {
    if (!githubConfig.owner || !githubConfig.repo || !githubConfig.token) {
      logger.error('Please configure GitHub settings first.');
      return;
    }
    setConfirmDelete({ name, type: 'github' });
  };

  const executeDeleteFromGitHub = async (name: string) => {
    try {
      setLoading(true);
      setStatus(`Deleting "${name}" from GitHub...`);
      await deleteModelPairFromGitHub(name, githubConfig);
      setSavedModels(getSavedModelPairs());
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
      setConfirmDelete(null);
    }
  };

  const handleLoadFromGitHub = async (name: string) => {
    if (!githubConfig.owner || !githubConfig.repo || !githubConfig.token) {
      logger.error('Please configure GitHub settings first.');
      return;
    }
    try {
      setLoading(true);
      setStatus(`Loading "${name}" from GitHub...`);
      const { model1h, model4h } = await loadModelPairFromGitHub(name, githubConfig);
      model1hRef.current = model1h;
      model4hRef.current = model4h;
      setStatus(`Model "${name}" loaded from GitHub!`);
      // Trigger a re-render to update UI buttons
      setPredictions([]); 
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

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

  const fetchStatus = async () => {
    try {
      const res = await fetch('/api/trading/status');
      if (res.ok) {
        const data = await res.json();
        setServerTradingStatus(data);
        setIsLiveMode(data.isRunning);
        // Only sync isRealTrading from server if the bot is actually running
        // This prevents the UI from resetting to "Paper Only" while the bot is stopped
        if (data.isRunning) {
          setIsRealTrading(data.isRealTrading);
        }
        if (data.lastPrice) setLivePrice(data.lastPrice);
        if (data.lastPrediction !== null) setLivePrediction(data.lastPrediction);
        if (data.lastParams) setLiveParams(data.lastParams);
        if (data.lastUpdate) setLastLiveUpdate(new Date(data.lastUpdate));
        if (data.closedTrades) setLiveTrades(data.closedTrades);
        
        // Sync settings from server on first successful load
        if (!hasSyncedFromServer.current && data.settings) {
          setSettings(prev => ({ ...prev, ...data.settings }));
          hasSyncedFromServer.current = true;
          logger.info('Trading settings synchronized from server');
        }
      }
    } catch (e) {
      console.error('Failed to fetch trading status');
    }
  };

  useEffect(() => {
    let interval: NodeJS.Timeout;
    fetchStatus();
    interval = setInterval(fetchStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  const pushSettingsToServer = async () => {
    try {
      const res = await fetch('/api/trading/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ settings })
      });
      if (res.ok) {
        logger.success('Settings pushed to server and persisted');
      } else {
        logger.error('Failed to push settings to server');
      }
    } catch (e) {
      logger.error('Failed to push settings to server');
    }
  };

  const syncModelToServer = async () => {
    if (!model1hRef.current || !model4hRef.current) {
      logger.error('No models loaded to sync.');
      return;
    }

    try {
      setIsSyncing(true);
      setStatus('Syncing models to server...');
      
      const model1hArtifacts = await model1hRef.current.getArtifacts();
      const model4hArtifacts = await model4hRef.current.getArtifacts();
      
      // Convert ArrayBuffer to Base64 for JSON serialization
      const arrayBufferToBase64 = (buffer: ArrayBuffer) => {
        const bytes = new Uint8Array(buffer);
        let binary = '';
        for (let i = 0; i < bytes.byteLength; i++) {
          binary += String.fromCharCode(bytes[i]);
        }
        return window.btoa(binary);
      };

      if (model1hArtifacts.weightData instanceof ArrayBuffer) {
        (model1hArtifacts as any).weightData = arrayBufferToBase64(model1hArtifacts.weightData);
      }
      if (model4hArtifacts.weightData instanceof ArrayBuffer) {
        (model4hArtifacts as any).weightData = arrayBufferToBase64(model4hArtifacts.weightData);
      }

      const metadata1h = JSON.parse(localStorage.getItem(`${model1hRef.current.name}_metadata`) || '{"windowSize":20,"featureCount":23}');
      const metadata4h = JSON.parse(localStorage.getItem(`${model4hRef.current.name}_metadata`) || '{"windowSize":20,"featureCount":22}');

      const res = await fetch('/api/trading/sync-model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model1hArtifacts,
          model4hArtifacts,
          metadata1h,
          metadata4h
        })
      });

      if (res.ok) {
        logger.success('Models synced to server successfully!');
      } else {
        const data = await res.json();
        logger.error(`Sync failed: ${data.error}`);
      }
    } catch (err) {
      console.error(err);
      logger.error('Error syncing models to server.');
    } finally {
      setIsSyncing(false);
      setStatus('Ready');
    }
  };

  const toggleServerTrading = async () => {
    if (isTogglingTrading) return;
    try {
      setIsTogglingTrading(true);
      if (isLiveMode) {
        await fetch('/api/trading/stop', { method: 'POST' });
        logger.info('Server-side trading stopped.');
      } else {
        if (!serverTradingStatus?.hasModels) {
          logger.error('Please sync models to server first.');
          return;
        }
        await fetch('/api/trading/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ settings, isRealTrading })
        });
        logger.success('Server-side trading started!');
      }
      await fetchStatus();
    } catch (err) {
      logger.error('Failed to toggle server trading.');
    } finally {
      setIsTogglingTrading(false);
    }
  };

  useEffect(() => {
    if (serverTradingStatus?.activeTrade) {
      setActiveLiveTrade(serverTradingStatus.activeTrade);
    } else {
      setActiveLiveTrade(null);
    }
  }, [serverTradingStatus?.activeTrade]);

  useEffect(() => {
    loadData();
  }, [trainingRange]);

  useEffect(() => {
    // Server-side trading handles the WebSocket and predictions
    return;
  }, [isLiveMode]);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isRealTrading) {
      fetchRealBalance();
      interval = setInterval(fetchRealBalance, 30000); // Fetch every 30 seconds
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isRealTrading]);

  const fetchRealBalance = async () => {
    try {
      const response = await fetch('/api/wallet');
      const data = await response.json();
      if (data.success && data.result) {
        // Find USD or BTC balance
        const usd = data.result.find((b: any) => b.asset_symbol === 'USD');
        if (usd) {
          setRealBalance(parseFloat(usd.available_balance));
        } else {
          logger.warning('USD balance not found in wallet.');
        }
      } else if (data.error) {
        logger.error(`Delta API Error: ${data.error}`);
      }
    } catch (error: any) {
      console.error('Error fetching real balance:', error);
      logger.error(`Failed to fetch wallet balance: ${error.message}`);
    }
  };

  const placeRealOrder = async (side: 'buy' | 'sell', quantity: number) => {
    try {
      let lots = 0;
      if (settings.quantityType === 'LOTS') {
        lots = quantity;
      } else if (settings.quantityType === 'BTC') {
        lots = quantity * 1000;
      } else if (settings.quantityType === 'USD' && livePrice) {
        // 1 lot = 0.001 BTC. So lots = (USD / price) / 0.001
        lots = (quantity / livePrice) / 0.001;
      }

      const finalSize = Math.max(1, Math.floor(lots));

      const response = await fetch('/api/order', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol: 'BTCUSD',
          side,
          order_type: 'limit_order',
          size: finalSize,
          limit_price: livePrice
        })
      });
      const data = await response.json();
      if (data.result) {
        logger.success(`Real order placed: ${side} ${finalSize} lots at $${livePrice}`);
        fetchRealBalance();
      } else {
        logger.error(`Real order failed: ${data.error || 'Unknown error'}`);
      }
    } catch (error) {
      logger.error(`Error placing real order: ${error}`);
    }
  };

  // Use a ref to store the latest processLiveUpdate function to avoid stale closures in the socket listener
  const processLiveUpdateRef = useRef<((price: number) => Promise<void>) | null>(null);

  const processLiveUpdate = async (price: number) => {
    if (!model1hRef.current || !model4hRef.current || candles.length < windowSize || candles4h.length < windowSize) return;

    const now = new Date();
    
    // 0. Check for completed candle entry if enabled
    if (settings.useOnlyCompletedCandles && !activeLiveTrade) {
      const lastCandleTime = candles[candles.length - 1].time;
      const oneHourInMs = 60 * 60 * 1000;
      const timeSinceLastCandle = now.getTime() - lastCandleTime;
      
      // If we are more than 1 minute into the new candle, it's not the "completion" moment
      // We want to trigger as close to the start of the new hour as possible.
      // However, the socket might not fire exactly at the hour.
      // A better way is to check if the current time has crossed into a new hour compared to the last candle.
      const currentHourStart = new Date(now.getFullYear(), now.getMonth(), now.getDate(), now.getHours()).getTime();
      if (currentHourStart <= lastCandleTime) {
        // Still in the same hour as the last candle
        return;
      }
    }

    // 1. Calculate 4h prediction as a feature
    const prices4h = candles4h.map(c => c.close);
    const highs4h = candles4h.map(c => c.high);
    const lows4h = candles4h.map(c => c.low);
    const volumes4h = candles4h.map(c => c.volume);
    const hours4h = candles4h.map(c => new Date(c.time).getHours());
    const days4h = candles4h.map(c => new Date(c.time).getDay());
    
    const rsi4h = calculateRSI(prices4h, indicatorPeriods.rsi);
    const ema4h = calculateEMA(prices4h, indicatorPeriods.ema);
    const ema9_4h = calculateEMA(prices4h, indicatorPeriods.ema9);
    const bb4h = calculateBollingerBands(prices4h, indicatorPeriods.bb);
    const macd4h = calculateMACD(prices4h);
    const stochRsi4h = calculateStochasticRSI(rsi4h);
    const atr4h = calculateATR(highs4h, lows4h, prices4h);
    const cross4h = calculateEMACross(ema9_4h, ema4h);
    const obv4h = calculateOBV(prices4h, volumes4h);
    const mfi4h = calculateMFI(highs4h, lows4h, prices4h, volumes4h, indicatorPeriods.mfi);
    const vol4h = calculateVolatility(prices4h, indicatorPeriods.volatility);
    const opens4h = candles4h.map(c => c.open);
    const harami4h = calculateBearishHarami(opens4h, prices4h);
    const marubozu4h = calculateMarubozu(opens4h, highs4h, lows4h, prices4h);
    const engulfing4h = calculateEngulfing(opens4h, prices4h);

    const normalize = (arr: number[]) => {
      const min = Math.min(...arr);
      const max = Math.max(...arr);
      return arr.map(v => (max === min ? 0 : (v - min) / (max - min)));
    };

    const last4hIdx = prices4h.length - 1;
    const windowIndices4h = Array.from({ length: windowSize }, (_, k) => last4hIdx - windowSize + 1 + k);
    
    const pWindow4h = windowIndices4h.map(idx => prices4h[idx]);
    const rWindow4h = windowIndices4h.map(idx => rsi4h[idx]);
    const eWindow4h = windowIndices4h.map(idx => ema4h[idx]);
    const uWindow4h = windowIndices4h.map(idx => bb4h.upper[idx]);
    const lWindow4h = windowIndices4h.map(idx => bb4h.lower[idx]);
    const mWindow4h = windowIndices4h.map(idx => macd4h.histogram[idx]);
    const sWindow4h = windowIndices4h.map(idx => stochRsi4h[idx]);
    const aWindow4h = windowIndices4h.map(idx => atr4h[idx]);
    const e9Window4h = windowIndices4h.map(idx => ema9_4h[idx]);
    const belowWindow4h = windowIndices4h.map(idx => cross4h.isBelow[idx]);
    const crossWindow4h = windowIndices4h.map(idx => cross4h.isCross[idx]);
    const obvWindow4h = windowIndices4h.map(idx => obv4h[idx]);
    const mfiWindow4h = windowIndices4h.map(idx => mfi4h[idx]);
    const volWindow4h = windowIndices4h.map(idx => vol4h[idx]);
    const hourWindow4h = windowIndices4h.map(idx => hours4h[idx]);

    const np4h = normalize(pWindow4h);
    const nr4h = rWindow4h.map(v => v / 100);
    const ne4h = normalize(eWindow4h);
    const nu4h = normalize(uWindow4h);
    const nl4h = normalize(lWindow4h);
    const nm4h = normalize(mWindow4h);
    const na4h = normalize(aWindow4h);
    const ne9_4h = normalize(e9Window4h);
    const nobv4h = normalize(obvWindow4h);
    const nvol4h = normalize(volWindow4h);
    const nday4h = windowIndices4h.map(idx => days4h[idx]);
    const nharami4h = windowIndices4h.map(idx => harami4h[idx]);
    const nmarubozu4h = windowIndices4h.map(idx => marubozu4h[idx]);
    const nengulfing4h = windowIndices4h.map(idx => engulfing4h[idx]);

    const x4h: number[] = [];
    for (let j = 0; j < windowSize; j++) {
      const h = hourWindow4h[j];
      x4h.push(
        np4h[j], nr4h[j], ne4h[j], nu4h[j], nl4h[j],
        nm4h[j], sWindow4h[j], na4h[j], ne9_4h[j],
        belowWindow4h[j], crossWindow4h[j], nobv4h[j],
        mfiWindow4h[j] / 100, nvol4h[j],
        h / 24, h >= 0 && h <= 9 ? 1 : 0, h >= 8 && h <= 17 ? 1 : 0, h >= 13 && h <= 22 ? 1 : 0, nday4h[j] / 7,
        nharami4h[j], nmarubozu4h[j], nengulfing4h[j]
      );
    }
    const secondaryPrediction = model4hRef.current.predict(x4h);

    // 2. Calculate 1h prediction
    const fullPrices1h = [...candles.map(c => c.close), price];
    const fullHighs1h = [...candles.map(c => c.high), price];
    const fullLows1h = [...candles.map(c => c.low), price];
    const fullVolumes1h = [...candles.map(c => c.volume), 0];
    const fullHours1h = [...candles.map(c => new Date(c.time).getHours()), now.getHours()];
    const fullDays1h = [...candles.map(c => new Date(c.time).getDay()), now.getDay()];
    
    const rsi1h = calculateRSI(fullPrices1h, indicatorPeriods.rsi);
    const ema1h = calculateEMA(fullPrices1h, indicatorPeriods.ema);
    const ema9_1h = calculateEMA(fullPrices1h, indicatorPeriods.ema9);
    const bb1h = calculateBollingerBands(fullPrices1h, indicatorPeriods.bb);
    const macd1h = calculateMACD(fullPrices1h);
    const stochRsi1h = calculateStochasticRSI(rsi1h);
    const atr1h = calculateATR(fullHighs1h, fullLows1h, fullPrices1h);
    const cross1h = calculateEMACross(ema9_1h, ema1h);
    const obv1h = calculateOBV(fullPrices1h, fullVolumes1h);
    const mfi1h = calculateMFI(fullHighs1h, fullLows1h, fullPrices1h, fullVolumes1h, indicatorPeriods.mfi);
    const vol1h = calculateVolatility(fullPrices1h, indicatorPeriods.volatility);
    const opens1h = [...candles.map(c => c.open), price];
    const harami1h = calculateBearishHarami(opens1h, fullPrices1h);
    const marubozu1h = calculateMarubozu(opens1h, fullHighs1h, fullLows1h, fullPrices1h);
    const engulfing1h = calculateEngulfing(opens1h, fullPrices1h);

    const last1hIdx = fullPrices1h.length - 1;
    const windowIndices1h = Array.from({ length: windowSize }, (_, k) => last1hIdx - windowSize + 1 + k);

    const pWindow1h = windowIndices1h.map(idx => fullPrices1h[idx]);
    const rWindow1h = windowIndices1h.map(idx => rsi1h[idx]);
    const eWindow1h = windowIndices1h.map(idx => ema1h[idx]);
    const uWindow1h = windowIndices1h.map(idx => bb1h.upper[idx]);
    const lWindow1h = windowIndices1h.map(idx => bb1h.lower[idx]);
    const mWindow1h = windowIndices1h.map(idx => macd1h.histogram[idx]);
    const sWindow1h = windowIndices1h.map(idx => stochRsi1h[idx]);
    const aWindow1h = windowIndices1h.map(idx => atr1h[idx]);
    const e9Window1h = windowIndices1h.map(idx => ema9_1h[idx]);
    const belowWindow1h = windowIndices1h.map(idx => cross1h.isBelow[idx]);
    const crossWindow1h = windowIndices1h.map(idx => cross1h.isCross[idx]);
    const obvWindow1h = windowIndices1h.map(idx => obv1h[idx]);
    const mfiWindow1h = windowIndices1h.map(idx => mfi1h[idx]);
    const volWindow1h = windowIndices1h.map(idx => vol1h[idx]);
    const hourWindow1h = windowIndices1h.map(idx => fullHours1h[idx]);

    const np1h = normalize(pWindow1h);
    const nr1h = rWindow1h.map(v => v / 100);
    const ne1h = normalize(eWindow1h);
    const nu1h = normalize(uWindow1h);
    const nl1h = normalize(lWindow1h);
    const nm1h = normalize(mWindow1h);
    const na1h = normalize(aWindow1h);
    const ne9_1h = normalize(e9Window1h);
    const nobv1h = normalize(obvWindow1h);
    const nvol1h = normalize(volWindow1h);
    const nday1h = windowIndices1h.map(idx => fullDays1h[idx]);
    const nharami1h = windowIndices1h.map(idx => harami1h[idx]);
    const nmarubozu1h = windowIndices1h.map(idx => marubozu1h[idx]);
    const nengulfing1h = windowIndices1h.map(idx => engulfing1h[idx]);

    const x1h: number[] = [];
    for (let j = 0; j < windowSize; j++) {
      const h = hourWindow1h[j];
      x1h.push(
        np1h[j], nr1h[j], ne1h[j], nu1h[j], nl1h[j],
        nm1h[j], sWindow1h[j], na1h[j], secondaryPrediction,
        ne9_1h[j], belowWindow1h[j], crossWindow1h[j],
        nobv1h[j], mfiWindow1h[j] / 100, nvol1h[j],
        h / 24, h >= 0 && h <= 9 ? 1 : 0, h >= 8 && h <= 17 ? 1 : 0, h >= 13 && h <= 22 ? 1 : 0, nday1h[j] / 7,
        nharami1h[j], nmarubozu1h[j], nengulfing1h[j]
      );
    }

    const featureNames = [
      'Price', 'RSI', 'EMA', 'BB_Upper', 'BB_Lower', 'MACD_Hist', 'Stoch_RSI', 'ATR', 'Secondary_Pred',
      'EMA9', 'Below', 'Cross', 'OBV', 'MFI', 'Volatility', 'Hour', 'Asia', 'London', 'NY', 'Day',
      'Harami', 'Marubozu', 'Engulfing'
    ];

    const currentFeatures: Record<string, number> = {};
    const last1hFeatures = [
      np1h[windowSize - 1], nr1h[windowSize - 1], ne1h[windowSize - 1], nu1h[windowSize - 1], nl1h[windowSize - 1],
      nm1h[windowSize - 1], sWindow1h[windowSize - 1], na1h[windowSize - 1], secondaryPrediction,
      ne9_1h[windowSize - 1], belowWindow1h[windowSize - 1], crossWindow1h[windowSize - 1],
      nobv1h[windowSize - 1], mfiWindow1h[windowSize - 1] / 100, nvol1h[windowSize - 1],
      hourWindow1h[windowSize - 1] / 24, 
      hourWindow1h[windowSize - 1] >= 0 && hourWindow1h[windowSize - 1] <= 9 ? 1 : 0,
      hourWindow1h[windowSize - 1] >= 8 && hourWindow1h[windowSize - 1] <= 17 ? 1 : 0,
      hourWindow1h[windowSize - 1] >= 13 && hourWindow1h[windowSize - 1] <= 22 ? 1 : 0,
      nday1h[windowSize - 1] / 7,
      nharami1h[windowSize - 1], nmarubozu1h[windowSize - 1], nengulfing1h[windowSize - 1]
    ];

    featureNames.forEach((name, idx) => {
      currentFeatures[name] = last1hFeatures[idx];
    });

    const prediction = model1hRef.current.predict(x1h);
    setLivePrediction(prediction);
    setLivePrice(price);
    setLastLiveUpdate(new Date());

    setLiveParams({
      rsi: rsi1h[rsi1h.length - 1],
      ema: ema1h[ema1h.length - 1],
      ema9: ema9_1h[ema9_1h.length - 1],
      bbUpper: bb1h.upper[bb1h.upper.length - 1],
      bbLower: bb1h.lower[bb1h.lower.length - 1],
      macdHist: macd1h.histogram[macd1h.histogram.length - 1],
      stochRsi: stochRsi1h[stochRsi1h.length - 1],
      atr: atr1h[atr1h.length - 1],
      emaCross: cross1h.isBelow[cross1h.isBelow.length - 1] === 1,
      secondaryPrediction: secondaryPrediction,
      obv: obv1h[obv1h.length - 1],
      mfi: mfi1h[mfi1h.length - 1],
      volatility: vol1h[vol1h.length - 1],
      harami: harami1h[harami1h.length - 1],
      marubozu: marubozu1h[marubozu1h.length - 1],
      engulfing: engulfing1h[engulfing1h.length - 1],
      session: now.getUTCHours() >= settings.asiaStart && now.getUTCHours() < settings.asiaEnd ? 'Asia' : 
               now.getUTCHours() >= settings.nyStart && now.getUTCHours() < settings.nyEnd ? 'New York' : 'Off-Session',
      dayOfWeek: now.getDay()
    });

    // 3. Live Paper Trading Logic
    if (activeLiveTrade) {
      const profitPct = (activeLiveTrade.entryPrice - price) / activeLiveTrade.entryPrice;
      const stopLossPrice = activeLiveTrade.entryPrice * (1 + settings.stopLoss);
      const takeProfitPrice = activeLiveTrade.entryPrice * (1 - settings.takeProfit);

      let currentTrailingStop = activeLiveTrade.trailingStopPrice;
      let highestProfit = activeLiveTrade.highestProfitPct;

      // Update trailing stop
      if (profitPct > highestProfit) {
        highestProfit = profitPct;
        if (profitPct >= settings.trailingStopActivation) {
          currentTrailingStop = price * (1 + settings.trailingStopOffset);
        }
      }

      let shouldExit = false;
      let reason: Trade['exitReason'] = 'TIME';

      if (price >= stopLossPrice) {
        shouldExit = true;
        reason = 'STOP_LOSS';
      } else if (price <= takeProfitPrice) {
        shouldExit = true;
        reason = 'TAKE_PROFIT';
      } else if (currentTrailingStop && price >= currentTrailingStop) {
        shouldExit = true;
        reason = 'TRAILING_STOP';
      } else if (prediction < settings.exitThreshold) {
        shouldExit = true;
        reason = 'PREDICTION';
      }

      if (shouldExit) {
        const btcQuantity = settings.quantityType === 'LOTS' ? settings.quantity * 0.001 : settings.quantity;
        const profit = settings.quantityType === 'USD' 
          ? settings.quantity * profitPct 
          : btcQuantity * (activeLiveTrade.entryPrice - price);
          
        const newTrade: Trade = {
          type: 'SHORT',
          entryPrice: activeLiveTrade.entryPrice,
          exitPrice: price,
          entryTime: activeLiveTrade.entryTime,
          exitTime: Date.now(),
          profit,
          profitPct: profitPct * 100,
          exitReason: reason,
          prediction: activeLiveTrade.prediction,
          features: activeLiveTrade.features
        };
        setLiveTrades(prev => [newTrade, ...prev]);
        setLivePaperBalance(prev => prev + profit);
        setActiveLiveTrade(null);
        logger.success(`Live Trade Closed: ${reason} | Profit: $${profit.toFixed(2)}`);
        
        if (isRealTrading) {
          placeRealOrder('buy', settings.quantity); // Close SHORT with BUY
        }
      } else {
        // Update active trade state
        setActiveLiveTrade({
          ...activeLiveTrade,
          highestProfitPct: highestProfit,
          trailingStopPrice: currentTrailingStop
        });
      }
    } else if (prediction > settings.threshold) {
      // Check for session-based trading
      let canTrade = true;
      const hour = now.getUTCHours();

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

      if (!canTrade) return;

      setActiveLiveTrade({
        entryPrice: price,
        entryTime: Date.now(),
        highestProfitPct: 0,
        trailingStopPrice: null,
        prediction: prediction,
        features: currentFeatures
      });
      logger.info(`Live Trade Opened: SHORT at $${price.toFixed(2)}`);
      
      if (isRealTrading) {
        placeRealOrder('sell', settings.quantity); // Open SHORT with SELL
      }
    }
  };

  useEffect(() => {
    processLiveUpdateRef.current = processLiveUpdate;
  }, [processLiveUpdate]);

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

  const handleDeleteModel = async (name: string) => {
    setConfirmDelete({ name, type: 'local' });
  };

  const executeDeleteModel = async (name: string) => {
    await deleteModelPair(name);
    setSavedModels(getSavedModelPairs());
    setConfirmDelete(null);
  };

  const handleDeleteCompletely = async (name: string) => {
    setConfirmDelete({ name, type: 'complete' });
  };

  const executeDeleteCompletely = async (name: string) => {
    const pair = savedModels.find(p => p.name === name);
    if (!pair) return;

    try {
      setLoading(true);
      if (pair.onGitHub) {
        setStatus(`Deleting "${name}" from GitHub...`);
        await deleteModelPairFromGitHub(name, githubConfig);
      }
      
      setStatus(`Deleting "${name}" locally...`);
      await deleteModelPair(name);
      
      setSavedModels(getSavedModelPairs());
      logger.success(`Model "${name}" deleted completely.`);
    } catch (err) {
      console.error(err);
      logger.error(`Failed to delete model completely: ${err}`);
    } finally {
      setLoading(false);
      setConfirmDelete(null);
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

  const getPredictionsForRange = async (candles1h: Candle[], candles4h: Candle[], m1h: GRUModel, m4h: GRUModel, mcPasses: number = 1) => {
    logger.info(`Generating predictions for provided range (MC Passes: ${mcPasses})...`);
    const prices4h = candles4h.map(c => c.close);
    const highs4h = candles4h.map(c => c.high);
    const lows4h = candles4h.map(c => c.low);
    const volumes4h = candles4h.map(c => c.volume);
    const hours4h = candles4h.map(c => new Date(c.time).getHours());
    const days4h = candles4h.map(c => new Date(c.time).getDay());

    const rsi4h = calculateRSI(prices4h, indicatorPeriods.rsi);
    const ema4h = calculateEMA(prices4h, indicatorPeriods.ema);
    const ema9_4h = calculateEMA(prices4h, indicatorPeriods.ema9);
    const cross4h = calculateEMACross(ema9_4h, ema4h);
    const bb4h = calculateBollingerBands(prices4h, indicatorPeriods.bb);
    const macd4h = calculateMACD(prices4h);
    const stochRsi4h = calculateStochasticRSI(rsi4h);
    const atr4h = calculateATR(highs4h, lows4h, prices4h);
    const obv4h = calculateOBV(prices4h, volumes4h);
    const mfi4h = calculateMFI(highs4h, lows4h, prices4h, volumes4h, indicatorPeriods.mfi);
    const vol4h = calculateVolatility(prices4h, indicatorPeriods.volatility);
    const opens4h = candles4h.map(c => c.open);
    const harami4h = calculateBearishHarami(opens4h, prices4h);
    const marubozu4h = calculateMarubozu(opens4h, highs4h, lows4h, prices4h);
    const engulfing4h = calculateEngulfing(opens4h, prices4h);

    const secondaryPredictionsFor1h: number[] = [];
    const prices1h = candles1h.map(c => c.close);
    
    for (let i = 0; i < candles1h.length; i++) {
      const currentTime = candles1h[i].time;
      let last4hIdx = -1;
      for (let j = candles4h.length - 1; j >= windowSize; j--) {
        // Ensure the 4h candle has finished before the current 1h candle time
        if (candles4h[j].time + (4 * 60 * 60 * 1000) <= currentTime) {
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

        const nObv = normalize(windowIndices.map(idx => obv4h[idx]));
        const nMfi = windowIndices.map(idx => mfi4h[idx]).map(v => v / 100);
        const nVol = normalize(windowIndices.map(idx => vol4h[idx]));
        const nHour = windowIndices.map(idx => hours4h[idx]);
        const nDay = windowIndices.map(idx => days4h[idx]);
        const nHarami = windowIndices.map(idx => harami4h[idx]);
        const nMarubozu = windowIndices.map(idx => marubozu4h[idx]);
        const nEngulfing = windowIndices.map(idx => engulfing4h[idx]);

        for (let j = 0; j < windowSize; j++) {
          const h = nHour[j];
          input.push(
            nP[j], nR[j], nE[j], nU[j], nL[j], nH[j], nS[j], nA[j], nE9[j], nBelow[j], nCross[j], nObv[j], nMfi[j], nVol[j],
            h / 24, h >= 0 && h <= 9 ? 1 : 0, h >= 8 && h <= 17 ? 1 : 0, h >= 13 && h <= 22 ? 1 : 0, nDay[j] / 7,
            nHarami[j], nMarubozu[j], nEngulfing[j]
          );
        }
        secondaryPredictionsFor1h.push(m4h.predict(input));
      } else {
        secondaryPredictionsFor1h.push(0.5);
      }
    }

    const highs1h = candles1h.map(c => c.high);
    const lows1h = candles1h.map(c => c.low);
    const volumes1h = candles1h.map(c => c.volume);
    const hours1h = candles1h.map(c => new Date(c.time).getHours());
    const days1h = candles1h.map(c => new Date(c.time).getDay());

    const rsi1h = calculateRSI(prices1h, indicatorPeriods.rsi);
    const ema1h = calculateEMA(prices1h, indicatorPeriods.ema);
    const ema9_1h = calculateEMA(prices1h, indicatorPeriods.ema9);
    const cross1h = calculateEMACross(ema9_1h, ema1h);
    const bb1h = calculateBollingerBands(prices1h, indicatorPeriods.bb);
    const macd1h = calculateMACD(prices1h);
    const stochRsi1h = calculateStochasticRSI(rsi1h);
    const atr1h = calculateATR(highs1h, lows1h, prices1h);
    const obv1h = calculateOBV(prices1h, volumes1h);
    const mfi1h = calculateMFI(highs1h, lows1h, prices1h, volumes1h, indicatorPeriods.mfi);
    const vol1h = calculateVolatility(prices1h, indicatorPeriods.volatility);
    const opens1h = candles1h.map(c => c.open);
    const harami1h = calculateBearishHarami(opens1h, prices1h);
    const marubozu1h = calculateMarubozu(opens1h, highs1h, lows1h, prices1h);
    const engulfing1h = calculateEngulfing(opens1h, prices1h);

    const generatedPredictions: number[] = [];
    const generatedUncertainties: number[] = [];
    const generatedFeatures: number[][] = [];
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
      const normObv = normalize(windowIndices.map(idx => obv1h[idx]));
      const normMfi = windowIndices.map(idx => mfi1h[idx]).map(v => v / 100);
      const normVol = normalize(windowIndices.map(idx => vol1h[idx]));
      const normHour = windowIndices.map(idx => hours1h[idx]);
      const normDay = windowIndices.map(idx => days1h[idx]);
      const normHarami = windowIndices.map(idx => harami1h[idx]);
      const normMarubozu = windowIndices.map(idx => marubozu1h[idx]);
      const normEngulfing = windowIndices.map(idx => engulfing1h[idx]);

      const input: number[] = [];
      for (let j = 0; j < windowSize; j++) {
        const h = normHour[j];
        input.push(
          normPrice[j], normRsi[j], normEma[j], normBbUpper[j], normBbLower[j],
          normMacd[j], normStochRsi[j], normAtr[j], secWindow[j],
          normEma9[j], normBelow[j], normCross[j], normObv[j], normMfi[j], normVol[j],
          h / 24, h >= 0 && h <= 9 ? 1 : 0, h >= 8 && h <= 17 ? 1 : 0, h >= 13 && h <= 22 ? 1 : 0, normDay[j] / 7,
          normHarami[j], normMarubozu[j], normEngulfing[j]
        );
      }
      const result = m1h.predictMultiple(input, mcPasses);
      generatedPredictions.push(result.mean);
      generatedUncertainties.push(result.std);
      generatedFeatures.push(input);
    }
    return { predictions: generatedPredictions, uncertainties: generatedUncertainties, features: generatedFeatures };
  };

  const startTraining = async () => {
    if (candles.length < windowSize + 50) return;
    
    setTraining(true);
    logger.info('--- Starting Training Process ---');
    setPredictions([]);
    setBacktestResult(null);
    setTrainingLogs([]);
    setSecondaryTrainingLogs([]);
    setTrainingStats1h(null);
    setTrainingStats4h(null);
    
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
      const volumes4h = candles4h.map(c => c.volume);
      const hours4h = candles4h.map(c => new Date(c.time).getHours());
      const days4h = candles4h.map(c => new Date(c.time).getDay());

      const rsi4h = calculateRSI(prices4h, indicatorPeriods.rsi);
      const ema4h = calculateEMA(prices4h, indicatorPeriods.ema);
      const ema9_4h = calculateEMA(prices4h, indicatorPeriods.ema9);
      const cross4h = calculateEMACross(ema9_4h, ema4h);
      const bb4h = calculateBollingerBands(prices4h, indicatorPeriods.bb);
      const macd4h = calculateMACD(prices4h);
      const stochRsi4h = calculateStochasticRSI(rsi4h);
      const atr4h = calculateATR(highs4h, lows4h, prices4h);
      const obv4h = calculateOBV(prices4h, volumes4h);
      const mfi4h = calculateMFI(highs4h, lows4h, prices4h, volumes4h, indicatorPeriods.mfi);
      const vol4h = calculateVolatility(prices4h, indicatorPeriods.volatility);
      const opens4h = candles4h.map(c => c.open);
      const harami4h = calculateBearishHarami(opens4h, prices4h);
      const marubozu4h = calculateMarubozu(opens4h, highs4h, lows4h, prices4h);
      const engulfing4h = calculateEngulfing(opens4h, prices4h);
      
      const data4h = prepareData(
        prices4h, rsi4h, ema4h, bb4h.upper, bb4h.lower, windowSize, 
        undefined, macd4h.histogram, stochRsi4h, atr4h, dropThreshold4h,
        ema9_4h, cross4h.isBelow, cross4h.isCross,
        obv4h, mfi4h, vol4h, hours4h, days4h,
        harami4h, marubozu4h, engulfing4h
      );
      setTrainingStats4h(data4h.stats);
      const model4h = new GRUModel(windowSize, 22, 'temp_4h'); // Updated feature count (19 + 3)
      localStorage.setItem('temp_4h_metadata', JSON.stringify({ windowSize, featureCount: 22 }));
      await model4h.buildModel(modelHyperparams.units, modelHyperparams.dropout, modelHyperparams.learningRate);
      
      await model4h.train(data4h.xs, data4h.ys, epochs, modelHyperparams.units, modelHyperparams.dropout, modelHyperparams.learningRate, (epoch, logs) => {
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
          const nObv = normalize(windowIndices.map(idx => obv4h[idx]));
          const nMfi = windowIndices.map(idx => mfi4h[idx]).map(v => v / 100);
          const nVol = normalize(windowIndices.map(idx => vol4h[idx]));
          const nHour = windowIndices.map(idx => hours4h[idx]);
          const nDay = windowIndices.map(idx => days4h[idx]);
          const nHarami = windowIndices.map(idx => harami4h[idx]);
          const nMarubozu = windowIndices.map(idx => marubozu4h[idx]);
          const nEngulfing = windowIndices.map(idx => engulfing4h[idx]);

          for (let j = 0; j < windowSize; j++) {
            const h = nHour[j];
            input.push(
              nP[j], nR[j], nE[j], nU[j], nL[j], nH[j], nS[j], nA[j], nE9[j], nBelow[j], nCross[j], nObv[j], nMfi[j], nVol[j],
              h / 24, h >= 0 && h <= 9 ? 1 : 0, h >= 8 && h <= 17 ? 1 : 0, h >= 13 && h <= 22 ? 1 : 0, nDay[j] / 7,
              nHarami[j], nMarubozu[j], nEngulfing[j]
            );
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
      const volumes1h = candles.map(c => c.volume);
      const hours1h = candles.map(c => new Date(c.time).getHours());
      const days1h = candles.map(c => new Date(c.time).getDay());

      const rsi1h = calculateRSI(prices1h, indicatorPeriods.rsi);
      const ema1h = calculateEMA(prices1h, indicatorPeriods.ema);
      const ema9_1h = calculateEMA(prices1h, indicatorPeriods.ema9);
      const cross1h = calculateEMACross(ema9_1h, ema1h);
      const bb1h = calculateBollingerBands(prices1h, indicatorPeriods.bb);
      const macd1h = calculateMACD(prices1h);
      const stochRsi1h = calculateStochasticRSI(rsi1h);
      const atr1h = calculateATR(highs1h, lows1h, prices1h);
      const obv1h = calculateOBV(prices1h, volumes1h);
      const mfi1h = calculateMFI(highs1h, lows1h, prices1h, volumes1h, indicatorPeriods.mfi);
      const vol1h = calculateVolatility(prices1h, indicatorPeriods.volatility);
      const opens1h = candles.map(c => c.open);
      const harami1h = calculateBearishHarami(opens1h, prices1h);
      const marubozu1h = calculateMarubozu(opens1h, highs1h, lows1h, prices1h);
      const engulfing1h = calculateEngulfing(opens1h, prices1h);

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
        cross1h.isCross,
        obv1h,
        mfi1h,
        vol1h,
        hours1h,
        days1h,
        harami1h,
        marubozu1h,
        engulfing1h
      );
      setTrainingStats1h(data1h.stats);
      
      const featureCount1h = 23; // Updated feature count (20 + 3)
      const model1h = new GRUModel(windowSize, featureCount1h, 'temp_1h');
      localStorage.setItem('temp_1h_metadata', JSON.stringify({ windowSize, featureCount: featureCount1h }));
      setStatus(`Building Deep GRU Architecture (${featureCount1h} Features)...`);
      await model1h.buildModel(modelHyperparams.units, modelHyperparams.dropout, modelHyperparams.learningRate);
      
      setStatus(`Training Primary Model (${epochs} Epochs)...`);
      logger.info(`Training primary model for ${epochs} epochs...`);
      await model1h.train(data1h.xs, data1h.ys, epochs, modelHyperparams.units, modelHyperparams.dropout, modelHyperparams.learningRate, (epoch, logs) => {
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
      const { predictions: generatedPredictions, features: generatedFeatures } = await getPredictionsForRange(candles, candles4h, model1h, model4h, settings.mcPasses);
      
      setPredictions(generatedPredictions);
      const above = generatedPredictions.filter(p => p > settings.threshold).length;
      setPredictionStats({ total: generatedPredictions.length, aboveThreshold: above });
      setStatus('Model trained! Ready for backtest.');
      logger.success(`Predictions generated. ${above} signals above threshold.`);
      
      // Auto-run initial backtest
      logger.info('Running initial backtest...');
      const result = runBacktest(candles.slice(windowSize), generatedPredictions, settings, 10000, 0, generatedFeatures, FEATURE_NAMES);
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
      const { 
        predictions: btPredictions, 
        uncertainties: btUncertainties,
        features: btFeatures 
      } = await getPredictionsForRange(btCandles1h, btCandles4h, model1hRef.current, model4hRef.current, settings.mcPasses);
      logger.info(`Generated ${btPredictions.length} predictions.`);

      // 2. Fetch 5m data for high-res backtest
      setStatus('Fetching 5m data for accurate backtest...');
      logger.info(`Fetching 5m candles for backtest from ${format(new Date(startTs), 'yyyy-MM-dd HH:mm')} to ${format(new Date(endTs), 'yyyy-MM-dd HH:mm')}...`);
      const highResCandles = await fetchBTCData(0, '5m', startTs, endTs);
      
      if (highResCandles.length === 0) {
        throw new Error('Failed to fetch 5m data for the selected range');
      }

      setStatus('Aligning predictions with 5m candles...');
      
      const predictionMap = new Map<number, { val: number, uncertainty: number, features: number[] }>();
      btPredictions.forEach((val, i) => {
        // btPredictions[i] corresponds to btCandles1h[i + windowSize]
        const candleIndex = i + windowSize;
        if (btCandles1h[candleIndex]) {
          const time = btCandles1h[candleIndex].time;
          predictionMap.set(time, { 
            val, 
            uncertainty: btUncertainties[i],
            features: btFeatures[i] 
          });
        }
      });

      const alignedPredictions: number[] = [];
      const alignedUncertainties: number[] = [];
      const alignedFeatures: number[][] = [];
      const validHighResCandles: Candle[] = [];

      highResCandles.forEach(c => {
        // Round down to the nearest hour to find the corresponding prediction
        const hourTimestamp = Math.floor(c.time / (1000 * 60 * 60)) * (1000 * 60 * 60);
        const predictionData = predictionMap.get(hourTimestamp);
        
        if (predictionData !== undefined) {
          // Carry the prediction for the entire hour
          alignedPredictions.push(predictionData.val);
          alignedUncertainties.push(predictionData.uncertainty);
          alignedFeatures.push(predictionData.features);
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
      const result = runBacktest(validHighResCandles, alignedPredictions, settings, 10000, windowSize, alignedFeatures, FEATURE_NAMES, alignedUncertainties);
      
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

  const backtestChartData = useMemo(() => {
    if (!backtestResult) return [];
    return backtestResult.candles.map((c, i) => ({
      time: format(new Date(c.time), 'MM/dd HH:mm'),
      price: c.close,
      prediction: parseFloat((backtestResult.predictions[i] * 100).toFixed(2)),
    }));
  }, [backtestResult]);

  return (
    <div className="min-h-screen bg-black text-white font-sans selection:bg-emerald-500/30">
      {/* Confirmation Modal */}
      {confirmDelete && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
          <div className="bg-[#0D0D0E] border border-white/10 rounded-2xl p-6 max-w-md w-full shadow-2xl">
            <div className="flex items-center gap-3 mb-4 text-red-400">
              <ShieldAlert className="w-6 h-6" />
              <h3 className="text-xl font-bold">Confirm Deletion</h3>
            </div>
            <p className="text-white/60 mb-6">
              {confirmDelete.type === 'github' && `Are you sure you want to delete model "${confirmDelete.name}" from GitHub?`}
              {confirmDelete.type === 'local' && `Are you sure you want to delete model "${confirmDelete.name}" locally?`}
              {confirmDelete.type === 'complete' && `Are you sure you want to delete model "${confirmDelete.name}" COMPLETELY from both Local and GitHub?`}
            </p>
            <div className="flex gap-3">
              <button 
                onClick={() => setConfirmDelete(null)}
                className="flex-1 py-3 bg-white/5 hover:bg-white/10 text-white font-bold rounded-xl transition-all"
              >
                Cancel
              </button>
              <button 
                onClick={() => {
                  if (confirmDelete.type === 'github') executeDeleteFromGitHub(confirmDelete.name);
                  else if (confirmDelete.type === 'local') executeDeleteModel(confirmDelete.name);
                  else if (confirmDelete.type === 'complete') executeDeleteCompletely(confirmDelete.name);
                }}
                className="flex-1 py-3 bg-red-600 hover:bg-red-500 text-white font-bold rounded-xl transition-all"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
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
            {serverStatus?.serverIp && (
              <div className="flex items-center gap-2 px-3 py-1.5 bg-white/5 rounded-full border border-white/10">
                <ShieldAlert className="w-3 h-3 text-blue-400" />
                <span className="text-[10px] font-bold text-white/40 uppercase tracking-widest">Server IP:</span>
                <span className="text-xs font-mono text-blue-400/70">{serverStatus.serverIp}</span>
              </div>
            )}

            <button
              onClick={syncModelToServer}
              disabled={!model1hRef.current || isSyncing}
              className={cn(
                "flex items-center gap-2 px-4 py-2 rounded-full border transition-all font-bold text-[10px] uppercase tracking-widest",
                serverTradingStatus?.hasModels
                  ? "bg-blue-500/20 border-blue-500/50 text-blue-400"
                  : "bg-white/5 border-white/10 text-white/40 hover:border-white/20"
              )}
            >
              <CloudUpload className={cn("w-3 h-3", isSyncing && "animate-bounce")} />
              {isSyncing ? 'Syncing...' : (serverTradingStatus?.hasModels ? 'Models Synced' : 'Sync to Server')}
            </button>

            <button
              onClick={toggleServerTrading}
              disabled={!model1hRef.current || isTogglingTrading}
              className={cn(
                "flex items-center gap-2 px-4 py-2 rounded-full border transition-all font-bold text-[10px] uppercase tracking-widest",
                isLiveMode 
                  ? "bg-emerald-500/20 border-emerald-500/50 text-emerald-400 shadow-[0_0_15px_rgba(16,185,129,0.2)]" 
                  : "bg-white/5 border-white/10 text-white/40 hover:border-white/20",
                isTogglingTrading && "opacity-50 cursor-not-allowed"
              )}
            >
              <Zap className={cn("w-3 h-3", isLiveMode && "fill-emerald-400", isTogglingTrading && "animate-pulse")} />
              {isTogglingTrading ? 'Processing...' : (isLiveMode ? 'Live Mode ON' : 'Live Mode OFF')}
            </button>

            {isLiveMode && (
              <button
                onClick={async () => {
                  const nextMode = !isRealTrading;
                  setIsRealTrading(nextMode);
                  if (isLiveMode) {
                    try {
                      await fetch('/api/trading/mode', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ isRealTrading: nextMode })
                      });
                      logger.warning(`Trading mode switched to: ${nextMode ? 'REAL' : 'PAPER'}`);
                    } catch (err) {
                      logger.error('Failed to switch trading mode on server.');
                    }
                  }
                }}
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
                    <h2 className="font-medium">BTC/USD Price (1h)</h2>
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
                      <label className="text-[10px] text-white/40 uppercase font-bold">Drop Target (1H) %</label>
                      <input type="number" step="0.1" min="0.1" max="5" value={dropThreshold} onChange={(e) => setDropThreshold(parseFloat(e.target.value) || 0.5)} className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-xs focus:border-purple-500/50 outline-none" />
                    </div>
                    <div className="space-y-1.5">
                      <label className="text-[10px] text-white/40 uppercase font-bold">Drop Target (4H) %</label>
                      <input type="number" step="0.1" min="0.1" max="5" value={dropThreshold4h} onChange={(e) => setDropThreshold4h(parseFloat(e.target.value) || 0.5)} className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-xs focus:border-purple-500/50 outline-none" />
                    </div>
                  </div>
                  <div className="grid grid-cols-3 gap-3">
                    <div className="space-y-1.5">
                      <label className="text-[10px] text-white/40 uppercase font-bold">Epochs</label>
                      <input type="number" min="5" max="100" value={epochs} onChange={(e) => setEpochs(parseInt(e.target.value) || 15)} className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-xs focus:border-purple-500/50 outline-none" />
                    </div>
                    <div className="space-y-1.5">
                      <label className="text-[10px] text-white/40 uppercase font-bold">GRU Units</label>
                      <input type="number" step="8" min="8" max="256" value={modelHyperparams.units} onChange={(e) => setModelHyperparams({...modelHyperparams, units: parseInt(e.target.value) || 64})} className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-xs focus:border-purple-500/50 outline-none" />
                    </div>
                    <div className="space-y-1.5">
                      <label className="text-[10px] text-white/40 uppercase font-bold">Dropout</label>
                      <input type="number" step="0.05" min="0" max="0.5" value={modelHyperparams.dropout} onChange={(e) => setModelHyperparams({...modelHyperparams, dropout: parseFloat(e.target.value) || 0.2})} className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-xs focus:border-purple-500/50 outline-none" />
                    </div>
                  </div>
                  <div className="grid grid-cols-1 gap-3">
                    <div className="space-y-1.5">
                      <label className="text-[10px] text-white/40 uppercase font-bold">Learning Rate</label>
                      <input type="number" step="0.0001" min="0.0001" max="0.01" value={modelHyperparams.learningRate} onChange={(e) => setModelHyperparams({...modelHyperparams, learningRate: parseFloat(e.target.value) || 0.001})} className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-xs focus:border-purple-500/50 outline-none" />
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
                  <div className="grid grid-cols-3 gap-3">
                    <div className="space-y-1.5">
                      <label className="text-[10px] text-white/40 uppercase font-bold">BB</label>
                      <input type="number" value={indicatorPeriods.bb} onChange={(e) => setIndicatorPeriods({...indicatorPeriods, bb: parseInt(e.target.value) || 20})} className="w-full bg-white/5 border border-white/10 rounded-lg px-2 py-2 text-xs focus:border-purple-500/50 outline-none" />
                    </div>
                    <div className="space-y-1.5">
                      <label className="text-[10px] text-white/40 uppercase font-bold">MFI</label>
                      <input type="number" value={indicatorPeriods.mfi} onChange={(e) => setIndicatorPeriods({...indicatorPeriods, mfi: parseInt(e.target.value) || 14})} className="w-full bg-white/5 border border-white/10 rounded-lg px-2 py-2 text-xs focus:border-purple-500/50 outline-none" />
                    </div>
                    <div className="space-y-1.5">
                      <label className="text-[10px] text-white/40 uppercase font-bold">Volatility</label>
                      <input type="number" value={indicatorPeriods.volatility} onChange={(e) => setIndicatorPeriods({...indicatorPeriods, volatility: parseInt(e.target.value) || 20})} className="w-full bg-white/5 border border-white/10 rounded-lg px-2 py-2 text-xs focus:border-purple-500/50 outline-none" />
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
                      <div className="flex items-center justify-between">
                        <h4 className="text-[10px] text-white/40 uppercase font-bold tracking-wider">Primary Logs (1h)</h4>
                        {trainingStats1h && (
                          <span className="text-[9px] text-purple-400 font-mono">
                            Drops: {trainingStats1h.positive} / {trainingStats1h.total} ({(trainingStats1h.positive / trainingStats1h.total * 100).toFixed(1)}%)
                          </span>
                        )}
                      </div>
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
                      <div className="flex items-center justify-between">
                        <h4 className="text-[10px] text-white/40 uppercase font-bold tracking-wider">Secondary Logs (4h)</h4>
                        {trainingStats4h && (
                          <span className="text-[9px] text-purple-400 font-mono">
                            Drops: {trainingStats4h.positive} / {trainingStats4h.total} ({(trainingStats4h.positive / trainingStats4h.total * 100).toFixed(1)}%)
                          </span>
                        )}
                      </div>
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
                      <select value={settings.quantityType} onChange={(e) => setSettings({...settings, quantityType: e.target.value as 'USD' | 'BTC' | 'LOTS'})} className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-xs outline-none focus:border-amber-500/50 appearance-none">
                        <option value="USD">USD</option>
                        <option value="BTC">BTC</option>
                        <option value="LOTS">Lots</option>
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
                    <div className="space-y-1.5">
                      <label className="text-[10px] text-white/40 uppercase font-bold">Min Velocity</label>
                      <input type="number" step="0.01" value={settings.minSignalVelocity} onChange={(e) => setSettings({...settings, minSignalVelocity: parseFloat(e.target.value) || 0})} className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-xs outline-none focus:border-amber-500/50" />
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-4 pt-2 border-t border-white/5">
                    <div className="space-y-1.5">
                      <label className="text-[10px] text-white/40 uppercase font-bold">MC Passes</label>
                      <input type="number" min="1" max="50" value={settings.mcPasses} onChange={(e) => setSettings({...settings, mcPasses: parseInt(e.target.value) || 1})} className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-xs outline-none focus:border-amber-500/50" />
                    </div>
                    <div className="space-y-1.5">
                      <label className="text-[10px] text-white/40 uppercase font-bold">Max Uncertainty</label>
                      <input type="number" step="0.01" value={settings.maxUncertainty} onChange={(e) => setSettings({...settings, maxUncertainty: parseFloat(e.target.value) || 0})} className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-xs outline-none focus:border-amber-500/50" />
                    </div>
                  </div>
                  <div className="flex flex-col gap-2 pt-2 border-t border-white/5">
                      <div className="flex items-center gap-2">
                        <input 
                          type="checkbox" 
                          id="useSessionTradingBT"
                          checked={settings.useSessionTrading} 
                          onChange={(e) => setSettings({...settings, useSessionTrading: e.target.checked})}
                          className="w-4 h-4 rounded border-white/10 bg-white/5 text-amber-500 focus:ring-amber-500/50"
                        />
                        <label htmlFor="useSessionTradingBT" className="text-[10px] text-white/60 uppercase font-bold cursor-pointer">Session Trading</label>
                      </div>
                      <div className="flex items-center gap-2">
                        <input 
                          type="checkbox" 
                          id="useOnlyCompletedCandlesBT"
                          checked={settings.useOnlyCompletedCandles} 
                          onChange={(e) => setSettings({...settings, useOnlyCompletedCandles: e.target.checked})}
                          className="w-4 h-4 rounded border-white/10 bg-white/5 text-amber-500 focus:ring-amber-500/50"
                        />
                        <label htmlFor="useOnlyCompletedCandlesBT" className="text-[10px] text-white/60 uppercase font-bold cursor-pointer">Completed Candles Only</label>
                      </div>
                    </div>
                  </div>
                  {settings.useSessionTrading && (
                    <div className="grid grid-cols-2 gap-4 pt-2 border-t border-white/5 animate-in fade-in slide-in-from-top-1">
                      <div className="space-y-1.5">
                        <label className="text-[10px] text-white/40 uppercase font-bold">Asia (Start-End)</label>
                        <div className="flex gap-2">
                          <input type="number" value={settings.asiaStart} onChange={(e) => setSettings({...settings, asiaStart: parseInt(e.target.value)})} className="w-full bg-white/5 border border-white/10 rounded-lg px-2 py-1.5 text-xs outline-none focus:border-amber-500/50" />
                          <input type="number" value={settings.asiaEnd} onChange={(e) => setSettings({...settings, asiaEnd: parseInt(e.target.value)})} className="w-full bg-white/5 border border-white/10 rounded-lg px-2 py-1.5 text-xs outline-none focus:border-amber-500/50" />
                        </div>
                      </div>
                      <div className="space-y-1.5">
                        <label className="text-[10px] text-white/40 uppercase font-bold">NY (Start-End)</label>
                        <div className="flex gap-2">
                          <input type="number" value={settings.nyStart} onChange={(e) => setSettings({...settings, nyStart: parseInt(e.target.value)})} className="w-full bg-white/5 border border-white/10 rounded-lg px-2 py-1.5 text-xs outline-none focus:border-amber-500/50" />
                          <input type="number" value={settings.nyEnd} onChange={(e) => setSettings({...settings, nyEnd: parseInt(e.target.value)})} className="w-full bg-white/5 border border-white/10 rounded-lg px-2 py-1.5 text-xs outline-none focus:border-amber-500/50" />
                        </div>
                      </div>
                    </div>
                  )}
                  <button onClick={startBacktest} disabled={loading || training || !model1hRef.current} className={cn("w-full py-3 rounded-xl font-semibold flex items-center justify-center gap-2 transition-all active:scale-95", !model1hRef.current ? "bg-white/5 text-white/50 cursor-not-allowed" : "bg-amber-500 hover:bg-amber-400 text-black shadow-[0_0_20px_rgba(245,158,11,0.2)]")}>
                    <Activity className="w-4 h-4" /> {!model1hRef.current ? 'Train Model First' : 'Run Backtest'}
                  </button>
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

              {/* Price & Prediction Chart */}
              {backtestResult && (
                <div className="bg-[#0D0D0E] border border-white/5 rounded-2xl p-6 shadow-xl">
                  <h3 className="font-medium mb-6 flex items-center gap-2 text-blue-400">
                    <Activity className="w-5 h-5" />
                    Price & Prediction
                  </h3>
                  <div className="h-[300px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={backtestChartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#ffffff05" vertical={false} />
                        <XAxis dataKey="time" stroke="#ffffff30" fontSize={10} tickLine={false} axisLine={false} />
                        <YAxis yAxisId="left" domain={['auto', 'auto']} stroke="#ffffff30" fontSize={10} tickLine={false} axisLine={false} />
                        <YAxis yAxisId="right" orientation="right" domain={[0, 100]} stroke="#ffffff30" fontSize={10} tickLine={false} axisLine={false} />
                        <Tooltip 
                          contentStyle={{ backgroundColor: '#0D0D0E', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                          itemStyle={{ fontSize: '12px' }}
                        />
                        <Line yAxisId="left" type="monotone" dataKey="price" stroke="#3b82f6" dot={false} strokeWidth={2} />
                        <Line yAxisId="right" type="monotone" dataKey="prediction" stroke="#f59e0b" dot={false} strokeWidth={1.5} strokeDasharray="5 5" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="flex justify-center gap-6 mt-4">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-0.5 bg-[#3b82f6]"></div>
                      <span className="text-[10px] text-white/40 uppercase">Price (USD)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-0.5 bg-[#f59e0b] border-t border-dashed"></div>
                      <span className="text-[10px] text-white/40 uppercase">Prediction (%)</span>
                    </div>
                  </div>
                </div>
              )}

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
                            {trade.prediction !== undefined && (
                              <p className="text-amber-400 font-bold mt-1">Pred: {(trade.prediction * 100).toFixed(1)}%</p>
                            )}
                          </div>
                          <div className="space-y-1 text-right">
                            <p className="text-white/40 uppercase font-bold tracking-wider">Exit</p>
                            <p className="text-white/80">{format(new Date(trade.exitTime), 'MM/dd HH:mm')}</p>
                            <p className="text-white font-mono font-medium">${trade.exitPrice.toLocaleString()}</p>
                          </div>
                        </div>

                        {trade.features && (
                          <div className="pt-2 border-t border-white/5">
                            <details className="group">
                              <summary className="text-[9px] text-white/30 cursor-pointer hover:text-white/50 flex items-center gap-1 list-none">
                                <ChevronDown className="w-2 h-2 group-open:rotate-180 transition-transform" />
                                View Features at Entry
                              </summary>
                              <div className="mt-2 grid grid-cols-3 gap-x-2 gap-y-1 text-[8px] text-white/40 font-mono bg-black/40 p-2 rounded-lg">
                                {Object.entries(trade.features).map(([name, val]) => (
                                  <div key={name} className="truncate">{name}: {(val as number).toFixed(4)}</div>
                                ))}
                                <div className="col-span-full mt-1 pt-1 border-t border-white/5 text-[7px] italic opacity-50">
                                  Showing entry features
                                </div>
                              </div>
                            </details>
                          </div>
                        )}

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

          {/* GitHub Config */}
          <div className="bg-[#0D0D0E] border border-white/5 rounded-2xl p-6 shadow-xl mb-6">
            <div className="flex items-center gap-2 mb-4">
              <Github className="w-4 h-4 text-white/60" />
              <h3 className="text-xs font-bold text-white/40 uppercase tracking-wider">GitHub Storage Settings</h3>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 items-end">
              <div className="space-y-2">
                <label className="text-[10px] text-white/40 uppercase">Owner</label>
                <input 
                  type="text" 
                  value={githubConfig.owner}
                  onChange={(e) => setGithubConfig({ ...githubConfig, owner: e.target.value })}
                  placeholder="github-username"
                  className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm focus:border-blue-500/50 outline-none transition-all"
                />
              </div>
              <div className="space-y-2">
                <label className="text-[10px] text-white/40 uppercase">Repo</label>
                <input 
                  type="text" 
                  value={githubConfig.repo}
                  onChange={(e) => setGithubConfig({ ...githubConfig, repo: e.target.value })}
                  placeholder="model-storage"
                  className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm focus:border-blue-500/50 outline-none transition-all"
                />
              </div>
              <div className="space-y-2">
                <label className="text-[10px] text-white/40 uppercase">Path</label>
                <input 
                  type="text" 
                  value={githubConfig.path}
                  onChange={(e) => setGithubConfig({ ...githubConfig, path: e.target.value })}
                  placeholder="models"
                  className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm focus:border-blue-500/50 outline-none transition-all"
                />
              </div>
              <div className="space-y-2">
                <label className="text-[10px] text-white/40 uppercase">Token</label>
                <input 
                  type="password" 
                  value={githubConfig.token}
                  onChange={(e) => setGithubConfig({ ...githubConfig, token: e.target.value })}
                  placeholder="ghp_..."
                  className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm focus:border-blue-500/50 outline-none transition-all"
                />
              </div>
              <div className="flex gap-2">
                <button 
                  onClick={handleSaveGithubConfig}
                  className="flex-1 py-2 bg-white/5 hover:bg-white/10 text-white/60 hover:text-white border border-white/10 rounded-lg text-xs font-bold transition-all"
                >
                  Save & Sync
                </button>
                <button 
                  onClick={async () => {
                    if (githubConfig.owner && githubConfig.repo && githubConfig.token) {
                      try {
                        setLoading(true);
                        setStatus('Syncing models from GitHub...');
                        const updated = await syncModelsFromGitHub(githubConfig);
                        setSavedModels(updated);
                      } catch (err) {
                        console.error(err);
                      } finally {
                        setLoading(false);
                      }
                    } else {
                      logger.error('Please configure GitHub settings first.');
                    }
                  }}
                  className="p-2 bg-white/5 hover:bg-white/10 text-white/60 hover:text-white border border-white/10 rounded-lg transition-all"
                  title="Sync Models"
                >
                  <RefreshCw className={cn("w-4 h-4", loading && status.includes('Syncing') && "animate-spin")} />
                </button>
              </div>
            </div>
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
                    <div className="flex gap-1">
                      <button 
                        onClick={() => handleUploadToGitHub(pair.name)}
                        title={pair.onGitHub ? "Update on GitHub" : "Upload to GitHub"}
                        className={cn(
                          "p-2 rounded-lg transition-all",
                          pair.onGitHub 
                            ? "hover:bg-emerald-500/10 text-emerald-400/40 hover:text-emerald-400" 
                            : "hover:bg-blue-500/10 text-blue-400/40 hover:text-blue-400"
                        )}
                      >
                        <CloudUpload className="w-4 h-4" />
                      </button>
                      
                      {pair.onGitHub && (
                        <button 
                          onClick={() => handleDeleteFromGitHub(pair.name)}
                          title="Delete from GitHub"
                          className="p-2 hover:bg-red-500/10 text-red-400/40 hover:text-red-400 rounded-lg transition-all"
                        >
                          <CloudOff className="w-4 h-4" />
                        </button>
                      )}

                      <button 
                        onClick={() => handleDeleteModel(pair.name)}
                        title="Delete Locally"
                        className="p-2 hover:bg-red-500/10 text-white/20 hover:text-red-400 rounded-lg transition-all"
                      >
                        <ShieldAlert className="w-4 h-4" />
                      </button>

                      <button 
                        onClick={() => handleDeleteCompletely(pair.name)}
                        title="Delete Completely (Local + GitHub)"
                        className="p-2 hover:bg-red-600/20 text-red-500/40 hover:text-red-500 rounded-lg transition-all"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                  <div className="flex items-center gap-2 mb-1">
                    <h4 className="font-bold text-white/80 truncate">{pair.name}</h4>
                    {pair.onGitHub && <Github className="w-3 h-3 text-emerald-500" />}
                  </div>
                  <p className="text-[10px] text-white/30 mb-6">Saved: {format(new Date(pair.timestamp), 'MMM dd, HH:mm')}</p>
                  
                  <div className="flex gap-2">
                    <button 
                      onClick={() => handleLoadModel(pair.name)}
                      disabled={loading || training}
                      className="flex-1 py-2 bg-blue-600/10 hover:bg-blue-600 text-blue-400 hover:text-white border border-blue-600/20 rounded-lg text-xs font-bold transition-all"
                    >
                      Load Local
                    </button>
                    {pair.onGitHub && (
                      <button 
                        onClick={() => handleLoadFromGitHub(pair.name)}
                        disabled={loading || training}
                        className="flex-1 py-2 bg-emerald-600/10 hover:bg-emerald-600 text-emerald-400 hover:text-white border border-emerald-600/20 rounded-lg text-xs font-bold transition-all"
                      >
                        Load GitHub
                      </button>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>
        </section>

        {/* Section 5: Live Trading Configuration */}
        <section className="space-y-6">
          <div className="flex items-center gap-2 border-b border-white/5 pb-2">
            <Settings className="w-5 h-5 text-purple-500" />
            <h2 className="text-xl font-bold tracking-tight">5. Live Trading Configuration</h2>
          </div>

          <div className="bg-[#0D0D0E] border border-white/5 rounded-2xl p-6 shadow-xl">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
              <div className="space-y-4">
                <h3 className="text-xs font-bold text-white/40 uppercase tracking-wider">Thresholds</h3>
                <div className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex justify-between text-xs">
                      <span className="text-white/60">Entry Threshold</span>
                      <span className="text-blue-400 font-mono">{(settings.threshold * 100).toFixed(1)}%</span>
                    </div>
                    <input 
                      type="range" min="0" max="1" step="0.01" 
                      value={settings.threshold} 
                      onChange={(e) => setSettings({...settings, threshold: parseFloat(e.target.value)})}
                      className="w-full h-1.5 bg-white/5 rounded-lg appearance-none cursor-pointer accent-blue-500"
                    />
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-xs">
                      <span className="text-white/60">Exit Threshold</span>
                      <span className="text-purple-400 font-mono">{(settings.exitThreshold * 100).toFixed(1)}%</span>
                    </div>
                    <input 
                      type="range" min="0" max="0.5" step="0.01" 
                      value={settings.exitThreshold} 
                      onChange={(e) => setSettings({...settings, exitThreshold: parseFloat(e.target.value)})}
                      className="w-full h-1.5 bg-white/5 rounded-lg appearance-none cursor-pointer accent-purple-500"
                    />
                  </div>
                </div>
              </div>

              <div className="space-y-4">
                <h3 className="text-xs font-bold text-white/40 uppercase tracking-wider">Risk Management</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1">
                    <label className="text-[10px] text-white/30 uppercase">Stop Loss (%)</label>
                    <input 
                      type="number" step="0.1"
                      value={isNaN(settings.stopLoss) ? '' : settings.stopLoss * 100}
                      onChange={(e) => setSettings({...settings, stopLoss: (parseFloat(e.target.value) || 0) / 100})}
                      className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white/80 focus:border-blue-500/50 outline-none transition-all"
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="text-[10px] text-white/30 uppercase">Take Profit (%)</label>
                    <input 
                      type="number" step="0.1"
                      value={isNaN(settings.takeProfit) ? '' : settings.takeProfit * 100}
                      onChange={(e) => setSettings({...settings, takeProfit: (parseFloat(e.target.value) || 0) / 100})}
                      className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white/80 focus:border-blue-500/50 outline-none transition-all"
                    />
                  </div>
                </div>
                <div className="space-y-1">
                  <label className="text-[10px] text-white/30 uppercase">Max Duration (Hours)</label>
                  <input 
                    type="number"
                    value={isNaN(settings.maxDurationHours) ? '' : settings.maxDurationHours}
                    onChange={(e) => setSettings({...settings, maxDurationHours: parseInt(e.target.value) || 0})}
                    className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white/80 focus:border-blue-500/50 outline-none transition-all"
                  />
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1">
                    <label className="text-[10px] text-white/30 uppercase">Daily Profit Limit ($)</label>
                    <input 
                      type="number" step="1"
                      value={settings.dailyProfitLimit || 0}
                      onChange={(e) => setSettings({...settings, dailyProfitLimit: parseFloat(e.target.value) || 0})}
                      className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white/80 focus:border-blue-500/50 outline-none transition-all"
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="text-[10px] text-white/30 uppercase">Daily Loss Limit ($)</label>
                    <input 
                      type="number" step="1"
                      value={settings.dailyLossLimit || 0}
                      onChange={(e) => setSettings({...settings, dailyLossLimit: parseFloat(e.target.value) || 0})}
                      className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white/80 focus:border-blue-500/50 outline-none transition-all"
                    />
                  </div>
                </div>
              </div>

              <div className="space-y-4">
                <h3 className="text-xs font-bold text-white/40 uppercase tracking-wider">Trailing Stop</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1">
                    <label className="text-[10px] text-white/30 uppercase">Activation (%)</label>
                    <input 
                      type="number" step="0.1"
                      value={isNaN(settings.trailingStopActivation) ? '' : settings.trailingStopActivation * 100}
                      onChange={(e) => setSettings({...settings, trailingStopActivation: (parseFloat(e.target.value) || 0) / 100})}
                      className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white/80 focus:border-blue-500/50 outline-none transition-all"
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="text-[10px] text-white/30 uppercase">Offset (%)</label>
                    <input 
                      type="number" step="0.1"
                      value={isNaN(settings.trailingStopOffset) ? '' : settings.trailingStopOffset * 100}
                      onChange={(e) => setSettings({...settings, trailingStopOffset: (parseFloat(e.target.value) || 0) / 100})}
                      className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white/80 focus:border-blue-500/50 outline-none transition-all"
                    />
                  </div>
                </div>
              </div>

              <div className="space-y-4">
                <h3 className="text-xs font-bold text-white/40 uppercase tracking-wider">Accuracy & Momentum</h3>
                <div className="space-y-4">
                  <div className="space-y-1">
                    <label className="text-[10px] text-white/30 uppercase">MC Dropout Passes</label>
                    <input 
                      type="number" min="1" max="50"
                      value={settings.mcPasses || 1}
                      onChange={(e) => setSettings({...settings, mcPasses: parseInt(e.target.value) || 1})}
                      className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white/80 focus:border-blue-500/50 outline-none transition-all"
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="text-[10px] text-white/30 uppercase">Max Uncertainty (Std Dev)</label>
                    <input 
                      type="number" step="0.01"
                      value={settings.maxUncertainty || 0}
                      onChange={(e) => setSettings({...settings, maxUncertainty: parseFloat(e.target.value) || 0})}
                      className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white/80 focus:border-blue-500/50 outline-none transition-all"
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="text-[10px] text-white/30 uppercase">Min Signal Velocity (Delta)</label>
                    <input 
                      type="number" step="0.01"
                      value={settings.minSignalVelocity || 0}
                      onChange={(e) => setSettings({...settings, minSignalVelocity: parseFloat(e.target.value) || 0})}
                      className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white/80 focus:border-blue-500/50 outline-none transition-all"
                    />
                  </div>
                </div>
              </div>

              <div className="space-y-4">
                <h3 className="text-xs font-bold text-white/40 uppercase tracking-wider">Strategy & Options</h3>
                <div className="space-y-4">
                  <div className="space-y-1">
                    <label className="text-[10px] text-white/30 uppercase">Strategy Type</label>
                    <select 
                      value={settings.strategyType || 'SHORT_BTC'}
                      onChange={(e) => setSettings({...settings, strategyType: e.target.value as any})}
                      className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white/80 focus:border-blue-500/50 outline-none transition-all"
                    >
                      <option value="SHORT_BTC">Short BTC Futures</option>
                      <option value="CALL_SPREAD">Sell Call Spread (Tomorrow)</option>
                      <option value="SHORT_CALL">Short Single Call (Tomorrow)</option>
                    </select>
                  </div>
                  {(settings.strategyType === 'CALL_SPREAD' || settings.strategyType === 'SHORT_CALL') && (
                    <div className="grid grid-cols-2 gap-4 animate-in fade-in slide-in-from-top-1 duration-200">
                      <div className="space-y-1">
                        <label className="text-[10px] text-white/30 uppercase">Short Call Delta</label>
                        <input 
                          type="number" step="0.01" min="0" max="1"
                          value={settings.shortCallDelta || 0.3}
                          onChange={(e) => setSettings({...settings, shortCallDelta: parseFloat(e.target.value) || 0})}
                          className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white/80 focus:border-blue-500/50 outline-none transition-all"
                        />
                      </div>
                      {settings.strategyType === 'CALL_SPREAD' && (
                        <div className="space-y-1">
                          <label className="text-[10px] text-white/30 uppercase">Long Call Delta</label>
                          <input 
                            type="number" step="0.01" min="0" max="1"
                            value={settings.longCallDelta || 0.1}
                            onChange={(e) => setSettings({...settings, longCallDelta: parseFloat(e.target.value) || 0})}
                            className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white/80 focus:border-blue-500/50 outline-none transition-all"
                          />
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>

              <div className="space-y-4">
                <h3 className="text-xs font-bold text-white/40 uppercase tracking-wider">Session & Candle Settings</h3>
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-3 bg-white/5 rounded-xl border border-white/5">
                    <div className="space-y-0.5">
                      <div className="text-xs font-bold text-white/80">Session Trading</div>
                      <div className="text-[10px] text-white/30">Only enter during Asia/NY</div>
                    </div>
                    <button 
                      onClick={() => setSettings({...settings, useSessionTrading: !settings.useSessionTrading})}
                      className={cn(
                        "w-10 h-5 rounded-full transition-all relative",
                        settings.useSessionTrading ? "bg-blue-600" : "bg-white/10"
                      )}
                    >
                      <div className={cn(
                        "absolute top-1 w-3 h-3 rounded-full bg-white transition-all",
                        settings.useSessionTrading ? "right-1" : "left-1"
                      )} />
                    </button>
                  </div>

                  {settings.useSessionTrading && (
                    <div className="grid grid-cols-2 gap-4 animate-in fade-in slide-in-from-top-1 duration-200">
                      <div className="space-y-1">
                        <label className="text-[10px] text-white/30 uppercase">Asia (Start-End)</label>
                        <div className="flex gap-2">
                          <input 
                            type="number" min="0" max="23"
                            value={isNaN(settings.asiaStart) ? '' : settings.asiaStart}
                            onChange={(e) => setSettings({...settings, asiaStart: parseInt(e.target.value) || 0})}
                            className="w-full bg-white/5 border border-white/10 rounded-lg px-2 py-1.5 text-xs text-white/80 focus:border-blue-500/50 outline-none"
                          />
                          <input 
                            type="number" min="0" max="23"
                            value={isNaN(settings.asiaEnd) ? '' : settings.asiaEnd}
                            onChange={(e) => setSettings({...settings, asiaEnd: parseInt(e.target.value) || 0})}
                            className="w-full bg-white/5 border border-white/10 rounded-lg px-2 py-1.5 text-xs text-white/80 focus:border-blue-500/50 outline-none"
                          />
                        </div>
                      </div>
                      <div className="space-y-1">
                        <label className="text-[10px] text-white/30 uppercase">NY (Start-End)</label>
                        <div className="flex gap-2">
                          <input 
                            type="number" min="0" max="23"
                            value={isNaN(settings.nyStart) ? '' : settings.nyStart}
                            onChange={(e) => setSettings({...settings, nyStart: parseInt(e.target.value) || 0})}
                            className="w-full bg-white/5 border border-white/10 rounded-lg px-2 py-1.5 text-xs text-white/80 focus:border-blue-500/50 outline-none"
                          />
                          <input 
                            type="number" min="0" max="23"
                            value={isNaN(settings.nyEnd) ? '' : settings.nyEnd}
                            onChange={(e) => setSettings({...settings, nyEnd: parseInt(e.target.value) || 0})}
                            className="w-full bg-white/5 border border-white/10 rounded-lg px-2 py-1.5 text-xs text-white/80 focus:border-blue-500/50 outline-none"
                          />
                        </div>
                      </div>
                    </div>
                  )}

                  <div className="flex items-center justify-between p-3 bg-white/5 rounded-xl border border-white/5">
                    <div className="space-y-0.5">
                      <div className="text-xs font-bold text-white/80">Completed Candles Only</div>
                      <div className="text-[10px] text-white/30">Entry only at hour start</div>
                    </div>
                    <button 
                      onClick={() => setSettings({...settings, useOnlyCompletedCandles: !settings.useOnlyCompletedCandles})}
                      className={cn(
                        "w-10 h-5 rounded-full transition-all relative",
                        settings.useOnlyCompletedCandles ? "bg-purple-600" : "bg-white/10"
                      )}
                    >
                      <div className={cn(
                        "absolute top-1 w-3 h-3 rounded-full bg-white transition-all",
                        settings.useOnlyCompletedCandles ? "right-1" : "left-1"
                      )} />
                    </button>
                  </div>
                </div>
              </div>

              <div className="space-y-4">
                <h3 className="text-xs font-bold text-white/40 uppercase tracking-wider">Position Size</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1">
                    <label className="text-[10px] text-white/30 uppercase">Unit</label>
                    <select 
                      value={settings.quantityType}
                      onChange={(e) => setSettings({...settings, quantityType: e.target.value as 'USD' | 'BTC' | 'LOTS'})}
                      className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white/80 focus:border-blue-500/50 outline-none transition-all"
                    >
                      <option value="USD">USD</option>
                      <option value="BTC">BTC</option>
                      <option value="LOTS">Lots (0.001 BTC)</option>
                    </select>
                  </div>
                  <div className="space-y-1">
                    <label className="text-[10px] text-white/30 uppercase">Quantity</label>
                    <input 
                      type="number" step="0.01"
                      value={isNaN(settings.quantity) ? '' : settings.quantity}
                      onChange={(e) => setSettings({...settings, quantity: parseFloat(e.target.value) || 0})}
                      className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white/80 focus:border-blue-500/50 outline-none transition-all"
                    />
                  </div>
                </div>
              </div>

              <div className="mt-8 pt-6 border-t border-white/5 flex justify-end">
                <button
                  onClick={pushSettingsToServer}
                  className="flex items-center gap-2 px-6 py-2.5 bg-blue-600 hover:bg-blue-500 text-white rounded-xl font-semibold transition-all active:scale-95 shadow-[0_0_20px_rgba(37,99,235,0.2)]"
                >
                  <Save className="w-4 h-4" />
                  Push Settings to Server
                </button>
              </div>
            </div>
          </div>
        </section>

        {/* Section 6: Live Paper Trading Monitor */}
        {isLiveMode && (
          <section className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-700">
            <div className="flex items-center gap-2 border-b border-emerald-500/20 pb-2">
              <Activity className="w-5 h-5 text-emerald-500" />
              <h2 className="text-xl font-bold tracking-tight">6. Live {isRealTrading ? 'Real' : 'Paper'} Trading Monitor</h2>
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
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-xs font-bold text-white/40 uppercase">{isRealTrading ? 'Real Wallet Balance (USD)' : 'Live Paper Balance'}</h3>
                    {isRealTrading && (
                      <button 
                        onClick={fetchRealBalance}
                        className="p-1 hover:bg-white/5 rounded-md transition-colors"
                        title="Refresh Balance"
                      >
                        <RefreshCw className="w-3 h-3 text-white/40 hover:text-white" />
                      </button>
                    )}
                  </div>
                  <div className={cn(
                    "text-3xl font-bold",
                    isRealTrading ? "text-red-400" : "text-emerald-400"
                  )}>
                    ${(isRealTrading ? (realBalance || 0) : livePaperBalance).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                  </div>
                  <div className="mt-2 flex justify-between items-center">
                    <div className="text-[10px] text-white/30">{isRealTrading ? 'Connected to Delta Exchange' : 'Starting: $10,000.00'}</div>
                    <div className={cn(
                      "text-[10px] font-bold",
                      (serverTradingStatus?.dailyProfit || 0) >= 0 ? "text-emerald-500" : "text-red-500"
                    )}>
                      Daily: ${(serverTradingStatus?.dailyProfit || 0).toFixed(2)}
                    </div>
                  </div>
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
                        <div className="flex justify-between items-center">
                          <span className="text-[10px] text-white/30 uppercase">OBV</span>
                          <span className="text-[10px] font-mono text-white/60">{liveParams.obv.toLocaleString()}</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-[10px] text-white/30 uppercase">MFI</span>
                          <span className="text-[10px] font-mono text-white/60">{liveParams.mfi.toFixed(1)}</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-[10px] text-white/30 uppercase">Volat.</span>
                          <span className="text-[10px] font-mono text-white/60">{liveParams.volatility.toFixed(1)}%</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-[10px] text-white/30 uppercase">Harami</span>
                          <span className={cn("text-[10px] font-mono", liveParams.harami === 1 ? "text-red-400" : "text-white/40")}>
                            {liveParams.harami === 1 ? 'YES' : 'NO'}
                          </span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-[10px] text-white/30 uppercase">Marubozu</span>
                          <span className={cn("text-[10px] font-mono", liveParams.marubozu === 1 ? "text-red-400" : "text-white/40")}>
                            {liveParams.marubozu === 1 ? 'YES' : 'NO'}
                          </span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-[10px] text-white/30 uppercase">Engulfing</span>
                          <span className={cn("text-[10px] font-mono", liveParams.engulfing === 1 ? "text-red-400" : "text-white/40")}>
                            {liveParams.engulfing === 1 ? 'YES' : 'NO'}
                          </span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-[10px] text-white/30 uppercase">Session</span>
                          <span className="text-[10px] font-mono text-amber-400">{liveParams.session}</span>
                        </div>
                        <div className="flex justify-between items-center pt-2 border-t border-white/5 col-span-2">
                          <span className="text-[10px] text-white/30 uppercase">Uncertainty (StdDev)</span>
                          <span className={cn("text-[10px] font-mono", (liveParams.uncertainty || 0) > settings.maxUncertainty ? "text-red-400" : "text-emerald-400")}>
                            {liveParams.uncertainty ? liveParams.uncertainty.toFixed(4) : '0.0000'}
                          </span>
                        </div>
                        <div className="flex justify-between items-center col-span-2">
                          <span className="text-[10px] text-white/30 uppercase">Velocity (Delta)</span>
                          <span className={cn("text-[10px] font-mono", (liveParams.velocity || 0) < settings.minSignalVelocity ? "text-amber-400" : "text-emerald-400")}>
                            {liveParams.velocity ? (liveParams.velocity > 0 ? '+' : '') + liveParams.velocity.toFixed(4) : '0.0000'}
                          </span>
                        </div>
                        <div className="col-span-2 flex justify-between items-center pt-2 border-t border-white/5 opacity-40">
                          <span className="text-[9px] uppercase tracking-tighter">Last Updated</span>
                          <span className="text-[9px] font-mono">{lastLiveUpdate ? format(lastLiveUpdate, 'HH:mm:ss') : '--:--:--'}</span>
                        </div>
                      </div>
                    )}

                    {activeLiveTrade && (activeLiveTrade.type === 'CALL_SPREAD' || activeLiveTrade.type === 'SHORT_CALL') && activeLiveTrade.legs && (
                      <div className="pt-4 border-t border-white/5 space-y-2">
                        <div className="text-[10px] text-white/40 font-bold uppercase tracking-wider">
                          {activeLiveTrade.type === 'CALL_SPREAD' ? 'Spread Legs' : 'Option Leg'}
                        </div>
                        {activeLiveTrade.legs.map((leg: any, idx: number) => (
                          <div key={idx} className="space-y-1 pb-1 border-b border-white/5 last:border-0">
                            <div className="flex justify-between items-center text-[10px]">
                              <span className={cn(leg.side === 'sell' ? "text-red-400" : "text-emerald-400")}>
                                {leg.side.toUpperCase()} {leg.symbol}
                              </span>
                              <span className="text-white/60 font-mono">${leg.entryPrice.toFixed(2)}</span>
                            </div>
                            {leg.pnl !== undefined && (
                              <div className="flex justify-between items-center text-[9px]">
                                <span className="text-white/30 uppercase">PNL</span>
                                <span className={cn("font-mono font-bold", leg.pnl >= 0 ? "text-emerald-400" : "text-red-400")}>
                                  {leg.pnl >= 0 ? '+' : ''}{leg.pnl.toFixed(4)} USDT
                                </span>
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    )}

                    <div className="flex items-center justify-between pt-2 border-t border-white/5">
                      <span className="text-xs text-white/60">Status</span>
                      <span className={cn(
                        "text-[10px] font-bold px-2 py-0.5 rounded uppercase",
                        activeLiveTrade ? "bg-amber-500/10 text-amber-500" : "bg-white/5 text-white/40"
                      )}>
                        {activeLiveTrade ? `In Trade (${activeLiveTrade.type})` : 'Waiting for Signal'}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Active Server Settings */}
                {serverTradingStatus?.settings && (
                  <div className="bg-[#0D0D0E] border border-white/5 rounded-2xl p-6 shadow-xl">
                    <div className="flex items-center gap-2 mb-4">
                      <Settings className="w-3 h-3 text-white/40" />
                      <h3 className="text-xs font-bold text-white/40 uppercase tracking-wider">Active Server Settings</h3>
                    </div>
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-[10px] text-white/30 uppercase">Threshold</span>
                        <span className="text-[10px] font-mono text-white/60">{serverTradingStatus.settings.threshold}</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-[10px] text-white/30 uppercase">Exit Thr.</span>
                        <span className="text-[10px] font-mono text-white/60">{serverTradingStatus.settings.exitThreshold}</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-[10px] text-white/30 uppercase">Stop Loss</span>
                        <span className="text-[10px] font-mono text-red-400">{(serverTradingStatus.settings.stopLoss * 100).toFixed(1)}%</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-[10px] text-white/30 uppercase">Take Profit</span>
                        <span className="text-[10px] font-mono text-emerald-400">{(serverTradingStatus.settings.takeProfit * 100).toFixed(1)}%</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-[10px] text-white/30 uppercase">TS Activation</span>
                        <span className="text-[10px] font-mono text-white/60">{(serverTradingStatus.settings.trailingStopActivation * 100).toFixed(1)}%</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-[10px] text-white/30 uppercase">TS Offset</span>
                        <span className="text-[10px] font-mono text-white/60">{(serverTradingStatus.settings.trailingStopOffset * 100).toFixed(1)}%</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-[10px] text-white/30 uppercase">Quantity</span>
                        <span className="text-[10px] font-mono text-white/60">{serverTradingStatus.settings.quantity} {serverTradingStatus.settings.quantityType}</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-[10px] text-white/30 uppercase">Session Trading</span>
                        <span className={cn("text-[10px] font-mono", serverTradingStatus.settings.useSessionTrading ? "text-emerald-400" : "text-white/40")}>
                          {serverTradingStatus.settings.useSessionTrading ? 'ON' : 'OFF'}
                        </span>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              <div className="lg:col-span-3 space-y-6">
                {/* Ongoing Trade Card */}
                {activeLiveTrade && livePrice && (
                  <div className="bg-[#0D0D0E] border border-amber-500/30 rounded-2xl p-6 shadow-xl animate-in zoom-in-95 duration-300">
                    <div className="flex items-center justify-between mb-6">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 bg-amber-500/10 rounded-xl flex items-center justify-center">
                          <Activity className="w-5 h-5 text-amber-500" />
                        </div>
                        <div>
                          <h3 className="font-bold text-white/90">Active SHORT Position</h3>
                          <p className="text-[10px] text-white/30">Opened {format(activeLiveTrade.entryTime, 'HH:mm:ss')}</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className={cn(
                          "text-2xl font-bold",
                          (activeLiveTrade.pnlPct !== undefined ? activeLiveTrade.pnlPct >= 0 : ((activeLiveTrade.entryPrice - livePrice) / activeLiveTrade.entryPrice) >= 0) ? "text-emerald-400" : "text-red-400"
                        )}>
                          {activeLiveTrade.pnlPct !== undefined ? activeLiveTrade.pnlPct.toFixed(2) : ((activeLiveTrade.entryPrice - livePrice) / activeLiveTrade.entryPrice * 100).toFixed(2)}%
                        </div>
                        <div className="text-[10px] text-white/30">Live P&L</div>
                      </div>
                    </div>

                    {activeLiveTrade.features && (
                      <div className="mb-6 p-4 bg-white/5 rounded-xl border border-white/5">
                        <div className="flex items-center justify-between mb-3">
                          <span className="text-[10px] text-white/30 uppercase font-bold tracking-wider">Entry Features</span>
                          <span className="text-[10px] text-blue-400 font-mono">Pred: {activeLiveTrade.prediction.toFixed(3)}</span>
                        </div>
                        <div className="grid grid-cols-4 gap-4">
                          {Object.entries(activeLiveTrade.features).slice(0, 12).map(([name, val]) => (
                            <div key={name} className="space-y-1">
                              <div className="text-[8px] text-white/20 uppercase truncate">{name}</div>
                              <div className="text-[10px] font-mono text-white/60">{(val as number).toFixed(2)}</div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                      <div className="space-y-1">
                        <span className="text-[10px] text-white/30 uppercase">Entry Price</span>
                        <div className="text-sm font-mono text-white/80">${activeLiveTrade.entryPrice.toLocaleString()}</div>
                      </div>
                      <div className="space-y-1">
                        <span className="text-[10px] text-white/30 uppercase">Current Price</span>
                        <div className="text-sm font-mono text-white/80">${livePrice.toLocaleString()}</div>
                      </div>
                      <div className="space-y-1">
                        <span className="text-[10px] text-white/30 uppercase">Stop Loss</span>
                        <div className="text-sm font-mono text-red-400/70">${(activeLiveTrade.entryPrice * (1 + settings.stopLoss)).toLocaleString()}</div>
                      </div>
                      <div className="space-y-1">
                        <span className="text-[10px] text-white/30 uppercase">Take Profit</span>
                        <div className="text-sm font-mono text-emerald-400/70">${(activeLiveTrade.entryPrice * (1 - settings.takeProfit)).toLocaleString()}</div>
                      </div>
                    </div>

                    <div className="mt-6 pt-6 border-t border-white/5 flex items-center justify-between">
                      <div className="flex items-center gap-4">
                        <div className="space-y-1">
                          <span className="text-[9px] text-white/30 uppercase">Unrealized P&L</span>
                          <div className={cn(
                            "text-sm font-bold",
                            (activeLiveTrade.pnlPct !== undefined ? activeLiveTrade.pnlPct >= 0 : ((activeLiveTrade.entryPrice - livePrice) / activeLiveTrade.entryPrice) >= 0) ? "text-emerald-400" : "text-red-400"
                          )}>
                            {activeLiveTrade.pnl !== undefined 
                              ? `$${activeLiveTrade.pnl.toFixed(2)} (${activeLiveTrade.pnlPct?.toFixed(2)}%)`
                              : `$${(settings.quantityType === 'USD' 
                                  ? (settings.quantity * (activeLiveTrade.entryPrice - livePrice) / activeLiveTrade.entryPrice)
                                  : (settings.quantityType === 'LOTS'
                                    ? (settings.quantity * 0.001 * (activeLiveTrade.entryPrice - livePrice))
                                    : (settings.quantity * (activeLiveTrade.entryPrice - livePrice)))
                                ).toFixed(2)} (${((activeLiveTrade.entryPrice - livePrice) / activeLiveTrade.entryPrice * 100).toFixed(2)}%)`
                            }
                          </div>
                        </div>
                      </div>
                      <button 
                        onClick={() => {
                          const profitPct = (activeLiveTrade.entryPrice - livePrice) / activeLiveTrade.entryPrice;
                          const btcQuantity = settings.quantityType === 'LOTS' ? settings.quantity * 0.001 : settings.quantity;
                          const profit = settings.quantityType === 'USD'
                            ? settings.quantity * profitPct
                            : btcQuantity * (activeLiveTrade.entryPrice - livePrice);
                            
                          const newTrade: Trade = {
                            type: 'SHORT',
                            entryPrice: activeLiveTrade.entryPrice,
                            exitPrice: livePrice,
                            entryTime: activeLiveTrade.entryTime,
                            exitTime: Date.now(),
                            profit,
                            profitPct: profitPct * 100,
                            exitReason: 'MANUAL',
                            prediction: activeLiveTrade.prediction,
                            features: activeLiveTrade.features
                          };
                          setLiveTrades(prev => [newTrade, ...prev]);
                          setLivePaperBalance(prev => prev + profit);
                          setActiveLiveTrade(null);
                          
                          // Call server to close trade and start cooldown
                          fetch('/api/trading/close', { method: 'POST' })
                            .then(() => logger.success('Manual close request sent to server'))
                            .catch(err => logger.error('Failed to send manual close request'));
                        }}
                        className="px-4 py-2 bg-red-500/10 hover:bg-red-500 text-red-500 hover:text-white border border-red-500/20 rounded-lg text-xs font-bold transition-all"
                      >
                        Close Position Manually
                      </button>
                    </div>
                  </div>
                )}

                <div className="bg-[#0D0D0E] border border-white/5 rounded-2xl p-6 shadow-xl h-full">
                  <h3 className="font-medium mb-4 flex items-center gap-2 text-emerald-400">
                    <History className="w-4 h-4" />
                    Live Trade History
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-h-[600px] overflow-y-auto pr-2 custom-scrollbar">
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
                                trade.exitReason === 'MANUAL' ? "text-blue-500 bg-blue-500/10" :
                                "text-white/20 bg-white/5"
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
                          
                          {trade.features && (
                            <div className="pt-2 border-t border-white/5 space-y-1.5">
                              <div className="flex items-center justify-between text-[9px] text-white/30 uppercase font-bold">
                                <span>Entry Features</span>
                                <span className="text-blue-400">Pred: {(trade.prediction || 0).toFixed(3)}</span>
                              </div>
                              <div className="grid grid-cols-3 gap-x-2 gap-y-1 text-[8px] font-mono">
                                {Object.entries(trade.features).slice(0, 9).map(([name, val]) => (
                                  <div key={name} className="flex justify-between border-b border-white/5 pb-0.5">
                                    <span className="text-white/20 truncate mr-1">{name}</span>
                                    <span className="text-white/60">{(val as number).toFixed(2)}</span>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}

                          <div className="pt-2 border-t border-white/5 flex justify-between items-center">
                            <span className="text-[9px] text-white/30 italic">{format(trade.exitTime, 'MMM dd, HH:mm')}</span>
                            <span className={cn("text-[10px] font-bold", trade.profitPct >= 0 ? "text-emerald-500/70" : "text-red-500/70")}>
                              {trade.profitPct >= 0 ? '+' : ''}{trade.profitPct.toFixed(2)}%
                            </span>
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
      <Terminal serverLogs={serverTradingStatus?.logs} />
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
