import WebSocket from 'ws';
import * as tf from '@tensorflow/tfjs';
import { GRUModel } from './modelService';
import * as indicators from './indicatorService';
import fetch from 'node-fetch';
import crypto from 'crypto';
import { deltaRequest, productMap, fetchProducts } from './deltaApi';
import fs from 'fs';
import path from 'path';

interface TradingSettings {
  shortThreshold: number;
  longThreshold: number;
  shortExitThreshold: number;
  longExitThreshold: number;
  biasThreshold?: number;
  stopLoss: number;
  takeProfit: number;
  trailingStopActivation: number;
  trailingStopOffset: number;
  maxDurationHours: number;
  quantity: number;
  quantityType: 'LOTS' | 'BTC' | 'USD';
  useSessionTrading: boolean;
  asiaStart: number;
  asiaEnd: number;
  nyStart: number;
  nyEnd: number;
  useOnlyCompletedCandles: boolean;
  mcPasses: number;
  maxUncertainty: number;
  minSignalVelocity: number;
  strategyType: 'SHORT_BTC' | 'LONG_BTC' | 'BOTH' | 'CALL_SPREAD' | 'SHORT_CALL' | 'PUT_SPREAD' | 'SHORT_PUT';
  shortCallDelta: number;
  longCallDelta: number;
  shortPutDelta: number;
  longPutDelta: number;
  dailyProfitLimit: number;
  dailyLossLimit: number;
  indicatorPeriods?: {
    rsi: number;
    ema: number;
    ema9: number;
    bb: number;
    mfi: number;
    volatility: number;
  };
  useDRLConfluence?: boolean;
  useDRLOnly?: boolean;
}

interface ActiveTrade {
  type: 'LONG' | 'SHORT' | 'CALL_SPREAD' | 'SHORT_CALL' | 'PUT_SPREAD' | 'SHORT_PUT';
  entryPrice: number;
  entryTime: number;
  highestProfitPct: number;
  trailingStopPrice: number | null;
  prediction: number[];
  pnl?: number;
  pnlPct?: number;
  size?: number;
  features?: Record<string, number>;
  orderId?: string;
  legs?: {
    symbol: string;
    side: 'buy' | 'sell';
    entryPrice: number;
    size: number;
    pnl?: number;
    markPrice?: number;
  }[];
}

interface ClosedTrade {
  type: 'LONG' | 'SHORT' | 'CALL_SPREAD' | 'SHORT_CALL' | 'PUT_SPREAD' | 'SHORT_PUT';
  entryPrice: number;
  exitPrice: number;
  entryTime: number;
  exitTime: number;
  profit: number;
  profitPct: number;
  exitReason: string;
  prediction: number[];
  features: Record<string, number>;
  orderId?: string;
}

interface LogEntry {
  timestamp: number;
  type: 'info' | 'error' | 'success' | 'warning';
  message: string;
}

export class TradingService {
  private isRunning: boolean = false;
  private isRealTrading: boolean = false;
  private isOpeningTrade: boolean = false;
  private settings: TradingSettings | null = null;
  private model1h: GRUModel | null = null;
  private model4h: GRUModel | null = null;
  private drlModel: tf.LayersModel | null = null;
  private ws: WebSocket | null = null;
  private candles: any[] = [];
  private candles4h: any[] = [];
  private activeTrade: ActiveTrade | null = null;
  private closedTrades: ClosedTrade[] = [];
  private lastPrediction: number[] | null = null;
  private lastPrediction4h: number[] | null = null;
  private previousPrediction: number[] | null = null;
  private lastPredictionTime: number = 0;
  private lastPredictionTime4h: number = 0;
  private lastUncertainty: number[] = [0, 0, 0];
  private lastVelocity: number[] = [0, 0, 0];
  private lastPrice: number | null = null;
  private lastFeatures: Record<string, number> | null = null;
  private lastParams: any = null;
  private lastUpdate: Date | null = null;
  private logs: LogEntry[] = [];
  private tickerCount: number = 0;
  private lastProcessTime: number = 0;
  private lastRefreshTime: number = 0;
  private lastTradeCloseTime: number = 0;
  private dailyProfit: number = 0;
  private lastDailyReset: number = 0;
  private processInterval: number = 5000; // Process at most once every 5 seconds

  private apiKey: string = process.env.DELTA_API_KEY || '';
  private apiSecret: string = process.env.DELTA_API_SECRET || '';
  private settingsPath: string = path.join(process.cwd(), 'server', 'trading-settings.json');
  private statePath: string = path.join(process.cwd(), 'server', 'trading-state.json');

  constructor() {
    this.log('Trading Service Initialized');
    this.loadSettings();
    this.loadState();
  }

  private loadSettings() {
    try {
      if (fs.existsSync(this.settingsPath)) {
        const data = fs.readFileSync(this.settingsPath, 'utf8');
        this.settings = JSON.parse(data);
        
        // Ensure defaults for new settings
        if (this.settings) {
          if (this.settings.mcPasses === undefined) this.settings.mcPasses = 1;
          if (this.settings.maxUncertainty === undefined) this.settings.maxUncertainty = 0.1;
          if (this.settings.minSignalVelocity === undefined) this.settings.minSignalVelocity = 0.1;
          if (this.settings.strategyType === undefined) this.settings.strategyType = 'SHORT_BTC';
          if (this.settings.shortCallDelta === undefined) this.settings.shortCallDelta = 0.3;
          if (this.settings.longCallDelta === undefined) this.settings.longCallDelta = 0.1;
          if (this.settings.shortPutDelta === undefined) this.settings.shortPutDelta = 0.3;
          if (this.settings.longPutDelta === undefined) this.settings.longPutDelta = 0.1;
          if (this.settings.dailyProfitLimit === undefined) this.settings.dailyProfitLimit = 0;
          if (this.settings.dailyLossLimit === undefined) this.settings.dailyLossLimit = 0;
        }
        
        this.log('Trading settings loaded from disk', 'success');
      }
    } catch (err) {
      this.log(`Error loading settings: ${err}`, 'error');
    }
  }

  private loadState() {
    try {
      if (fs.existsSync(this.statePath)) {
        const data = fs.readFileSync(this.statePath, 'utf8');
        const state = JSON.parse(data);
        this.activeTrade = state.activeTrade || null;
        if (this.activeTrade && !this.activeTrade.type) {
          this.activeTrade.type = 'SHORT';
        }
        this.closedTrades = state.closedTrades || [];
        this.isRunning = state.isRunning || false;
        this.isRealTrading = state.isRealTrading || false;
        this.dailyProfit = state.dailyProfit || 0;
        this.lastDailyReset = state.lastDailyReset || 0;
        this.log('Trading state loaded from disk', 'success');
        
        if (this.isRunning) {
          this.log('Resuming trading session...', 'info');
          this.fetchInitialData().then(() => this.connectWebSocket());
        }
      }
    } catch (err) {
      this.log(`Error loading state: ${err}`, 'error');
    }
  }

  private saveState() {
    try {
      const state = {
        activeTrade: this.activeTrade,
        closedTrades: this.closedTrades,
        isRunning: this.isRunning,
        isRealTrading: this.isRealTrading,
        dailyProfit: this.dailyProfit,
        lastDailyReset: this.lastDailyReset
      };
      fs.writeFileSync(this.statePath, JSON.stringify(state, null, 2));
    } catch (err) {
      console.error('Error saving state:', err);
    }
  }

  private saveSettings() {
    try {
      if (this.settings) {
        const dir = path.dirname(this.settingsPath);
        if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
        fs.writeFileSync(this.settingsPath, JSON.stringify(this.settings, null, 2));
        this.log('Trading settings saved to disk', 'info');
      }
    } catch (err) {
      this.log(`Error saving settings: ${err}`, 'error');
    }
  }

  private log(message: string, type: LogEntry['type'] = 'info') {
    const timestamp = Date.now();
    const entry: LogEntry = { timestamp, type, message };
    console.log(`[${new Date(timestamp).toISOString()}] [${type.toUpperCase()}] ${message}`);
    this.logs.unshift(entry);
    if (this.logs.length > 100) this.logs.pop();
  }

  public async start(settings: TradingSettings, isRealTrading: boolean = false) {
    if (this.isRunning) {
      this.settings = settings;
      this.isRealTrading = isRealTrading;
      this.saveSettings();
      this.log('Settings updated while running', 'info');
      return;
    }
    this.settings = settings;
    this.isRealTrading = isRealTrading;
    this.isRunning = true;
    this.saveSettings();
    this.saveState();
    this.log(`Starting Live Trading (Real: ${this.isRealTrading})`, 'success');
    
    await this.fetchInitialData();
    this.connectWebSocket();
  }

  public updateSettings(settings: TradingSettings) {
    this.settings = settings;
    this.saveSettings();
    this.saveState();
    this.log('Trading settings synchronized', 'info');
  }

  public async closeActiveTrade() {
    try {
      if (!this.activeTrade) {
        this.log('No active trade to close', 'warning');
        return;
      }

      this.log('Manual trade closure requested via API', 'info');
      
      if (this.isRealTrading) {
        if ((this.activeTrade.type === 'CALL_SPREAD' || this.activeTrade.type === 'SHORT_CALL' || this.activeTrade.type === 'PUT_SPREAD' || this.activeTrade.type === 'SHORT_PUT') && this.activeTrade.legs) {
          for (const leg of this.activeTrade.legs) {
            const exitSide = leg.side === 'sell' ? 'buy' : 'sell';
            await this.placeOptionOrder(leg.symbol, exitSide, leg.size);
          }
        } else {
          const side = this.activeTrade.type === 'SHORT' ? 'buy' : 'sell';
          await this.placeRealOrder(side, this.settings?.quantity || 1, true);
        }
      }

      this.activeTrade = null;
      this.lastTradeCloseTime = Date.now();
      this.saveState();
      this.log('Active trade cleared and cooldown started', 'success');
    } catch (err) {
      this.log(`Error closing active trade: ${err}`, 'error');
    }
  }

  public async setTradingMode(isRealTrading: boolean) {
    if( this.isRealTrading != isRealTrading) {
      await this.closeActiveTrade();
    }

    this.isRealTrading = isRealTrading;
    this.saveState();
    this.log(`Trading mode switched to: ${isRealTrading ? 'REAL' : 'PAPER'}`, 'warning');
  }

  public stop() {
    this.isRunning = false;
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.saveState();
    this.log('Stopped Live Trading', 'warning');
  }

  public setModels(model1h: GRUModel, model4h?: GRUModel, drlModel?: tf.LayersModel) {
    this.model1h = model1h;
    if (model4h) this.model4h = model4h;
    if (drlModel) this.drlModel = drlModel;
    this.log('MTF 1h/4h and DRL Models updated on server', 'success');
  }

  public getStatus() {
    return {
      isRunning: this.isRunning,
      isRealTrading: this.isRealTrading,
      activeTrade: this.activeTrade,
      closedTrades: this.closedTrades.slice(0, 50),
      lastPrediction: this.lastPrediction,
      lastPrediction4h: this.lastPrediction4h,
      lastPrice: this.lastPrice,
      lastFeatures: this.lastFeatures,
      lastParams: this.lastParams,
      lastUpdate: this.lastUpdate,
      logs: this.logs.slice(0, 20),
      hasModels: !!this.model1h,
      has4hModel: !!this.model4h,
      settings: this.settings,
      dailyProfit: this.dailyProfit
    };
  }

  private checkDailyReset() {
    const now = new Date();
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
    if (this.lastDailyReset < today) {
      this.dailyProfit = 0;
      this.lastDailyReset = today;
      this.log('Daily profit/loss reset for the new day', 'info');
      this.saveState();
    }
  }

  private async fetchInitialData() {
    try {
      this.log('Fetching initial candle data (MTF 1h & 4h)...');
      
      const fetchTimeframe = async (resolution: string, days: number) => {
        const endTs = Date.now();
        const startMs = endTs - (days * 24 * 60 * 60 * 1000);
        const secondsPerCandle = resolution === '1h' ? 3600 : (resolution === '4h' ? 14400 : 900);
        
        let candles: any[] = [];
        let currentEnd = Math.floor(endTs / 1000);
        let targetStart = Math.floor(startMs / 1000);
        let symbol = 'BTCUSD';
        
        const maxSamples = 1000;
        let fetchedCount = 0;

        while (fetchedCount < maxSamples && currentEnd > targetStart) {
          const batchSize = Math.min(maxSamples - fetchedCount, 500);
          let batchStart = currentEnd - (batchSize * secondsPerCandle);
          if (batchStart < targetStart) batchStart = targetStart;
          
          const url = `https://api.india.delta.exchange/v2/history/candles?symbol=${symbol}&resolution=${resolution}&start=${batchStart}&end=${currentEnd}`;
          const res = await fetch(url);
          const data = await res.json() as any;
          
          if (!data.result || data.result.length === 0) {
            if (symbol === 'BTCUSD') {
              symbol = 'BTCUSD_P';
              continue;
            }
            break;
          }
          
          candles = [...data.result, ...candles];
          fetchedCount += data.result.length;
          const oldestInBatch = Math.min(...data.result.map((c: any) => c.time));
          currentEnd = oldestInBatch - secondsPerCandle;
        }

        return candles
          .map(d => ({
            time: d.time,
            open: parseFloat(d.open),
            high: parseFloat(d.high),
            low: parseFloat(d.low),
            close: parseFloat(d.close),
            volume: parseFloat(d.volume)
          }))
          .sort((a, b) => a.time - b.time)
          .filter((c, i, self) => i === 0 || c.time !== self[i-1].time);
      };

      const [c1h, c4h] = await Promise.all([
        fetchTimeframe('1h', 30),
        fetchTimeframe('4h', 60)
      ]);

      this.candles = c1h;
      this.candles4h = c4h;
      this.lastRefreshTime = Date.now();
      this.log(`Loaded ${this.candles.length}h and ${this.candles4h.length} 4h historical candles.`);
    } catch (err) {
      this.log(`Error fetching initial data: ${err}`, 'error');
    }
  }

  private connectWebSocket() {
    if (this.ws) this.ws.close();
    
    this.log('Connecting to WebSocket: wss://socket.india.delta.exchange', 'info');
    this.ws = new WebSocket('wss://socket.india.delta.exchange');
    this.tickerCount = 0;
    
    this.ws.on('open', () => {
      this.log('WebSocket Connected to Delta Exchange', 'success');
      
      // 1. Authenticate if real trading is enabled
      if (this.isRealTrading && this.apiKey && this.apiSecret) {
        const timestamp = Math.floor(Date.now() / 1000).toString();
        const signature = crypto.createHmac('sha256', this.apiSecret)
          .update('GET' + timestamp + '/v2/websocket')
          .digest('hex');
          
        this.ws?.send(JSON.stringify({
          type: 'auth',
          payload: {
            api_key: this.apiKey,
            signature: signature,
            timestamp: timestamp
          }
        }));
      }

      // 2. Subscribe to channels
      const channels: any[] = [{ name: 'v2/ticker', symbols: ['BTCUSD', 'BTCUSD_P'] }];
      if (this.isRealTrading) {
        channels.push({ name: 'positions', symbols: ['BTCUSD', 'BTCUSD_P'] });
      }

      this.ws?.send(JSON.stringify({
        type: 'subscribe',
        payload: {
          channels: channels
        }
      }));
    });

    this.ws.on('message', (data: any) => {
      try {
        const msg = JSON.parse(data.toString());
        
        if (msg.type === 'v2/ticker' || msg.type === 'ticker') {
          if (this.tickerCount < 5) {
            this.log(`Ticker Received: ${msg.symbol || msg.symbol_id} | Price: ${msg.close || msg.mark_price || msg.last_price}`, 'info');
            this.tickerCount++;
          }
          if (msg.symbol === 'BTCUSD' || msg.symbol === 'BTCUSD_P' || msg.symbol_id === 'BTCUSD') {
            const price = parseFloat(msg.close || msg.mark_price || msg.last_price || msg.price);
            if (!isNaN(price)) {
              this.processPriceUpdate(price);
            }
          }
        } else if (msg.type === 'positions') {
          const pos = Array.isArray(msg.result) ? msg.result[0] : msg;
          if (pos && (pos.symbol === 'BTCUSD' || pos.symbol === 'BTCUSD_P')) {
            const entry = parseFloat(pos.entry_price || pos.avg_entry_price);
            const size = parseFloat(pos.size);
            
            if (this.activeTrade && size !== 0 && entry > 0) {
              this.activeTrade.entryPrice = entry;
              this.activeTrade.size = size;
              this.saveState();
            } else if (this.activeTrade && size === 0) {
              this.log('Position closed on exchange. Clearing active trade.', 'info');
              this.activeTrade = null;
              this.lastTradeCloseTime = Date.now();
              this.saveState();
            }
          }
        } else if (msg.type === 'subscriptions') {
          this.log('WebSocket Subscriptions Confirmed', 'success');
        } else if (msg.type === 'auth') {
          if (msg.success) {
            this.log('WebSocket Authentication Successful', 'success');
          } else {
            this.log(`WebSocket Authentication Failed: ${msg.error}`, 'error');
          }
        } else if (msg.type === 'error') {
          this.log(`WebSocket Server Error: ${msg.message}`, 'error');
        } else if (msg.type === 'heartbeat' || msg.type === 'ping') {
          if (msg.type === 'ping') this.ws?.send(JSON.stringify({ type: 'pong' }));
        } else {
          // Log unknown message types to debug
          if (msg.type !== 'ping' && msg.type !== 'pong') {
            this.log(`WS Message Type: ${msg.type} | Symbol: ${msg.symbol}`, 'info');
          }
          // If we are not getting prices, maybe the type is different?
          if (msg.mark_price || msg.last_price || msg.close) {
            const price = parseFloat(msg.close || msg.mark_price || msg.last_price);
            if (!isNaN(price)) this.processPriceUpdate(price);
          }
        }
      } catch (e) {
        // Ignore parse errors
      }
    });

    this.ws.on('error', (err) => {
      this.log(`WebSocket Error: ${err}`, 'error');
    });

    this.ws.on('close', () => {
      if (this.isRunning) {
        this.log('WebSocket Closed. Reconnecting in 5s...', 'warning');
        setTimeout(() => this.connectWebSocket(), 5000);
      }
    });
  }

  private encodePosition(pos: 'LONG' | 'SHORT' | 'NEUTRAL'): number[] {
    if (pos === 'LONG') return [1, 0, 0];
    if (pos === 'SHORT') return [0, 1, 0];
    return [0, 0, 1];
  }

  private generateMTFPrediction(candles: any[], model: GRUModel | null) {
    if (!model || candles.length < 300) return null;
    
    const bufferSize = 300;
    const windowSize = 20;
    const context = candles.slice(candles.length - bufferSize);
    
    const p = context.map(c => c.close);
    const h = context.map(c => c.high);
    const l = context.map(c => c.low);
    const v = context.map(c => c.volume);
    const o = context.map(c => c.open);
    
    const periods = this.settings?.indicatorPeriods || { rsi: 14, ema: 20, ema9: 9, bb: 20, mfi: 14, volatility: 20 };
    const rsi = indicators.calculateRSI(p, periods.rsi);
    const ema = indicators.calculateEMA(p, periods.ema);
    const ema9 = indicators.calculateEMA(p, periods.ema9);
    const bb = indicators.calculateBollingerBands(p, periods.bb);
    const macd = indicators.calculateMACD(p);
    const stochRsi = indicators.calculateStochasticRSI(rsi);
    const atr = indicators.calculateATR(h, l, p);
    const cross = indicators.calculateEMACross(ema9, ema);
    const obv = indicators.calculateOBV(p, v);
    const mfi = indicators.calculateMFI(h, l, p, v, periods.mfi);
    const vol = indicators.calculateVolatility(p, periods.volatility);
    const roc = indicators.calculateROC(p, 12);
    const harami = indicators.calculateBearishHarami(o, p);
    const marubozu = indicators.calculateMarubozu(o, h, l, p);
    const engulfing = indicators.calculateEngulfing(o, p);

    const x: number[] = [];
    const winIdx = Array.from({ length: windowSize }, (_, k) => bufferSize - windowSize + k);
    
    const pWin = winIdx.map(idx => p[idx]);
    const pMin = Math.min(...pWin);
    const pMax = Math.max(...pWin);
    const pRange = pMax - pMin;
    const normVal = (v: number) => (pRange === 0 ? 0.5 : (v - pMin) / pRange);

    const np = pWin.map(normVal);
    const nr = winIdx.map(idx => rsi[idx] / 100);
    const ne = winIdx.map(idx => ema[idx]).map(normVal);
    const nu = winIdx.map(idx => bb.upper[idx]).map(normVal);
    const nl = winIdx.map(idx => bb.lower[idx]).map(normVal);
    
    const normalizeLocal = (nums: number[]) => {
      const min = Math.min(...nums);
      const max = Math.max(...nums);
      return nums.map(v => (max === min ? 0.5 : (v - min) / (max - min)));
    };

    const nm = normalizeLocal(winIdx.map(idx => macd.histogram[idx]));
    const nml = normalizeLocal(winIdx.map(idx => macd.macdLine[idx]));
    const nroc = normalizeLocal(winIdx.map(idx => roc[idx]));
    const nas = normalizeLocal(winIdx.map(idx => atr[idx]));
    const ne9 = winIdx.map(idx => ema9[idx]).map(normVal);
    const nobv = normalizeLocal(winIdx.map(idx => obv[idx]));
    const nvol = normalizeLocal(winIdx.map(idx => vol[idx]));

    for (let j = 0; j < windowSize; j++) {
      const t = new Date(context[winIdx[j]].time * 1000);
      const hour = t.getUTCHours();
      x.push(
        np[j], nr[j], ne[j], nu[j], nl[j],
        nm[j], nml[j], nroc[j], stochRsi[winIdx[j]], nas[j],
        ne9[j], cross.isBelow[winIdx[j]], cross.isCross[winIdx[j]],
        nobv[j], mfi[winIdx[j]] / 100, nvol[j],
        hour / 24, hour >= 0 && hour <= 9 ? 1 : 0, hour >= 8 && hour <= 17 ? 1 : 0, hour >= 13 && hour <= 22 ? 1 : 0, t.getUTCDay() / 7,
        harami[winIdx[j]], marubozu[winIdx[j]], engulfing[winIdx[j]]
      );
    }
    const result = model.predictMultiple(x, this.settings?.mcPasses || 1);
    return { result, features: x };
  }

  private async processPriceUpdate(price: number) {
    if (!this.isRunning) return;
    
    const nowMs = Date.now();
    if (nowMs - this.lastProcessTime < this.processInterval) return;
    this.lastProcessTime = nowMs;

    this.loadSettings();
    this.checkDailyReset();

    this.lastPrice = price;
    this.lastUpdate = new Date();

    if (!this.model1h || !this.settings) return;

    // 1. Maintain MTF Candle Buffers
    const nowDate = new Date();
    const last1h = (this.candles[this.candles.length - 1]?.time || 0) * 1000;
    const last4h = (this.candles4h[this.candles4h.length - 1]?.time || 0) * 1000;
    const nowTs = nowDate.getTime();

    if (nowTs - last1h >= 3600000 || nowTs - last4h >= 14400000) {
      await this.fetchInitialData();
    }

    // 2. Generate MTF Predictions
    const res1hData = this.generateMTFPrediction(this.candles, this.model1h);
    const res4hData = this.generateMTFPrediction(this.candles4h, this.model4h);

    const res1h = res1hData?.result;
    const res4h = res4hData?.result;
    this.lastFeatures = res1hData?.features ? indicators.mapFeaturesToRecord(res1hData.features.slice((20-1)*24)) : null;

    if (res1h) {
      this.lastVelocity = this.lastPrediction !== null ? res1h.mean.map((p, i) => p - this.lastPrediction![i]) : [0, 0, 0];
      this.lastPrediction = res1h.mean;
      this.lastUncertainty = res1h.std;
    }
    if (res4h) {
      this.lastPrediction4h = res4h.mean;
    }

    const prediction = this.lastPrediction || [0, 0, 1];
    const prediction4h = this.lastPrediction4h || [0, 0, 1];
    const uncertainty = this.lastUncertainty || [0, 0, 0];
    const velocity = this.lastVelocity || [0, 0, 0];

    // 2.1 Calculate DRL Action if model present
    let drlAction: number | null = null;
    if (this.drlModel && res1hData?.features) {
      drlAction = tf.tidy(() => {
        const pos = this.activeTrade ? this.activeTrade.type : 'NEUTRAL';
        const state = [...res1hData.features, ...this.encodePosition(pos as any)];
        const input = tf.tensor2d([state]);
        const probs = (this.drlModel as tf.LayersModel).predict(input) as tf.Tensor;
        return probs.argMax(1).dataSync()[0];
      });
    }

    // Logging MTF signals periodically
    if (this.tickerCount % 20 === 0) {
      this.log(`MTF Sync: 1h[S:${prediction[0].toFixed(2)} L:${prediction[1].toFixed(2)}] 4h[S:${prediction4h[0].toFixed(2)} L:${prediction4h[1].toFixed(2)}]${drlAction !== null ? ` DRL:${drlAction}` : ''}`);
    }

    this.lastParams = {
      ...this.lastParams,
      velocity: velocity,
      session: nowDate.getUTCHours() >= this.settings.asiaStart && nowDate.getUTCHours() < this.settings.asiaEnd ? 'Asia' : 
               nowDate.getUTCHours() >= this.settings.nyStart && nowDate.getUTCHours() < this.settings.nyEnd ? 'New York' : 'Off-Session',
      dayOfWeek: nowDate.getDay(),
      drlAction: drlAction
    };

    // 3. Trading Logic
    if (this.activeTrade) {
      let profitPct = 0;
      
      if (this.isRealTrading) {
        // Use the latest price from ticker and position data from WS/REST
        const entry = this.activeTrade.entryPrice;
        const size = this.activeTrade.size || 0;
        
        if ((this.activeTrade.type === 'CALL_SPREAD' || this.activeTrade.type === 'SHORT_CALL' || this.activeTrade.type === 'PUT_SPREAD' || this.activeTrade.type === 'SHORT_PUT') && this.activeTrade.legs) {
          // Use real PNL from syncPositionWithRest if available
          if (this.activeTrade.pnl !== undefined) {
            // Estimate initial value for profit percentage
            const initialValueUSD = this.activeTrade.legs.reduce((acc, leg) => acc + (leg.entryPrice * Math.abs(leg.size) * 0.001), 0);
            profitPct = this.activeTrade.pnl / Math.max(1, Math.abs(initialValueUSD));
            this.activeTrade.pnlPct = profitPct * 100;
          } else {
            // Fallback to underlying price movement
            const isOptionBearish = this.activeTrade.type === 'CALL_SPREAD' || this.activeTrade.type === 'SHORT_CALL';
            profitPct = isOptionBearish
              ? (this.activeTrade.entryPrice - price) / this.activeTrade.entryPrice
              : (price - this.activeTrade.entryPrice) / this.activeTrade.entryPrice;
          }
          
          // Sync positions periodically
          if (Date.now() - this.lastProcessTime > 15000) {
            this.syncPositionWithRest();
          }
        } else if (size !== 0 && entry > 0) {
          // PNL Calculation logic from user script: (Entry - Mark) * Size * 0.001 (for SHORT)
          // For BTCUSD, contract_value is 0.001
          const contractValue = 0.001;
          let pnl = 0;
          if (this.activeTrade.type === 'SHORT') {
            pnl = (entry - price) * Math.abs(size) * contractValue;
          } else {
            pnl = (price - entry) * Math.abs(size) * contractValue;
          }
          
          const initialValueUSD = Math.abs(size) * contractValue * entry;
          profitPct = pnl / initialValueUSD;
          
          this.activeTrade.pnl = pnl;
          this.activeTrade.pnlPct = profitPct * 100;
          
          // Periodically sync with REST as backup
          if (Date.now() - this.lastProcessTime > 30000) {
            this.syncPositionWithRest();
          }
        } else {
          // Fallback to local calculation if size/entry not yet synced
      if (this.activeTrade.type === 'SHORT' || this.activeTrade.type === 'CALL_SPREAD' || this.activeTrade.type === 'SHORT_CALL') {
        // Fallback to local calculation if size/entry not yet synced
        profitPct = (this.activeTrade.entryPrice - price) / this.activeTrade.entryPrice;
      } else {
        profitPct = (price - this.activeTrade.entryPrice) / this.activeTrade.entryPrice;
      }
    }
  } else {
    // Paper trading logic
    if (this.activeTrade.type === 'SHORT' || this.activeTrade.type === 'CALL_SPREAD' || this.activeTrade.type === 'SHORT_CALL') {
      profitPct = (this.activeTrade.entryPrice - price) / this.activeTrade.entryPrice;
    } else {
      profitPct = (price - this.activeTrade.entryPrice) / this.activeTrade.entryPrice;
    }
  }

  const stopLossPrice = (this.activeTrade.type === 'SHORT' || this.activeTrade.type === 'CALL_SPREAD' || this.activeTrade.type === 'SHORT_CALL')
    ? this.activeTrade.entryPrice * (1 + this.settings.stopLoss)
    : this.activeTrade.entryPrice * (1 - this.settings.stopLoss);
  const takeProfitPrice = (this.activeTrade.type === 'SHORT' || this.activeTrade.type === 'CALL_SPREAD' || this.activeTrade.type === 'SHORT_CALL')
    ? this.activeTrade.entryPrice * (1 - this.settings.takeProfit)
    : this.activeTrade.entryPrice * (1 + this.settings.takeProfit);

  let currentTrailingStop = this.activeTrade.trailingStopPrice;
  let highestProfit = this.activeTrade.highestProfitPct;

  if (profitPct > highestProfit) {
    highestProfit = profitPct;
    this.activeTrade.highestProfitPct = highestProfit;
    if (profitPct >= this.settings.trailingStopActivation) {
      if (this.activeTrade.type === 'SHORT' || this.activeTrade.type === 'CALL_SPREAD' || this.activeTrade.type === 'SHORT_CALL') {
        currentTrailingStop = price * (1 + this.settings.trailingStopOffset);
        if (this.activeTrade.trailingStopPrice === null || currentTrailingStop < this.activeTrade.trailingStopPrice) {
          this.activeTrade.trailingStopPrice = currentTrailingStop;
        }
      } else {
        currentTrailingStop = price * (1 - this.settings.trailingStopOffset);
        if (this.activeTrade.trailingStopPrice === null || currentTrailingStop > this.activeTrade.trailingStopPrice) {
          this.activeTrade.trailingStopPrice = currentTrailingStop;
        }
      }
    }
  }

  let shouldExit = false;
  let reason = '';

  if (this.activeTrade.type === 'SHORT' || this.activeTrade.type === 'CALL_SPREAD' || this.activeTrade.type === 'SHORT_CALL') {
    if (price >= stopLossPrice) {
      shouldExit = true;
      reason = 'STOP_LOSS';
    } else if (price <= takeProfitPrice) {
      shouldExit = true;
      reason = 'TAKE_PROFIT';
    } else if (currentTrailingStop && price >= currentTrailingStop) {
      shouldExit = true;
      reason = 'TRAILING_STOP';
    } else if (prediction[0] * 100 < this.settings.shortExitThreshold) {
      shouldExit = true;
      reason = 'PREDICTION';
    } else if (this.settings.useDRLOnly && drlAction !== null && drlAction !== 1) {
      shouldExit = true;
      reason = 'DRL_REVERSAL';
    }
  } else {
    // Bullish (LONG, PUT_SPREAD, SHORT_PUT)
    if (price <= stopLossPrice) {
      shouldExit = true;
      reason = 'STOP_LOSS';
    } else if (price >= takeProfitPrice) {
      shouldExit = true;
      reason = 'TAKE_PROFIT';
    } else if (currentTrailingStop && price <= currentTrailingStop) {
      shouldExit = true;
      reason = 'TRAILING_STOP';
    } else if (prediction[1] * 100 < this.settings.longExitThreshold) {
      shouldExit = true;
      reason = 'PREDICTION';
    } else if (this.settings.useDRLOnly && drlAction !== null && drlAction !== 0) {
      shouldExit = true;
      reason = 'DRL_REVERSAL';
    }
  }

  // Time-based exit limit
  if (!shouldExit && this.activeTrade) {
    const tradeDurationHrs = (Date.now() - this.activeTrade.entryTime) / (1000 * 60 * 60);
    if (tradeDurationHrs >= (this.settings.maxDurationHours || 12)) {
      shouldExit = true;
      reason = 'TIME_LIMIT';
    }
  }

      if (shouldExit) {
        this.log(`Closing Trade: ${reason} at $${price} | Prediction: S:${prediction[0].toFixed(2)} L:${prediction[1].toFixed(2)} | Profit: ${((profitPct || 0) * 100).toFixed(2)}%`, profitPct >= 0 ? 'success' : 'warning');
        
        const btcQuantity = this.settings.quantityType === 'LOTS' ? this.settings.quantity * 0.001 : this.settings.quantity;
        const profit = this.settings.quantityType === 'USD'
          ? this.settings.quantity * profitPct
          : btcQuantity * ((this.activeTrade.type === 'SHORT' || this.activeTrade.type === 'CALL_SPREAD' || this.activeTrade.type === 'SHORT_CALL') ? (this.activeTrade.entryPrice - price) : (price - this.activeTrade.entryPrice));

        const closedTrade: ClosedTrade = {
          type: this.activeTrade.type,
          entryPrice: this.activeTrade.entryPrice,
          exitPrice: price,
          entryTime: this.activeTrade.entryTime,
          exitTime: Date.now(),
          profit,
          profitPct: profitPct * 100,
          exitReason: reason,
          prediction: this.activeTrade.prediction,
          features: this.activeTrade.features || {},
          orderId: this.activeTrade.orderId
        };
        this.closedTrades.unshift(closedTrade);
        if (this.closedTrades.length > 100) this.closedTrades.pop();

        this.dailyProfit += profit;

        const isShort = this.activeTrade.type === 'SHORT';
        if (this.isRealTrading) {
          if ((this.activeTrade.type === 'CALL_SPREAD' || this.activeTrade.type === 'SHORT_CALL' || this.activeTrade.type === 'PUT_SPREAD' || this.activeTrade.type === 'SHORT_PUT') && this.activeTrade.legs) {
            this.log(`Closing Leg-based Trade: ${this.activeTrade.type}`, 'info');
            for (const leg of this.activeTrade.legs) {
              const exitSide = leg.side === 'sell' ? 'buy' : 'sell';
              await this.placeOptionOrder(leg.symbol, exitSide, leg.size);
            }
          } else {
            await this.placeRealOrder(isShort ? 'buy' : 'sell', this.settings.quantity, true);
          }
        }
        this.activeTrade = null;
        this.lastTradeCloseTime = Date.now();
        this.saveState();
      } else {
        this.activeTrade.highestProfitPct = highestProfit;
        this.activeTrade.trailingStopPrice = currentTrailingStop;
        this.saveState();
      }
    } else {
      // Entry Logic
      const bias = this.settings.biasThreshold || 0;
      const mtfConfluenceThresh = 40; // Hardcoded fallback or use setting

      const canShort = (prediction[0] * 100) > this.settings.shortThreshold && 
                      (prediction4h[0] * 100) > mtfConfluenceThresh &&
                      velocity[0] >= this.settings.minSignalVelocity && 
                      (this.settings.mcPasses <= 1 || uncertainty[0] <= this.settings.maxUncertainty);
      
      const canLong = (prediction[1] * 100) > this.settings.longThreshold && 
                     (prediction4h[1] * 100) > mtfConfluenceThresh &&
                     velocity[1] >= this.settings.minSignalVelocity && 
                     (this.settings.mcPasses <= 1 || uncertainty[1] <= this.settings.maxUncertainty);
      
      let entryType: ActiveTrade['type'] | null = null;
      
      if (this.settings.useDRLOnly) {
        if (drlAction === 1) entryType = 'SHORT';
        else if (drlAction === 0) entryType = 'LONG';
      } else {
        const drlMatchesShort = drlAction === 1;
        const drlMatchesLong = drlAction === 0;

        if (canShort && (!canLong || prediction[0] > prediction[1])) {
          if (!this.settings.useDRLConfluence || drlMatchesShort) {
            entryType = 'SHORT';
          }
        } else if (canLong) {
          if (!this.settings.useDRLConfluence || drlMatchesLong) {
            entryType = 'LONG';
          }
        }
      }

      if (entryType) {
        // Mode filter
        if (entryType === 'SHORT' && !(this.settings.strategyType === 'SHORT_BTC' || this.settings.strategyType === 'BOTH' || this.settings.strategyType === 'CALL_SPREAD' || this.settings.strategyType === 'SHORT_CALL')) {
          entryType = null;
        } else if (entryType === 'LONG' && !(this.settings.strategyType === 'LONG_BTC' || this.settings.strategyType === 'BOTH' || this.settings.strategyType === 'PUT_SPREAD' || this.settings.strategyType === 'SHORT_PUT')) {
          entryType = null;
        }
      }

      if (entryType) {
        const cooldownMs = 10 * 1000;
        if (Date.now() - this.lastTradeCloseTime < cooldownMs) return;

        // Daily Limits check
        if (this.settings.dailyProfitLimit > 0 && this.dailyProfit >= this.settings.dailyProfitLimit) return;
        if (this.settings.dailyLossLimit > 0 && this.dailyProfit <= -this.settings.dailyLossLimit) return;

        // Session check
        let canTrade = true;
        const hour = nowDate.getUTCHours();
        if (this.settings.useSessionTrading) {
          const isAsia = hour >= this.settings.asiaStart && hour < this.settings.asiaEnd;
          const isNY = hour >= this.settings.nyStart && hour < this.settings.nyEnd;
          canTrade = isAsia || isNY;
        }

        if (canTrade && !this.activeTrade && !this.isOpeningTrade) {
          this.isOpeningTrade = true;
          try {
            if (this.settings.strategyType === 'CALL_SPREAD' && entryType === 'SHORT') {
              await this.executeCallSpread(price, prediction, this.lastFeatures || {});
            } else if (this.settings.strategyType === 'SHORT_CALL' && entryType === 'SHORT') {
              await this.executeShortCall(price, prediction, this.lastFeatures || {});
            } else if (this.settings.strategyType === 'PUT_SPREAD' && entryType === 'LONG') {
              await this.executePutSpread(price, prediction, this.lastFeatures || {});
            } else if (this.settings.strategyType === 'SHORT_PUT' && entryType === 'LONG') {
              await this.executeShortPut(price, prediction, this.lastFeatures || {});
            } else {
              this.log(`Opening ${entryType} at $${price} | Prediction: S:${prediction[0].toFixed(2)} L:${prediction[1].toFixed(2)}`, 'success');
              this.activeTrade = {
                type: entryType,
                entryPrice: price,
                entryTime: Date.now(),
                highestProfitPct: 0,
                trailingStopPrice: null,
                prediction: prediction,
                features: this.lastFeatures || {}
              };
              if (this.isRealTrading) {
                this.placeRealOrder(entryType === 'SHORT' ? 'sell' : 'buy', this.settings.quantity, false).then(orderId => {
                  if (orderId && this.activeTrade) {
                    this.activeTrade.orderId = orderId;
                    this.saveState();
                    setTimeout(() => this.syncPositionWithRest(), 1000);
                  }
                });
              }
              this.saveState();
            }
          } finally {
            this.isOpeningTrade = false;
          }
        }
      }
    }
  }

  private async syncPositionWithRest() {
    if (!this.isRealTrading || !this.activeTrade) return;
    try {
      const positions = await deltaRequest('GET', '/v2/positions?underlying_asset_symbol=BTC');
      if (positions.success && Array.isArray(positions.result)) {
        if ((this.activeTrade.type === 'CALL_SPREAD' || this.activeTrade.type === 'SHORT_CALL' || this.activeTrade.type === 'PUT_SPREAD' || this.activeTrade.type === 'SHORT_PUT') && this.activeTrade.legs) {
          let totalPnl = 0;
          for (const leg of this.activeTrade.legs) {
            const legPos = positions.result.find((p: any) => p.symbol === leg.symbol);
            if (legPos) {
              leg.entryPrice = parseFloat(legPos.avg_entry_price);
              leg.size = parseFloat(legPos.size);
              leg.pnl = parseFloat(legPos.unrealized_pnl);
              leg.markPrice = parseFloat(legPos.mark_price);
              totalPnl += leg.pnl;
            }
          }
          this.activeTrade.pnl = totalPnl;
          // For profitPct of a spread, it's complex, but we can use totalPnl / totalInitialMargin if we had it.
          // For now, let's just update pnl.
          this.saveState();
        } else {
          const btcPos = positions.result.find((p: any) => p.symbol === 'BTCUSD' || p.symbol === 'BTCUSD_P');
          if (btcPos) {
            const entry = parseFloat(btcPos.avg_entry_price);
            const size = parseFloat(btcPos.size);
            if (size !== 0 && entry > 0) {
              this.activeTrade.entryPrice = entry;
              this.activeTrade.size = size;
              this.activeTrade.pnl = parseFloat(btcPos.unrealized_pnl);
              this.saveState();
            }
          }
        }
      }
    } catch (err) {
      // Silent catch
    }
  }

  private async placeRealOrder(side: 'buy' | 'sell', quantity: number, isClose: boolean = false): Promise<string | null> {
    try {
      let lots = quantity;
      if (this.settings?.quantityType === 'BTC') lots = quantity * 1000;
      else if (this.settings?.quantityType === 'USD' && this.lastPrice) {
        lots = (quantity / this.lastPrice) / 0.001;
      }
      const finalSize = Math.max(1, Math.floor(lots));

      if (!this.lastPrice) {
        this.log('Cannot place order: last price unknown', 'error');
        return null;
      }

      const productId = productMap['BTCUSD'] || 1;

      const orderData: any = {
        product_id: productId,
        side,
        order_type: 'market_order',
        size: finalSize
      };

      if (isClose) {
        orderData.reduce_only = true;
      }

      this.log(`Placing real market order: ${side} ${finalSize} lots | Close: ${isClose}`, 'info');
      
      const result = await deltaRequest('POST', '/v2/orders', orderData);
      
      if (result.success || result.result) {
        const orderId = result.result?.id || result.result?.order_id;
        this.log(`Real order successful: ${side} ${finalSize} lots | Order ID: ${orderId}`, 'success');
        return orderId ? String(orderId) : null;
      } else {
        this.log(`Real order failed: ${JSON.stringify(result.error || result)}`, 'error');
        return null;
      }
    } catch (err: any) {
      // deltaRequest throws stringified error, let's try to parse it for better logging if possible
      let errorMessage = err.message || String(err);
      try {
        const parsed = JSON.parse(err.message);
        errorMessage = JSON.stringify(parsed);
      } catch (e) {
        // Not JSON, keep original
      }
      this.log(`Error placing real order: ${errorMessage}`, 'error');
      return null;
    }
  }

  private async findOptionByDelta(isCall: boolean, targetDelta: number): Promise<any> {
    try {
      const query = isCall ? "C-BTC" : "P-BTC";
      const productsResult = await deltaRequest("GET", `/v2/products?query=${query}`);
      if (!productsResult.success) return null;

      const now = new Date();
      const tomorrow = new Date(now);
      tomorrow.setUTCDate(tomorrow.getUTCDate() + 1);
      
      const day = tomorrow.getUTCDate();
      const month = tomorrow.getUTCMonth() + 1;
      const year = tomorrow.getUTCFullYear();
      
      const d = day.toString();
      const dd = day < 10 ? `0${day}` : d;
      const m = month.toString();
      const mm = month < 10 ? `0${month}` : m;
      const y = year.toString();

      const formats = [
        `${d}-${m}-${y}`,
        `${dd}-${m}-${y}`,
        `${dd}-${mm}-${y}`,
        `${d}-${mm}-${y}`
      ];

      const tomorrowOptions = productsResult.result.filter((p: any) => {
        const isCorrectType = isCall ? p.contract_type === 'call_options' : p.contract_type === 'put_options';
        const isBTC = p.underlying_asset && p.underlying_asset.symbol === 'BTC';
        const desc = p.description || "";
        const isTomorrow = formats.some(f => desc.includes(f));
        return isCorrectType && isBTC && isTomorrow;
      });

      if (tomorrowOptions.length === 0) {
        this.log(`No BTC ${isCall ? 'call' : 'put'} options found for expiry ${d}-${m}-${y}`, 'warning');
        return null;
      }

      let bestOption = null;
      let minDeltaDiff = Infinity;

      this.log(`Checking ${tomorrowOptions.length} options for delta ${targetDelta}...`, 'info');

      // Fetch tickers in parallel with concurrency limit to avoid hanging on large /v2/tickers call
      const tickers: any[] = [];
      const batchSize = 10;
      for (let i = 0; i < tomorrowOptions.length; i += batchSize) {
        const batch = tomorrowOptions.slice(i, i + batchSize);
        const results = await Promise.all(batch.map(async (opt: any) => {
          try {
            const res = await deltaRequest("GET", `/v2/tickers/${opt.symbol}`);
            return res.success ? { opt, ticker: res.result } : null;
          } catch (e) {
            return null;
          }
        }));
        tickers.push(...results.filter((r: any) => r !== null));
      }

      for (const item of tickers) {
        const { opt, ticker } = item;
        if (ticker && ticker.greeks) {
          const delta = Math.abs(parseFloat(ticker.greeks.delta));
          const diff = Math.abs(delta - targetDelta);
          if (diff < minDeltaDiff) {
            minDeltaDiff = diff;
            bestOption = {
              symbol: opt.symbol,
              id: opt.id,
              delta: delta,
              mark_price: parseFloat(ticker.mark_price),
              best_bid: parseFloat(ticker.quotes.best_bid),
              best_ask: parseFloat(ticker.quotes.best_ask)
            };
          }
        }
      }
      if (bestOption) {
        this.log(`Found best option: ${bestOption.symbol} with delta ${(bestOption.delta || 0).toFixed(3)}`, 'success');
      } else {
        this.log(`No option found with greeks for delta ${targetDelta}`, 'warning');
      }
      return bestOption;
    } catch (err) {
      this.log(`Error finding options by delta: ${err}`, 'error');
      return null;
    }
  }

  private async placeOptionOrder(symbol: string, side: 'buy' | 'sell', quantity: number): Promise<string | null> {
    try {
      const productId = productMap[symbol];
      if (!productId) {
        // Try to fetch products again if not found
        await fetchProducts();
      }
      const finalProductId = productMap[symbol];
      if (!finalProductId) {
        this.log(`Product ID not found for ${symbol}`, 'error');
        return null;
      }

      // For options, size must be an integer (number of lots/contracts)
      let lots = quantity;
      if (this.settings?.quantityType === 'BTC') lots = quantity * 1000;
      else if (this.settings?.quantityType === 'USD' && this.lastPrice) {
        lots = (quantity / this.lastPrice) / 0.001;
      }
      const finalSize = Math.max(1, Math.floor(lots));

      const orderData = {
        product_id: finalProductId,
        side,
        order_type: 'market_order',
        size: finalSize
      };

      this.log(`Placing option market order: ${side} ${finalSize} ${symbol}`, 'info');
      const result = await deltaRequest('POST', '/v2/orders', orderData);
      
      if (result.success || result.result) {
        return String(result.result?.id || result.result?.order_id);
      }
      return null;
    } catch (err) {
      this.log(`Error placing option order: ${err}`, 'error');
      return null;
    }
  }

  private async executeCallSpread(price: number, prediction: number[], features: any) {
    if (!this.settings) return;
    this.log(`Executing Call Spread strategy...`, 'info');

    const shortLeg = await this.findOptionByDelta(true, this.settings.shortCallDelta);
    const longLeg = await this.findOptionByDelta(true, this.settings.longCallDelta);

    if (!shortLeg || !longLeg) {
      this.log('Failed to find suitable option legs for call spread', 'error');
      return;
    }

    if (shortLeg.symbol === longLeg.symbol) {
      this.log('Short and long legs are the same option. Check delta settings.', 'warning');
      return;
    }

    if (this.isRealTrading) {
      const shortOrderId = await this.placeOptionOrder(shortLeg.symbol, 'sell', this.settings.quantity);
      const longOrderId = await this.placeOptionOrder(longLeg.symbol, 'buy', this.settings.quantity);

      if (shortOrderId && longOrderId) {
        this.activeTrade = {
          type: 'CALL_SPREAD',
          entryPrice: price,
          entryTime: Date.now(),
          highestProfitPct: 0,
          trailingStopPrice: null,
          prediction: prediction,
          features: features,
          legs: [
            { symbol: shortLeg.symbol, side: 'sell', entryPrice: shortLeg.best_bid, size: this.settings.quantity },
            { symbol: longLeg.symbol, side: 'buy', entryPrice: longLeg.best_ask, size: this.settings.quantity }
          ]
        };
        this.saveState();
        this.log(`Real Call Spread opened: Sell ${shortLeg.symbol} | Buy ${longLeg.symbol}`, 'success');
        // Sync positions after a short delay to allow exchange to process
        setTimeout(() => this.syncPositionWithRest(), 2000);
      }
    } else {
      this.activeTrade = {
        type: 'CALL_SPREAD',
        entryPrice: price,
        entryTime: Date.now(),
        highestProfitPct: 0,
        trailingStopPrice: null,
        prediction: prediction,
        features: features,
        legs: [
          { symbol: shortLeg.symbol, side: 'sell', entryPrice: shortLeg.mark_price, size: this.settings.quantity },
          { symbol: longLeg.symbol, side: 'buy', entryPrice: longLeg.mark_price, size: this.settings.quantity }
        ]
      };
      this.saveState();
      this.log(`Paper Call Spread opened: Sell ${shortLeg.symbol} @ ${shortLeg.mark_price} | Buy ${longLeg.symbol} @ ${longLeg.mark_price}`, 'success');
    }
  }

  private async executeShortCall(price: number, prediction: number[], features: any) {
    if (!this.settings) return;
    this.log(`Executing Short Call strategy...`, 'info');

    const shortLeg = await this.findOptionByDelta(true, this.settings.shortCallDelta);

    if (!shortLeg) {
      this.log('Failed to find suitable option leg for short call', 'error');
      this.isOpeningTrade = false;
      return;
    }

    if (this.isRealTrading) {
      const shortOrderId = await this.placeOptionOrder(shortLeg.symbol, 'sell', this.settings.quantity);

      if (shortOrderId) {
        this.activeTrade = {
          type: 'SHORT_CALL',
          entryPrice: price,
          entryTime: Date.now(),
          highestProfitPct: 0,
          trailingStopPrice: null,
          prediction: prediction,
          features: features,
          legs: [
            { symbol: shortLeg.symbol, side: 'sell', entryPrice: shortLeg.best_bid, size: this.settings.quantity }
          ]
        };
        this.saveState();
        this.log(`Real Short Call opened: Sell ${shortLeg.symbol}`, 'success');
        // Sync positions after a short delay to allow exchange to process
        setTimeout(() => this.syncPositionWithRest(), 2000);
      }
    } else {
      this.activeTrade = {
        type: 'SHORT_CALL',
        entryPrice: price,
        entryTime: Date.now(),
        highestProfitPct: 0,
        trailingStopPrice: null,
        prediction: prediction,
        features: features,
        legs: [
          { symbol: shortLeg.symbol, side: 'sell', entryPrice: shortLeg.mark_price, size: this.settings.quantity }
        ]
      };
      this.saveState();
      this.log(`Paper Short Call opened: Sell ${shortLeg.symbol} @ ${shortLeg.mark_price}`, 'success');
    }
  }

  private async executePutSpread(price: number, prediction: number[], features: any) {
    if (!this.settings) return;
    this.log(`Executing Put Spread strategy...`, 'info');

    const shortLeg = await this.findOptionByDelta(false, this.settings.shortPutDelta);
    const longLeg = await this.findOptionByDelta(false, this.settings.longPutDelta);

    if (!shortLeg || !longLeg) {
      this.log('Failed to find suitable option legs for put spread', 'error');
      this.isOpeningTrade = false;
      return;
    }

    if (shortLeg.symbol === longLeg.symbol) {
      this.log('Short and long legs are the same option. Check delta settings.', 'warning');
      this.isOpeningTrade = false;
      return;
    }

    if (this.isRealTrading) {
      const shortOrderId = await this.placeOptionOrder(shortLeg.symbol, 'sell', this.settings.quantity);
      const longOrderId = await this.placeOptionOrder(longLeg.symbol, 'buy', this.settings.quantity);

      if (shortOrderId && longOrderId) {
        this.activeTrade = {
          type: 'PUT_SPREAD',
          entryPrice: price,
          entryTime: Date.now(),
          highestProfitPct: 0,
          trailingStopPrice: null,
          prediction: prediction,
          features: features,
          legs: [
            { symbol: shortLeg.symbol, side: 'sell', entryPrice: shortLeg.best_bid, size: this.settings.quantity },
            { symbol: longLeg.symbol, side: 'buy', entryPrice: longLeg.best_ask, size: this.settings.quantity }
          ]
        };
        this.saveState();
        this.log(`Real Put Spread opened: Sell ${shortLeg.symbol} | Buy ${longLeg.symbol}`, 'success');
        setTimeout(() => this.syncPositionWithRest(), 2000);
      }
    } else {
      this.activeTrade = {
        type: 'PUT_SPREAD',
        entryPrice: price,
        entryTime: Date.now(),
        highestProfitPct: 0,
        trailingStopPrice: null,
        prediction: prediction,
        features: features,
        legs: [
          { symbol: shortLeg.symbol, side: 'sell', entryPrice: shortLeg.mark_price, size: this.settings.quantity },
          { symbol: longLeg.symbol, side: 'buy', entryPrice: longLeg.mark_price, size: this.settings.quantity }
        ]
      };
      this.saveState();
      this.log(`Paper Put Spread opened: Sell ${shortLeg.symbol} @ ${shortLeg.mark_price} | Buy ${longLeg.symbol} @ ${longLeg.mark_price}`, 'success');
    }
  }

  private async executeShortPut(price: number, prediction: number[], features: any) {
    if (!this.settings) return;
    this.log(`Executing Short Put strategy...`, 'info');

    const shortLeg = await this.findOptionByDelta(false, this.settings.shortPutDelta);

    if (!shortLeg) {
      this.log('Failed to find suitable option leg for short put', 'error');
      this.isOpeningTrade = false;
      return;
    }

    if (this.isRealTrading) {
      const shortOrderId = await this.placeOptionOrder(shortLeg.symbol, 'sell', this.settings.quantity);

      if (shortOrderId) {
        this.activeTrade = {
          type: 'SHORT_PUT',
          entryPrice: price,
          entryTime: Date.now(),
          highestProfitPct: 0,
          trailingStopPrice: null,
          prediction: prediction,
          features: features,
          legs: [
            { symbol: shortLeg.symbol, side: 'sell', entryPrice: shortLeg.best_bid, size: this.settings.quantity }
          ]
        };
        this.saveState();
        this.log(`Real Short Put opened: Sell ${shortLeg.symbol}`, 'success');
        setTimeout(() => this.syncPositionWithRest(), 2000);
      }
    } else {
      this.activeTrade = {
        type: 'SHORT_PUT',
        entryPrice: price,
        entryTime: Date.now(),
        highestProfitPct: 0,
        trailingStopPrice: null,
        prediction: prediction,
        features: features,
        legs: [
          { symbol: shortLeg.symbol, side: 'sell', entryPrice: shortLeg.mark_price, size: this.settings.quantity }
        ]
      };
      this.saveState();
      this.log(`Paper Short Put opened: Sell ${shortLeg.symbol} @ ${shortLeg.mark_price}`, 'success');
    }
  }
}

export const tradingService = new TradingService();
