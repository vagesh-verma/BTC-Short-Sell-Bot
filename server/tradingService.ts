import WebSocket from 'ws';
import { GRUModel } from './modelService';
import * as indicators from './indicatorService';
import fetch from 'node-fetch';
import crypto from 'crypto';
import { deltaRequest, productMap, fetchProducts } from './deltaApi';
import fs from 'fs';
import path from 'path';

interface TradingSettings {
  threshold: number;
  exitThreshold: number;
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
  strategyType: 'SHORT_BTC' | 'CALL_SPREAD' | 'SHORT_CALL';
  shortCallDelta: number;
  longCallDelta: number;
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
}

interface ActiveTrade {
  type: 'LONG' | 'SHORT' | 'CALL_SPREAD' | 'SHORT_CALL';
  entryPrice: number;
  entryTime: number;
  highestProfitPct: number;
  trailingStopPrice: number | null;
  prediction: number;
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
  type: 'LONG' | 'SHORT' | 'CALL_SPREAD' | 'SHORT_CALL';
  entryPrice: number;
  exitPrice: number;
  entryTime: number;
  exitTime: number;
  profit: number;
  profitPct: number;
  exitReason: string;
  prediction: number;
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
  private ws: WebSocket | null = null;
  private candles: any[] = [];
  private candles4h: any[] = [];
  private activeTrade: ActiveTrade | null = null;
  private closedTrades: ClosedTrade[] = [];
  private lastPrediction: number | null = null;
  private previousPrediction: number | null = null;
  private lastPredictionTime: number = 0;
  private lastUncertainty: number = 0;
  private lastVelocity: number = 0;
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
        if ((this.activeTrade.type === 'CALL_SPREAD' || this.activeTrade.type === 'SHORT_CALL') && this.activeTrade.legs) {
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

  public setModels(model1h: GRUModel, model4h: GRUModel) {
    this.model1h = model1h;
    this.model4h = model4h;
    this.log('Models updated on server', 'success');
  }

  public getStatus() {
    return {
      isRunning: this.isRunning,
      isRealTrading: this.isRealTrading,
      activeTrade: this.activeTrade,
      closedTrades: this.closedTrades.slice(0, 50),
      lastPrediction: this.lastPrediction,
      lastPrice: this.lastPrice,
      lastFeatures: this.lastFeatures,
      lastParams: this.lastParams,
      lastUpdate: this.lastUpdate,
      logs: this.logs.slice(0, 20),
      hasModels: !!(this.model1h && this.model4h),
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
      this.log('Fetching initial candle data...');
      const endTs = Date.now();
      const startTs = endTs - (30 * 24 * 60 * 60 * 1000);
      
      const fetchCandles = async (resolution: string, start: number) => {
        let symbol = 'BTCUSD';
        let url = `https://api.india.delta.exchange/v2/history/candles?symbol=${symbol}&resolution=${resolution}&start=${Math.floor(start/1000)}&end=${Math.floor(endTs/1000)}`;
        let res = await fetch(url);
        let data = await res.json() as any;
        
        if (!data.result || data.result.length === 0) {
          symbol = 'BTCUSD_P';
          url = `https://api.india.delta.exchange/v2/history/candles?symbol=${symbol}&resolution=${resolution}&start=${Math.floor(start/1000)}&end=${Math.floor(endTs/1000)}`;
          res = await fetch(url);
          data = await res.json() as any;
        }
        
        return data.result || [];
      };

      this.candles = await fetchCandles('1h', startTs);
      this.candles4h = await fetchCandles('4h', startTs - (20 * 4 * 60 * 60 * 1000));
      this.lastRefreshTime = Date.now();
      this.log(`Loaded ${this.candles.length} 1h candles and ${this.candles4h.length} 4h candles`);
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

  private async processPriceUpdate(price: number) {
    if (!this.isRunning) return;
    
    const nowMs = Date.now();
    if (nowMs - this.lastProcessTime < this.processInterval) {
      return;
    }
    this.lastProcessTime = nowMs;

    // Reload settings to ensure we use the latest threshold
    this.loadSettings();
    this.checkDailyReset();

    if (this.lastPrice === null) {
      this.log(`First price received: $${price}`, 'success');
    }
    this.lastPrice = price;
    this.lastUpdate = new Date();

    if (!this.model1h || !this.model4h || !this.settings) {
      if (this.tickerCount % 100 === 0) {
        this.log(`Waiting for models/settings. Models: ${!!this.model1h}/${!!this.model4h}, Settings: ${!!this.settings}`, 'warning');
      }
      return;
    }

    // Check if we need to update candles (every hour)
    const nowDate = new Date();
    if (this.candles.length > 0) {
      const lastCandleTime = this.candles[this.candles.length - 1].time * 1000;
      const oneHourMs = 60 * 60 * 1000;
      const timeSinceLastCandle = nowDate.getTime() - lastCandleTime;
      const timeSinceLastRefresh = nowDate.getTime() - this.lastRefreshTime;

      // Only refresh if:
      // 1. More than an hour has passed since the last candle started
      // 2. We haven't tried refreshing in the last 2 minutes (to avoid spamming if the exchange is slow to finalize)
      if (timeSinceLastCandle >= oneHourMs && timeSinceLastRefresh >= 2 * 60 * 1000) {
        this.log('New hour detected, refreshing candles...');
        await this.fetchInitialData();
      }
    }

    // 1. Get completed candles
    const nowSeconds = Math.floor(Date.now() / 1000);
    const completed1h = this.candles.filter(c => c.time + 3600 <= nowSeconds);
    const completed4h = this.candles4h.filter(c => c.time + 14400 <= nowSeconds);

    const windowSize = 20;
    if (completed1h.length < windowSize || completed4h.length < windowSize) {
      if (this.tickerCount % 100 === 0) {
        this.log(`Insufficient data: 1h=${completed1h.length}, 4h=${completed4h.length}`, 'warning');
      }
      return;
    }

    const lastCompleted1hTime = completed1h[completed1h.length - 1].time;
    
    // Only recalculate prediction when a new hour is finalized
    if (lastCompleted1hTime !== this.lastPredictionTime) {
      this.log(`New hour finalized: ${new Date(lastCompleted1hTime * 1000).toISOString()}. Calculating prediction...`);

      const normalize = (arr: number[]) => {
        const min = Math.min(...arr);
        const max = Math.max(...arr);
        return arr.map(v => (max === min ? 0 : (v - min) / (max - min)));
      };

      // A. Calculate Secondary Predictions for the 1h Window
      const bufferSize = 100;
      const windowSize = 20;
      
      if (completed1h.length < bufferSize || completed4h.length < bufferSize) {
        this.log(`Insufficient data: 1h=${completed1h.length}, 4h=${completed4h.length}`, 'warning');
        return;
      }

      const context1h = completed1h.slice(completed1h.length - bufferSize);
      const prices1h = context1h.map(c => c.close);
      const highs1h = context1h.map(c => c.high);
      const lows1h = context1h.map(c => c.low);
      const volumes1h = context1h.map(c => c.volume);
      const hours1h = context1h.map(c => new Date(c.time * 1000).getHours());
      const days1h = context1h.map(c => new Date(c.time * 1000).getDay());
      const opens1h = context1h.map(c => c.open);

      // We need secondary predictions for each of the last 20 1h candles
      const secondaryPredictions: number[] = [];
      const winIdx1h = Array.from({ length: windowSize }, (_, k) => bufferSize - windowSize + k);
      
      for (const idx of winIdx1h) {
        const targetTime = context1h[idx].time;
        
        let last4hIdx = -1;
        for (let j = completed4h.length - 1; j >= 0; j--) {
          if (completed4h[j].time + 14400 <= targetTime) {
            last4hIdx = j;
            break;
          }
        }

        if (last4hIdx < windowSize - 1) {
          secondaryPredictions.push(0.5);
          continue;
        }

        const subContext4h = completed4h.slice(last4hIdx - windowSize + 1, last4hIdx + 1);
        const subPrices4h = subContext4h.map(c => c.close);
        const subHighs4h = subContext4h.map(c => c.high);
        const subLows4h = subContext4h.map(c => c.low);
        const subVolumes4h = subContext4h.map(c => c.volume);
        const subHours4h = subContext4h.map(c => new Date(c.time * 1000).getHours());
        const subDays4h = subContext4h.map(c => new Date(c.time * 1000).getDay());
        const subOpens4h = subContext4h.map(c => c.open);

        const periods = this.settings?.indicatorPeriods || { rsi: 14, ema: 20, ema9: 9, bb: 20, mfi: 14, volatility: 20 };
        const rsi4h = indicators.calculateRSI(subPrices4h, periods.rsi);
        const ema4h = indicators.calculateEMA(subPrices4h, periods.ema);
        const ema9_4h = indicators.calculateEMA(subPrices4h, periods.ema9);
        const bb4h = indicators.calculateBollingerBands(subPrices4h, periods.bb);
        const macd4h = indicators.calculateMACD(subPrices4h);
        const stochRsi4h = indicators.calculateStochasticRSI(rsi4h);
        const atr4h = indicators.calculateATR(subHighs4h, subLows4h, subPrices4h);
        const cross4h = indicators.calculateEMACross(ema9_4h, ema4h);
        const obv4h = indicators.calculateOBV(subPrices4h, subVolumes4h);
        const mfi4h = indicators.calculateMFI(subHighs4h, subLows4h, subPrices4h, subVolumes4h, periods.mfi);
        const harami4h = indicators.calculateBearishHarami(subOpens4h, subPrices4h);
        const marubozu4h = indicators.calculateMarubozu(subOpens4h, subHighs4h, subLows4h, subPrices4h);
        const engulfing4h = indicators.calculateEngulfing(subOpens4h, subPrices4h);

        const x4h: number[] = [];
        const np4h = normalize(subPrices4h);
        const nr4h = rsi4h.map(v => v / 100);
        const ne4h = normalize(ema4h);
        const nu4h = normalize(bb4h.upper);
        const nl4h = normalize(bb4h.lower);
        const nm4h = normalize(macd4h.histogram);
        const na4h = normalize(atr4h);
        const ne9_4h = normalize(ema9_4h);
        const nobv4h = normalize(obv4h);
        const nvol4h = normalize(indicators.calculateVolatility(subPrices4h));

        for (let j = 0; j < windowSize; j++) {
          const h = subHours4h[j];
          x4h.push(
            np4h[j], nr4h[j], ne4h[j], nu4h[j], nl4h[j],
            nm4h[j], stochRsi4h[j], na4h[j], ne9_4h[j],
            cross4h.isBelow[j], cross4h.isCross[j], nobv4h[j],
            mfi4h[j] / 100, nvol4h[j],
            h / 24, h >= 0 && h <= 9 ? 1 : 0, h >= 8 && h <= 17 ? 1 : 0, h >= 13 && h <= 22 ? 1 : 0, subDays4h[j] / 7,
            harami4h[j], marubozu4h[j], engulfing4h[j]
          );
        }

        try {
          secondaryPredictions.push(this.model4h.predict(x4h));
        } catch (err) {
          secondaryPredictions.push(0.5);
        }
      }

      // B. Calculate 1h Prediction
      const periods = this.settings?.indicatorPeriods || { rsi: 14, ema: 20, ema9: 9, bb: 20, mfi: 14, volatility: 20 };
      const rsi1h = indicators.calculateRSI(prices1h, periods.rsi);
      const ema1h = indicators.calculateEMA(prices1h, periods.ema);
      const ema9_1h = indicators.calculateEMA(prices1h, periods.ema9);
      const bb1h = indicators.calculateBollingerBands(prices1h, periods.bb);
      const macd1h = indicators.calculateMACD(prices1h);
      const stochRsi1h = indicators.calculateStochasticRSI(rsi1h);
      const atr1h = indicators.calculateATR(highs1h, lows1h, prices1h);
      const cross1h = indicators.calculateEMACross(ema9_1h, ema1h);
      const obv1h = indicators.calculateOBV(prices1h, volumes1h);
      const mfi1h = indicators.calculateMFI(highs1h, lows1h, prices1h, volumes1h, periods.mfi);
      const vol1h = indicators.calculateVolatility(prices1h, periods.volatility);
      const harami1h = indicators.calculateBearishHarami(opens1h, prices1h);
      const marubozu1h = indicators.calculateMarubozu(opens1h, highs1h, lows1h, prices1h);
      const engulfing1h = indicators.calculateEngulfing(opens1h, prices1h);

      const x1h: number[] = [];
      
      const pWin1h = winIdx1h.map(idx => prices1h[idx]);
      const rWin1h = winIdx1h.map(idx => rsi1h[idx]);
      const eWin1h = winIdx1h.map(idx => ema1h[idx]);
      const uWin1h = winIdx1h.map(idx => bb1h.upper[idx]);
      const lWin1h = winIdx1h.map(idx => bb1h.lower[idx]);
      const mWin1h = winIdx1h.map(idx => macd1h.histogram[idx]);
      const sWin1h = winIdx1h.map(idx => stochRsi1h[idx]);
      const aWin1h = winIdx1h.map(idx => atr1h[idx]);
      const e9Win1h = winIdx1h.map(idx => ema9_1h[idx]);
      const belowWin1h = winIdx1h.map(idx => cross1h.isBelow[idx]);
      const crossWin1h = winIdx1h.map(idx => cross1h.isCross[idx]);
      const obvWin1h = winIdx1h.map(idx => obv1h[idx]);
      const mfiWin1h = winIdx1h.map(idx => mfi1h[idx]);
      const volWin1h = winIdx1h.map(idx => vol1h[idx]);
      const hourWin1h = winIdx1h.map(idx => hours1h[idx]);
      const dayWin1h = winIdx1h.map(idx => days1h[idx]);
      const haramiWin1h = winIdx1h.map(idx => harami1h[idx]);
      const marubozuWin1h = winIdx1h.map(idx => marubozu1h[idx]);
      const engulfingWin1h = winIdx1h.map(idx => engulfing1h[idx]);

      const np1h = normalize(pWin1h);
      const nr1h = rWin1h.map(v => v / 100);
      const ne1h = normalize(eWin1h);
      const nu1h = normalize(uWin1h);
      const nl1h = normalize(lWin1h);
      const nm1h = normalize(mWin1h);
      const na1h = normalize(aWin1h);
      const ne9_1h = normalize(e9Win1h);
      const nobv1h = normalize(obvWin1h);
      const nvol1h = normalize(volWin1h);

      for (let j = 0; j < windowSize; j++) {
        const h = hourWin1h[j];
        x1h.push(
          np1h[j], nr1h[j], ne1h[j], nu1h[j], nl1h[j],
          nm1h[j], sWin1h[j], na1h[j], secondaryPredictions[j],
          ne9_1h[j], belowWin1h[j], crossWin1h[j],
          nobv1h[j], mfiWin1h[j] / 100, nvol1h[j],
          h / 24, h >= 0 && h <= 9 ? 1 : 0, h >= 8 && h <= 17 ? 1 : 0, h >= 13 && h <= 22 ? 1 : 0, dayWin1h[j] / 7,
          haramiWin1h[j], marubozuWin1h[j], engulfingWin1h[j]
        );
      }

      let prediction = 0;
      let uncertainty = 0;
      try {
        const mcPasses = this.settings?.mcPasses || 1;
        const result = this.model1h.predictMultiple(x1h, mcPasses);
        prediction = result.mean;
        uncertainty = result.std;
      } catch (err: any) {
        this.log(`1h Prediction failed: ${err.message}`, 'error');
      }

      const featureNames = [
        'Price', 'RSI', 'EMA', 'BB_Upper', 'BB_Lower', 'MACD_Hist', 'Stoch_RSI', 'ATR', 'Secondary_Pred',
        'EMA9', 'Below', 'Cross', 'OBV', 'MFI', 'Volatility', 'Hour', 'Asia', 'London', 'NY', 'Day',
        'Harami', 'Marubozu', 'Engulfing'
      ];
      const currentFeatures: Record<string, number> = {};
      const last1hFeatures = [
        np1h[windowSize - 1], nr1h[windowSize - 1], ne1h[windowSize - 1], nu1h[windowSize - 1], nl1h[windowSize - 1],
        nm1h[windowSize - 1], stochRsi1h[windowSize - 1], na1h[windowSize - 1], secondaryPredictions[secondaryPredictions.length - 1],
        ne9_1h[windowSize - 1], cross1h.isBelow[windowSize - 1], cross1h.isCross[windowSize - 1],
        nobv1h[windowSize - 1], mfi1h[windowSize - 1] / 100, nvol1h[windowSize - 1],
        hours1h[windowSize - 1] / 24, 
        hours1h[windowSize - 1] >= 0 && hours1h[windowSize - 1] <= 9 ? 1 : 0,
        hours1h[windowSize - 1] >= 8 && hours1h[windowSize - 1] <= 17 ? 1 : 0,
        hours1h[windowSize - 1] >= 13 && hours1h[windowSize - 1] <= 22 ? 1 : 0,
        days1h[windowSize - 1] / 7,
        harami1h[windowSize - 1], marubozu1h[windowSize - 1], engulfing1h[windowSize - 1]
      ];
      featureNames.forEach((name, idx) => { currentFeatures[name] = last1hFeatures[idx]; });

      this.lastVelocity = this.lastPrediction !== null ? (prediction - this.lastPrediction) : 0;
      this.previousPrediction = this.lastPrediction;
      this.lastPrediction = prediction;
      this.lastUncertainty = uncertainty;
      this.lastFeatures = currentFeatures;
      this.lastPredictionTime = lastCompleted1hTime;
      this.lastParams = {
        ...this.lastParams,
        rsi: rsi1h[rsi1h.length - 1],
        ema: ema1h[ema1h.length - 1],
        ema9: ema9_1h[ema9_1h.length - 1],
        bbUpper: bb1h.upper[bb1h.upper.length - 1],
        bbLower: bb1h.lower[bb1h.lower.length - 1],
        macdHist: macd1h.histogram[macd1h.histogram.length - 1],
        stochRsi: stochRsi1h[stochRsi1h.length - 1],
        atr: atr1h[atr1h.length - 1],
        emaCross: cross1h.isBelow[cross1h.isBelow.length - 1] === 1,
        secondaryPrediction: secondaryPredictions[secondaryPredictions.length - 1],
        obv: obv1h[obv1h.length - 1],
        mfi: mfi1h[mfi1h.length - 1],
        volatility: vol1h[vol1h.length - 1],
        harami: harami1h[harami1h.length - 1],
        marubozu: marubozu1h[marubozu1h.length - 1],
        uncertainty: uncertainty,
      };

      this.log(`New Hourly Prediction: ${(prediction || 0).toFixed(4)} (Velocity: ${(this.lastVelocity || 0).toFixed(4)})`);
    }

    const prediction = this.lastPrediction || 0;
    const uncertainty = this.lastUncertainty || 0;
    const velocity = this.lastVelocity || 0;

    this.lastParams = {
      ...this.lastParams,
      velocity: velocity,
      session: nowDate.getUTCHours() >= this.settings.asiaStart && nowDate.getUTCHours() < this.settings.asiaEnd ? 'Asia' : 
               nowDate.getUTCHours() >= this.settings.nyStart && nowDate.getUTCHours() < this.settings.nyEnd ? 'New York' : 'Off-Session',
      dayOfWeek: nowDate.getDay()
    };

    // 3. Trading Logic
    if (this.activeTrade) {
      let profitPct = 0;
      
      if (this.isRealTrading) {
        // Use the latest price from ticker and position data from WS/REST
        const entry = this.activeTrade.entryPrice;
        const size = this.activeTrade.size || 0;
        
        if ((this.activeTrade.type === 'CALL_SPREAD' || this.activeTrade.type === 'SHORT_CALL') && this.activeTrade.legs) {
          // Use real PNL from syncPositionWithRest if available
          if (this.activeTrade.pnl !== undefined) {
            // Estimate initial value for profit percentage
            const initialValueUSD = this.activeTrade.legs.reduce((acc, leg) => acc + (leg.entryPrice * Math.abs(leg.size) * 0.001), 0);
            profitPct = this.activeTrade.pnl / Math.max(1, Math.abs(initialValueUSD));
            this.activeTrade.pnlPct = profitPct * 100;
          } else {
            // Fallback to underlying price movement
            profitPct = (this.activeTrade.entryPrice - price) / this.activeTrade.entryPrice;
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
        } else if (prediction < this.settings.exitThreshold) {
          shouldExit = true;
          reason = 'PREDICTION';
        }
      } else {
        // LONG
        if (price <= stopLossPrice) {
          shouldExit = true;
          reason = 'STOP_LOSS';
        } else if (price >= takeProfitPrice) {
          shouldExit = true;
          reason = 'TAKE_PROFIT';
        } else if (currentTrailingStop && price <= currentTrailingStop) {
          shouldExit = true;
          reason = 'TRAILING_STOP';
        } else if (prediction > -this.settings.exitThreshold) {
          shouldExit = true;
          reason = 'PREDICTION';
        }
      }

      if (shouldExit) {
        this.log(`Closing Trade: ${reason} at $${price} | Prediction: ${(prediction || 0).toFixed(4)} | Profit: ${((profitPct || 0) * 100).toFixed(2)}%`, profitPct >= 0 ? 'success' : 'warning');
        
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

        if (this.isRealTrading) {
          if ((this.activeTrade.type === 'CALL_SPREAD' || this.activeTrade.type === 'SHORT_CALL') && this.activeTrade.legs) {
            for (const leg of this.activeTrade.legs) {
              const exitSide = leg.side === 'sell' ? 'buy' : 'sell';
              await this.placeOptionOrder(leg.symbol, exitSide, leg.size);
            }
          } else {
            await this.placeRealOrder(this.activeTrade.type === 'SHORT' ? 'buy' : 'sell', this.settings.quantity, true);
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
    } else if (prediction > this.settings.threshold) {
      // 1. Uncertainty Filter (MC Dropout)
      if (this.settings.mcPasses > 1 && uncertainty > this.settings.maxUncertainty) {
        if (this.tickerCount % 100 === 0) {
          this.log(`Signal detected but uncertainty too high: ${(uncertainty || 0).toFixed(4)} > ${this.settings.maxUncertainty}`, 'info');
        }
        return;
      }

      // 2. Signal Velocity Filter (Temporal Delta)
      if (velocity < this.settings.minSignalVelocity) {
        if (this.tickerCount % 100 === 0) {
          this.log(`Signal detected but velocity too low: ${(velocity || 0).toFixed(4)} < ${this.settings.minSignalVelocity}`, 'info');
        }
        return;
      }

      // Cooldown check: don't re-open within 5 minutes of closing a trade
      const cooldownMs = 10 * 1000;
      if (Date.now() - this.lastTradeCloseTime < cooldownMs) {
        if (this.tickerCount % 100 === 0) {
          this.log(`Signal detected but in cooldown period (${Math.ceil((cooldownMs - (Date.now() - this.lastTradeCloseTime)) / 1000)}s remaining)`, 'info');
        }
        return;
      }

      // Session check
      let canTrade = true;
      const hour = nowDate.getUTCHours();

      // Daily Limits check
      if (this.settings.dailyProfitLimit > 0 && this.dailyProfit >= this.settings.dailyProfitLimit) {
        if (this.tickerCount % 100 === 0) {
          this.log(`Daily profit limit reached: $${(this.dailyProfit || 0).toFixed(2)} >= $${this.settings.dailyProfitLimit}`, 'warning');
        }
        canTrade = false;
      }
      if (this.settings.dailyLossLimit > 0 && this.dailyProfit <= -this.settings.dailyLossLimit) {
        if (this.tickerCount % 100 === 0) {
          this.log(`Daily loss limit reached: $${(this.dailyProfit || 0).toFixed(2)} <= -$${this.settings.dailyLossLimit}`, 'warning');
        }
        canTrade = false;
      }

      if (this.settings.useSessionTrading && canTrade) {
        const isAsia = hour >= this.settings.asiaStart && hour < this.settings.asiaEnd;
        const isNY = hour >= this.settings.nyStart && hour < this.settings.nyEnd;
        canTrade = isAsia || isNY;
      }

      if (canTrade && !this.activeTrade && !this.isOpeningTrade) {
        this.isOpeningTrade = true;
        try {
          if (this.settings.strategyType === 'CALL_SPREAD') {
            await this.executeCallSpread(price, prediction, this.lastFeatures || {});
          } else if (this.settings.strategyType === 'SHORT_CALL') {
            await this.executeShortCall(price, prediction, this.lastFeatures || {});
          } else {
            this.log(`Opening SHORT at $${price} | Prediction: ${(prediction || 0).toFixed(4)}`, 'success');
            this.activeTrade = {
              type: 'SHORT',
              entryPrice: price,
              entryTime: Date.now(),
              highestProfitPct: 0,
              trailingStopPrice: null,
              prediction: prediction,
              features: this.lastFeatures || {}
            };
            if (this.isRealTrading) {
              this.placeRealOrder('sell', this.settings.quantity, false).then(orderId => {
                if (orderId && this.activeTrade) {
                  this.activeTrade.orderId = orderId;
                  this.saveState();
                  // Try to sync position immediately after market order
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

  private async syncPositionWithRest() {
    if (!this.isRealTrading || !this.activeTrade) return;
    try {
      const positions = await deltaRequest('GET', '/v2/positions?underlying_asset_symbol=BTC');
      if (positions.success && Array.isArray(positions.result)) {
        if ((this.activeTrade.type === 'CALL_SPREAD' || this.activeTrade.type === 'SHORT_CALL') && this.activeTrade.legs) {
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

  private async executeCallSpread(price: number, prediction: number, features: any) {
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

  private async executeShortCall(price: number, prediction: number, features: any) {
    if (!this.settings) return;
    this.log(`Executing Short Call strategy...`, 'info');

    const shortLeg = await this.findOptionByDelta(true, this.settings.shortCallDelta);

    if (!shortLeg) {
      this.log('Failed to find suitable option leg for short call', 'error');
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
}

export const tradingService = new TradingService();
