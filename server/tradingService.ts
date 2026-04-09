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
  strategyType: 'SHORT_BTC' | 'CALL_SPREAD';
  shortCallDelta: number;
  longCallDelta: number;
  dailyProfitLimit: number;
  dailyLossLimit: number;
}

interface ActiveTrade {
  type: 'LONG' | 'SHORT' | 'CALL_SPREAD';
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
  }[];
}

interface ClosedTrade {
  type: 'LONG' | 'SHORT' | 'CALL_SPREAD';
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
  private lastPrice: number | null = null;
  private lastFeatures: Record<string, number> | null = null;
  private lastParams: any = null;
  private lastUpdate: Date | null = null;
  private logs: LogEntry[] = [];
  private tickerCount: number = 0;
  private lastProcessTime: number = 0;
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
        if (this.activeTrade.type === 'CALL_SPREAD' && this.activeTrade.legs) {
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
      if (nowDate.getTime() - lastCandleTime >= 60 * 60 * 1000) {
        this.log('New hour detected, refreshing candles...');
        await this.fetchInitialData();
      }
    }

    // 1. Calculate 4h prediction
    const prices4h = this.candles4h.map(c => c.close);
    const highs4h = this.candles4h.map(c => c.high);
    const lows4h = this.candles4h.map(c => c.low);
    const volumes4h = this.candles4h.map(c => c.volume);
    const hours4h = this.candles4h.map(c => new Date(c.time * 1000).getHours());
    const days4h = this.candles4h.map(c => new Date(c.time * 1000).getDay());

    const rsi4h = indicators.calculateRSI(prices4h);
    const ema4h = indicators.calculateEMA(prices4h, 20);
    const ema9_4h = indicators.calculateEMA(prices4h, 9);
    const bb4h = indicators.calculateBollingerBands(prices4h);
    const macd4h = indicators.calculateMACD(prices4h);
    const stochRsi4h = indicators.calculateStochasticRSI(rsi4h);
    const atr4h = indicators.calculateATR(highs4h, lows4h, prices4h);
    const cross4h = indicators.calculateEMACross(ema9_4h, ema4h);
    const obv4h = indicators.calculateOBV(prices4h, volumes4h);
    const mfi4h = indicators.calculateMFI(highs4h, lows4h, prices4h, volumes4h);
    const vol4h = indicators.calculateVolatility(prices4h);
    const opens4h = this.candles4h.map(c => c.open);
    const harami4h = indicators.calculateBearishHarami(opens4h, prices4h);
    const marubozu4h = indicators.calculateMarubozu(opens4h, highs4h, lows4h, prices4h);
    const engulfing4h = indicators.calculateEngulfing(opens4h, prices4h);

    const normalize = (arr: number[]) => {
      const min = Math.min(...arr);
      const max = Math.max(...arr);
      return arr.map(v => (max === min ? 0 : (v - min) / (max - min)));
    };

    const windowSize = 20;
    const last4hIdx = prices4h.length - 1;
    if (last4hIdx < windowSize) {
      if (this.tickerCount % 100 === 0) {
        this.log(`Insufficient 4h data: ${prices4h.length} candles (need ${windowSize})`, 'warning');
      }
      return;
    }
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
    let secondaryPrediction = 0;
    try {
      secondaryPrediction = this.model4h.predict(x4h);
    } catch (err: any) {
      this.log(`4h Prediction failed: ${err.message}`, 'error');
    }

    // 2. Calculate 1h prediction
    const fullPrices1h = [...this.candles.map(c => c.close), price];
    const fullHighs1h = [...this.candles.map(c => c.high), price];
    const fullLows1h = [...this.candles.map(c => c.low), price];
    const fullVolumes1h = [...this.candles.map(c => c.volume), 0];
    const fullHours1h = [...this.candles.map(c => new Date(c.time * 1000).getHours()), nowDate.getHours()];
    const fullDays1h = [...this.candles.map(c => new Date(c.time * 1000).getDay()), nowDate.getDay()];

    const rsi1h = indicators.calculateRSI(fullPrices1h);
    const ema1h = indicators.calculateEMA(fullPrices1h, 20);
    const ema9_1h = indicators.calculateEMA(fullPrices1h, 9);
    const bb1h = indicators.calculateBollingerBands(fullPrices1h);
    const macd1h = indicators.calculateMACD(fullPrices1h);
    const stochRsi1h = indicators.calculateStochasticRSI(rsi1h);
    const atr1h = indicators.calculateATR(fullHighs1h, fullLows1h, fullPrices1h);
    const cross1h = indicators.calculateEMACross(ema9_1h, ema1h);
    const obv1h = indicators.calculateOBV(fullPrices1h, fullVolumes1h);
    const mfi1h = indicators.calculateMFI(fullHighs1h, fullLows1h, fullPrices1h, fullVolumes1h);
    const vol1h = indicators.calculateVolatility(fullPrices1h);
    const opens1h = [...this.candles.map(c => c.open), price];
    const harami1h = indicators.calculateBearishHarami(opens1h, fullPrices1h);
    const marubozu1h = indicators.calculateMarubozu(opens1h, fullHighs1h, fullLows1h, fullPrices1h);
    const engulfing1h = indicators.calculateEngulfing(opens1h, fullPrices1h);

    const last1hIdx = fullPrices1h.length - 1;
    if (last1hIdx < windowSize) {
      if (this.tickerCount % 100 === 0) {
        this.log(`Insufficient 1h data: ${fullPrices1h.length} candles (need ${windowSize})`, 'warning');
      }
      return;
    }
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
    
    // Calculate Signal Velocity (Temporal Delta)
    const velocity = this.previousPrediction !== null ? (prediction - this.previousPrediction) : 0;
    
    this.previousPrediction = this.lastPrediction;
    this.lastPrediction = prediction;
    this.lastFeatures = currentFeatures;
    this.lastParams = {
      ...this.lastParams, // Keep existing if any
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
      uncertainty: uncertainty,
      velocity: velocity,
      engulfing: engulfing1h[engulfing1h.length - 1],
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
        
        if (this.activeTrade.type === 'CALL_SPREAD' && this.activeTrade.legs) {
          // For options spread, we'd need to fetch current mark prices of legs
          // For now, we'll use a simplified underlying-based profit or just 0 until we fetch leg prices
          // Let's try to fetch mark prices for legs if it's been a while
          profitPct = (this.activeTrade.entryPrice - price) / this.activeTrade.entryPrice;
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
          if (this.activeTrade.type === 'SHORT') {
            profitPct = (this.activeTrade.entryPrice - price) / this.activeTrade.entryPrice;
          } else {
            profitPct = (price - this.activeTrade.entryPrice) / this.activeTrade.entryPrice;
          }
        }
      } else {
        // Paper trading logic
        if (this.activeTrade.type === 'SHORT') {
          profitPct = (this.activeTrade.entryPrice - price) / this.activeTrade.entryPrice;
        } else {
          profitPct = (price - this.activeTrade.entryPrice) / this.activeTrade.entryPrice;
        }
      }

      const stopLossPrice = (this.activeTrade.type === 'SHORT' || this.activeTrade.type === 'CALL_SPREAD')
        ? this.activeTrade.entryPrice * (1 + this.settings.stopLoss)
        : this.activeTrade.entryPrice * (1 - this.settings.stopLoss);
      const takeProfitPrice = (this.activeTrade.type === 'SHORT' || this.activeTrade.type === 'CALL_SPREAD')
        ? this.activeTrade.entryPrice * (1 - this.settings.takeProfit)
        : this.activeTrade.entryPrice * (1 + this.settings.takeProfit);

      let currentTrailingStop = this.activeTrade.trailingStopPrice;
      let highestProfit = this.activeTrade.highestProfitPct;

      if (profitPct > highestProfit) {
        highestProfit = profitPct;
        this.activeTrade.highestProfitPct = highestProfit;
        if (profitPct >= this.settings.trailingStopActivation) {
          if (this.activeTrade.type === 'SHORT' || this.activeTrade.type === 'CALL_SPREAD') {
            currentTrailingStop = price * (1 + this.settings.trailingStopOffset);
          } else {
            currentTrailingStop = price * (1 - this.settings.trailingStopOffset);
          }
          this.activeTrade.trailingStopPrice = currentTrailingStop;
        }
      }

      let shouldExit = false;
      let reason = '';

      if (this.activeTrade.type === 'SHORT' || this.activeTrade.type === 'CALL_SPREAD') {
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
        this.log(`Closing Trade: ${reason} at $${price} | Prediction: ${prediction.toFixed(4)} | Profit: ${(profitPct * 100).toFixed(2)}%`, profitPct >= 0 ? 'success' : 'warning');
        
        const btcQuantity = this.settings.quantityType === 'LOTS' ? this.settings.quantity * 0.001 : this.settings.quantity;
        const profit = this.settings.quantityType === 'USD'
          ? this.settings.quantity * profitPct
          : btcQuantity * (this.activeTrade.type === 'SHORT' ? (this.activeTrade.entryPrice - price) : (price - this.activeTrade.entryPrice));

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
          if (this.activeTrade.type === 'CALL_SPREAD' && this.activeTrade.legs) {
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
          this.log(`Signal detected but uncertainty too high: ${uncertainty.toFixed(4)} > ${this.settings.maxUncertainty}`, 'info');
        }
        return;
      }

      // 2. Signal Velocity Filter (Temporal Delta)
      if (velocity < this.settings.minSignalVelocity) {
        if (this.tickerCount % 100 === 0) {
          this.log(`Signal detected but velocity too low: ${velocity.toFixed(4)} < ${this.settings.minSignalVelocity}`, 'info');
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
          this.log(`Daily profit limit reached: $${this.dailyProfit.toFixed(2)} >= $${this.settings.dailyProfitLimit}`, 'warning');
        }
        canTrade = false;
      }
      if (this.settings.dailyLossLimit > 0 && this.dailyProfit <= -this.settings.dailyLossLimit) {
        if (this.tickerCount % 100 === 0) {
          this.log(`Daily loss limit reached: $${this.dailyProfit.toFixed(2)} <= -$${this.settings.dailyLossLimit}`, 'warning');
        }
        canTrade = false;
      }

      if (this.settings.useSessionTrading && canTrade) {
        const isAsia = hour >= this.settings.asiaStart && hour < this.settings.asiaEnd;
        const isNY = hour >= this.settings.nyStart && hour < this.settings.nyEnd;
        canTrade = isAsia || isNY;
      }

      if (canTrade) {
        if (this.settings.strategyType === 'CALL_SPREAD') {
          await this.executeCallSpread(price, prediction, currentFeatures);
        } else {
          this.log(`Opening SHORT at $${price} | Prediction: ${prediction.toFixed(4)}`, 'success');
          this.activeTrade = {
            type: 'SHORT',
            entryPrice: price,
            entryTime: Date.now(),
            highestProfitPct: 0,
            trailingStopPrice: null,
            prediction: prediction,
            features: currentFeatures
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
      }
    }
  }

  private async syncPositionWithRest() {
    if (!this.isRealTrading || !this.activeTrade) return;
    try {
      const positions = await deltaRequest('GET', '/v2/positions?underlying_asset_symbol=BTC');
      if (positions.success && Array.isArray(positions.result)) {
        const btcPos = positions.result.find((p: any) => p.symbol === 'BTCUSD' || p.symbol === 'BTCUSD_P');
        if (btcPos) {
          const entry = parseFloat(btcPos.avg_entry_price);
          const size = parseFloat(btcPos.size);
          if (size !== 0 && entry > 0) {
            this.activeTrade.entryPrice = entry;
            this.activeTrade.size = size;
            this.saveState();
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
        this.log(`Found best option: ${bestOption.symbol} with delta ${bestOption.delta.toFixed(3)}`, 'success');
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

      // For options, quantity is usually in units of underlying (BTC)
      // If quantityType is LOTS (0.001 BTC), we convert
      let size = quantity;
      if (this.settings?.quantityType === 'LOTS') size = quantity * 0.001;
      else if (this.settings?.quantityType === 'USD' && this.lastPrice) {
        size = quantity / this.lastPrice;
      }
      
      // Delta options usually have a minimum size (e.g. 0.01 BTC)
      // We'll just use the calculated size
      const finalSize = size;

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
}

export const tradingService = new TradingService();
