import WebSocket from 'ws';
import { GRUModel } from './modelService';
import * as indicators from './indicatorService';
import fetch from 'node-fetch';
import crypto from 'crypto';
import { deltaRequest } from './deltaApi';

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
}

interface ActiveTrade {
  entryPrice: number;
  entryTime: number;
  highestProfitPct: number;
  trailingStopPrice: number | null;
  prediction: number;
  pnl?: number;
  pnlPct?: number;
  size?: number;
  features?: Record<string, number>;
}

interface ClosedTrade {
  type: 'SHORT';
  entryPrice: number;
  exitPrice: number;
  entryTime: number;
  exitTime: number;
  profit: number;
  profitPct: number;
  exitReason: string;
  prediction: number;
  features: Record<string, number>;
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
  private lastPrice: number | null = null;
  private lastFeatures: Record<string, number> | null = null;
  private lastParams: any = null;
  private lastUpdate: Date | null = null;
  private logs: LogEntry[] = [];
  private tickerCount: number = 0;

  private apiKey: string = process.env.DELTA_API_KEY || '';
  private apiSecret: string = process.env.DELTA_API_SECRET || '';

  constructor() {
    this.log('Trading Service Initialized');
  }

  private log(message: string, type: LogEntry['type'] = 'info') {
    const timestamp = Date.now();
    const entry: LogEntry = { timestamp, type, message };
    console.log(`[${new Date(timestamp).toISOString()}] [${type.toUpperCase()}] ${message}`);
    this.logs.unshift(entry);
    if (this.logs.length > 100) this.logs.pop();
  }

  public async start(settings: TradingSettings, isReal: boolean) {
    if (this.isRunning) {
      this.settings = settings;
      this.isRealTrading = isReal;
      this.log('Settings updated while running', 'info');
      return;
    }
    this.settings = settings;
    this.isRealTrading = isReal;
    this.isRunning = true;
    this.log(`Starting Live Trading (Real: ${isReal})`, 'success');
    
    await this.fetchInitialData();
    this.connectWebSocket();
  }

  public updateSettings(settings: TradingSettings) {
    this.settings = settings;
    this.log('Trading settings synchronized', 'info');
  }

  public stop() {
    this.isRunning = false;
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
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
      hasModels: !!(this.model1h && this.model4h)
    };
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
      this.ws?.send(JSON.stringify({
        type: 'subscribe',
        payload: {
          channels: [{ name: 'v2/ticker', symbols: ['BTCUSD', 'BTCUSD_P'] }]
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
        } else if (msg.type === 'subscriptions') {
          this.log('WebSocket Subscriptions Confirmed', 'success');
        } else if (msg.type === 'error') {
          this.log(`WebSocket Server Error: ${msg.message}`, 'error');
        } else if (msg.type === 'heartbeat') {
          // Ignore heartbeat
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
    if (this.lastPrice === null) {
      this.log(`First price received: $${price}`, 'success');
    }
    this.lastPrice = price;
    this.lastUpdate = new Date();

    if (!this.model1h || !this.model4h || !this.settings) return;

    // Check if we need to update candles (every hour)
    const now = new Date();
    if (this.candles.length > 0) {
      const lastCandleTime = this.candles[this.candles.length - 1].time * 1000;
      if (now.getTime() - lastCandleTime >= 60 * 60 * 1000) {
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
      this.log(`Insufficient 4h data: ${prices4h.length} candles`, 'warning');
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
    const fullHours1h = [...this.candles.map(c => new Date(c.time * 1000).getHours()), now.getHours()];
    const fullDays1h = [...this.candles.map(c => new Date(c.time * 1000).getDay()), now.getDay()];

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
      this.log(`Insufficient 1h data: ${fullPrices1h.length} candles`, 'warning');
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
    try {
      prediction = this.model1h.predict(x1h);
    } catch (err: any) {
      this.log(`1h Prediction failed: ${err.message}`, 'error');
    }
    this.lastPrediction = prediction;
    this.lastFeatures = currentFeatures;
    this.lastParams = {
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
      session: now.getUTCHours() >= this.settings.asiaStart && now.getUTCHours() < this.settings.asiaEnd ? 'Asia' : 
               now.getUTCHours() >= this.settings.nyStart && now.getUTCHours() < this.settings.nyEnd ? 'New York' : 'Off-Session',
      dayOfWeek: now.getDay()
    };

    // 3. Trading Logic
    if (this.activeTrade) {
      let profitPct = (this.activeTrade.entryPrice - price) / this.activeTrade.entryPrice;
      
      // If real trading, try to get actual PNL from exchange
      if (this.isRealTrading) {
        try {
          const positions = await deltaRequest('GET', '/v2/positions');
          if (positions.success && Array.isArray(positions.result)) {
            const btcPos = positions.result.find((p: any) => p.symbol === 'BTCUSD' || p.symbol === 'BTCUSD_P');
            if (btcPos) {
              // Delta returns PNL in quote currency or base depending on contract
              // For inverse contracts it's different. 
              // We'll use their reported unrealized_pnl and entry_price
              this.activeTrade.pnl = parseFloat(btcPos.unrealized_pnl);
              this.activeTrade.size = parseFloat(btcPos.size);
              // Calculate PNL % based on their margin or entry
              // For simplicity, we'll stick to our calculation for trailing stop but show their PNL
              this.activeTrade.pnlPct = (this.activeTrade.pnl / (Math.abs(this.activeTrade.size) * this.activeTrade.entryPrice)) * 100;
            }
          }
        } catch (err) {
          this.log(`Failed to fetch exchange position: ${err}`, 'error');
        }
      }

      const stopLossPrice = this.activeTrade.entryPrice * (1 + this.settings.stopLoss);
      const takeProfitPrice = this.activeTrade.entryPrice * (1 - this.settings.takeProfit);

      let currentTrailingStop = this.activeTrade.trailingStopPrice;
      let highestProfit = this.activeTrade.highestProfitPct;

      if (profitPct > highestProfit) {
        highestProfit = profitPct;
        this.activeTrade.highestProfitPct = highestProfit;
        if (profitPct >= this.settings.trailingStopActivation) {
          currentTrailingStop = price * (1 + this.settings.trailingStopOffset);
          this.activeTrade.trailingStopPrice = currentTrailingStop;
        }
      }

      let shouldExit = false;
      let reason = '';

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

      if (shouldExit) {
        this.log(`Closing Trade: ${reason} at $${price} | Profit: ${(profitPct * 100).toFixed(2)}%`, profitPct >= 0 ? 'success' : 'error');
        
        const btcQuantity = this.settings.quantityType === 'LOTS' ? this.settings.quantity * 0.001 : this.settings.quantity;
        const profit = this.settings.quantityType === 'USD'
          ? this.settings.quantity * profitPct
          : btcQuantity * (this.activeTrade.entryPrice - price);

        const closedTrade: ClosedTrade = {
          type: 'SHORT',
          entryPrice: this.activeTrade.entryPrice,
          exitPrice: price,
          entryTime: this.activeTrade.entryTime,
          exitTime: Date.now(),
          profit,
          profitPct: profitPct * 100,
          exitReason: reason,
          prediction: this.activeTrade.prediction,
          features: this.activeTrade.features || {}
        };
        this.closedTrades.unshift(closedTrade);
        if (this.closedTrades.length > 100) this.closedTrades.pop();

        if (this.isRealTrading) {
          await this.placeRealOrder('buy', this.settings.quantity);
        }
        this.activeTrade = null;
      } else {
        this.activeTrade.highestProfitPct = highestProfit;
        this.activeTrade.trailingStopPrice = currentTrailingStop;
      }
    } else if (prediction > this.settings.threshold) {
      // Session check
      let canTrade = true;
      const hour = now.getUTCHours();
      if (this.settings.useSessionTrading) {
        const isAsia = hour >= this.settings.asiaStart && hour < this.settings.asiaEnd;
        const isNY = hour >= this.settings.nyStart && hour < this.settings.nyEnd;
        canTrade = isAsia || isNY;
      }

      if (canTrade) {
        this.log(`Opening SHORT at $${price} | Prediction: ${prediction.toFixed(4)}`, 'success');
        this.activeTrade = {
          entryPrice: price,
          entryTime: Date.now(),
          highestProfitPct: 0,
          trailingStopPrice: null,
          prediction: prediction,
          features: currentFeatures
        };
        if (this.isRealTrading) {
          await this.placeRealOrder('sell', this.settings.quantity);
        }
      }
    }
  }

  private async placeRealOrder(side: 'buy' | 'sell', quantity: number) {
    try {
      if (!this.apiKey || !this.apiSecret) {
        this.log('API Keys missing, cannot place real order');
        return;
      }

      let lots = quantity;
      if (this.settings?.quantityType === 'BTC') lots = quantity * 1000;
      else if (this.settings?.quantityType === 'USD' && this.lastPrice) {
        lots = (quantity / this.lastPrice) / 0.001;
      }
      const finalSize = Math.max(1, Math.floor(lots));

      const timestamp = Math.floor(Date.now() / 1000);
      const method = 'POST';
      const path = '/v2/orders';
      const body = JSON.stringify({
        symbol: 'BTCUSD',
        side,
        order_type: 'limit_order',
        size: finalSize,
        limit_price: this.lastPrice
      });

      const signatureData = method + timestamp + path + body;
      const signature = crypto.createHmac('sha256', this.apiSecret).update(signatureData).digest('hex');

      const response = await fetch('https://api.delta.exchange' + path, {
        method,
        headers: {
          'Content-Type': 'application/json',
          'api-key': this.apiKey,
          'signature': signature,
          'timestamp': timestamp.toString()
        },
        body
      });

      const data = await response.json() as any;
      if (data.result) {
        this.log(`Real order successful: ${side} ${finalSize} lots`, 'success');
      } else {
        this.log(`Real order failed: ${data.error || 'Unknown error'}`, 'error');
      }
    } catch (err) {
      this.log(`Error placing real order: ${err}`, 'error');
    }
  }
}

export const tradingService = new TradingService();
