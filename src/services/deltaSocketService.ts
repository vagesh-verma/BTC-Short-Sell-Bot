import { logger } from './loggerService';

export interface TickerUpdate {
  symbol: string;
  price: number;
  timestamp: number;
}

export class DeltaSocketService {
  private socket: WebSocket | null = null;
  private url = 'wss://api.india.delta.exchange/v2/l2_updates'; // Delta Exchange WS endpoint
  private onUpdateCallback: (update: TickerUpdate) => void;
  private symbol: string;

  constructor(symbol: string, onUpdate: (update: TickerUpdate) => void) {
    this.symbol = symbol;
    this.onUpdateCallback = onUpdate;
  }

  connect() {
    try {
      this.socket = new WebSocket(this.url);

      this.socket.onopen = () => {
        logger.info(`WebSocket Connected to Delta Exchange for ${this.symbol}`);
        this.subscribe();
      };

      this.socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        // Delta ticker format handling
        if (data.type === 'v2/ticker' && data.symbol === this.symbol) {
          this.onUpdateCallback({
            symbol: data.symbol,
            price: parseFloat(data.last_price),
            timestamp: Date.now()
          });
        }
      };

      this.socket.onerror = (error) => {
        logger.error(`WebSocket Error: ${error}`);
      };

      this.socket.onclose = () => {
        logger.warning('WebSocket Disconnected. Reconnecting in 5s...');
        setTimeout(() => this.connect(), 5000);
      };
    } catch (err) {
      logger.error(`Failed to connect to WebSocket: ${err}`);
    }
  }

  private subscribe() {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      const msg = {
        type: 'subscribe',
        payload: {
          channels: [
            {
              name: 'v2/ticker',
              symbols: [this.symbol]
            }
          ]
        }
      };
      this.socket.send(JSON.stringify(msg));
    }
  }

  disconnect() {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
  }
}
