import { logger } from './loggerService';

export interface TickerUpdate {
  symbol: string;
  price: number;
  timestamp: number;
}

export class DeltaSocketService {
  private socket: WebSocket | null = null;
  private url = 'wss://socket.india.delta.exchange';
  private onUpdateCallback: (update: TickerUpdate) => void;
  private symbol: string;
  private reconnectTimeout: any = null;
  private pingInterval: any = null;
  private isIntentionallyClosed = false;
  private lastMessageTime = Date.now();
  private reconnectAttempts = 0;
  private isConnecting = false;

  constructor(symbol: string, onUpdate: (update: TickerUpdate) => void) {
    this.symbol = symbol;
    this.onUpdateCallback = onUpdate;
  }

  async connect() {
    if (this.isConnecting) return;
    this.isConnecting = true;
    this.isIntentionallyClosed = false;
    clearTimeout(this.reconnectTimeout);
    
    try {
      // 1. Fetch Auth Payload from backend to keep API Secret secure
      const authRes = await fetch('/api/auth/websocket');
      if (!authRes.ok) throw new Error('Failed to get WebSocket auth payload');
      const authData = await authRes.json();

      if (this.socket) {
        this.socket.onopen = null;
        this.socket.onmessage = null;
        this.socket.onerror = null;
        this.socket.onclose = null;
        this.socket.close();
      }

      this.socket = new WebSocket(this.url);
      this.lastMessageTime = Date.now();

      this.socket.onopen = () => {
        this.isConnecting = false;
        logger.info(`WebSocket Connection Opened for ${this.symbol}`);
        this.reconnectAttempts = 0;
        
        // 2. Send Authentication Payload
        const authPayload = {
          type: 'auth',
          payload: {
            'api-key': authData.api_key,
            signature: authData.signature,
            timestamp: authData.timestamp
          }
        };
        this.socket?.send(JSON.stringify(authPayload));
        this.startHeartbeat();
      };

      this.socket.onmessage = (event) => {
        this.lastMessageTime = Date.now();
        try {
          const data = JSON.parse(event.data);
          
          // Handle successful authentication
          if (data.type === 'success' && data.message === 'Authenticated') {
            logger.info('WebSocket Authentication Successful. Subscribing...');
            this.subscribe();
            return;
          }

          // Handle heartbeat/ping from server
          if (data.type === 'ping' || data.op === 'ping') {
            this.socket?.send(JSON.stringify({ type: 'pong' }));
            return;
          }

          // Handle pong from server (response to our ping)
          if (data.type === 'pong' || data.op === 'pong') {
            return;
          }

          // Handle subscription confirmation or errors
          if (data.type === 'subscriptions' || data.type === 'subscribed') {
            logger.info('WebSocket Subscriptions confirmed');
            return;
          }

          if (data.type === 'error') {
            logger.error(`WebSocket Server Error: ${data.message || JSON.stringify(data)}`);
            return;
          }

          // Delta ticker format handling - following Python example fields
          const isTicker = data.type === 'v2/ticker' || data.type === 'ticker';
          if (isTicker && data.symbol === this.symbol) {
            // Python example uses 'close' for last price and 'mark_price'
            const price = parseFloat(data.close || data.last_price || data.mark_price || data.price);
            if (!isNaN(price)) {
              this.onUpdateCallback({
                symbol: data.symbol,
                price: price,
                timestamp: Date.now()
              });
            }
          }
        } catch (e) {
          // Ignore parse errors for non-JSON messages
        }
      };

      this.socket.onerror = (error) => {
        this.isConnecting = false;
        console.error('WebSocket Error:', error);
      };

      this.socket.onclose = (event) => {
        this.isConnecting = false;
        this.stopHeartbeat();
        if (!this.isIntentionallyClosed) {
          this.reconnectAttempts++;
          const delay = Math.min(30000, 2000 * Math.pow(1.5, this.reconnectAttempts - 1));
          logger.warning(`WebSocket Disconnected (Code: ${event.code}, Reason: ${event.reason || 'None'}). Reconnecting in ${(delay / 1000).toFixed(1)}s...`);
          clearTimeout(this.reconnectTimeout);
          this.reconnectTimeout = setTimeout(() => this.connect(), delay);
        }
      };
    } catch (err) {
      logger.error(`Failed to connect to WebSocket: ${err}`);
    }
  }

  private startHeartbeat() {
    this.stopHeartbeat();
    this.pingInterval = setInterval(() => {
      if (this.socket && this.socket.readyState === WebSocket.OPEN) {
        // Check for inactivity
        const now = Date.now();
        if (now - this.lastMessageTime > 60000) { // 60 seconds of no messages
          logger.warning('WebSocket Inactivity detected. Reconnecting...');
          this.isConnecting = false; // Allow reconnect
          this.connect();
          return;
        }
        // Delta Exchange expects a pong if they send a ping, 
        // but client-initiated pings can also help keep the connection alive.
        this.socket.send(JSON.stringify({ type: 'ping' }));
      }
    }, 15000); // Reduced to 15 seconds for better stability
  }

  private stopHeartbeat() {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

  private subscribe() {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      const msg = {
        type: 'subscribe',
        op: 'subscribe', // Some Delta API versions use 'op' instead of 'type'
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
    this.isIntentionallyClosed = true;
    this.stopHeartbeat();
    clearTimeout(this.reconnectTimeout);
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
  }
}
