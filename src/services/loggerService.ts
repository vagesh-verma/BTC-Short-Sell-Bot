type LogType = 'info' | 'error' | 'success' | 'warning';

interface LogEntry {
  timestamp: number;
  message: string;
  type: LogType;
}

class Logger {
  private logs: LogEntry[] = [];
  private listeners: ((logs: LogEntry[]) => void)[] = [];

  log(message: string, type: LogType = 'info') {
    const entry: LogEntry = {
      timestamp: Date.now(),
      message,
      type,
    };
    this.logs.push(entry);
    // Keep only last 100 logs
    if (this.logs.length > 100) {
      this.logs.shift();
    }
    this.notify();
    console.log(`[${type.toUpperCase()}] ${message}`);
  }

  info(message: string) { this.log(message, 'info'); }
  error(message: string) { this.log(message, 'error'); }
  success(message: string) { this.log(message, 'success'); }
  warning(message: string) { this.log(message, 'warning'); }

  subscribe(listener: (logs: LogEntry[]) => void) {
    this.listeners.push(listener);
    listener(this.logs);
    return () => {
      this.listeners = this.listeners.filter(l => l !== listener);
    };
  }

  private notify() {
    this.listeners.forEach(l => l(this.logs));
  }

  getLogs() {
    return this.logs;
  }

  clear() {
    this.logs = [];
    this.notify();
  }
}

export const logger = new Logger();
