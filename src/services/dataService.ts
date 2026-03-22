import { format } from 'date-fns';
import { logger } from './loggerService';

export interface Candle {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

// In-memory cache for candles
// Key format: "interval:timestamp"
const candleCache = new Map<string, Candle>();

export async function fetchBTCData(limit: number = 500, interval: string = '1h', startTimeParam?: number, endTimeParam?: number): Promise<Candle[]> {
  const MAX_BATCH_SIZE = 500;
  let allCandles: Candle[] = [];
  
  // Delta Exchange resolution mapping
  const resolutionMap: { [key: string]: string } = {
    '5m': '5m',
    '1h': '1h',
    '4h': '4h',
    '1d': '1d',
  };

  const resolution = resolutionMap[interval] || '1h';
  const symbol = 'BTCUSD'; // Delta Exchange perpetual symbol
  
  // Calculate start time based on limit and interval if not provided
  const intervalInSeconds: { [key: string]: number } = {
    '5m': 300,
    '1h': 3600,
    '4h': 14400,
    '1d': 86400,
  };
  
  const secondsPerCandle = intervalInSeconds[interval] || 3600;
  let endTime = endTimeParam ? Math.floor(endTimeParam / 1000) : Math.floor(Date.now() / 1000);
  let startTime = startTimeParam ? Math.floor(startTimeParam / 1000) : endTime - (limit * secondsPerCandle);

  // Normalize to interval grid
  startTime = Math.floor(startTime / secondsPerCandle) * secondsPerCandle;
  endTime = Math.floor(endTime / secondsPerCandle) * secondsPerCandle;

  // Check if we have the ENTIRE range in cache
  const cachedResult: Candle[] = [];
  let isFullyCached = true;
  for (let t = startTime; t <= endTime; t += secondsPerCandle) {
    const cached = candleCache.get(`${interval}:${t * 1000}`);
    if (cached) {
      cachedResult.push(cached);
    } else {
      isFullyCached = false;
      break;
    }
  }

  if (isFullyCached && cachedResult.length > 0) {
    logger.info(`Cache hit! Returning ${cachedResult.length} candles from memory for ${interval}.`);
    return cachedResult.sort((a, b) => a.time - b.time);
  }

  const totalSeconds = endTime - startTime;
  const estimatedCandles = Math.ceil(totalSeconds / secondsPerCandle);
  let remainingLimit = startTimeParam && endTimeParam ? estimatedCandles : limit;

  logger.info(`Fetching candles (${interval}) from Delta Exchange for ${symbol} from ${format(new Date(startTime * 1000), 'yyyy-MM-dd')} to ${format(new Date(endTime * 1000), 'yyyy-MM-dd')}...`);

  try {
    let currentEndTime = endTime;
    while (remainingLimit > 0 && currentEndTime > startTime) {
      const currentBatchSize = Math.min(remainingLimit, MAX_BATCH_SIZE);
      let batchStartTime = currentEndTime - (currentBatchSize * secondsPerCandle);
      if (batchStartTime < startTime) batchStartTime = startTime;
      
      const url = new URL('https://api.india.delta.exchange/v2/history/candles');
      url.searchParams.append('symbol', symbol);
      url.searchParams.append('resolution', resolution);
      url.searchParams.append('start', batchStartTime.toString());
      url.searchParams.append('end', currentEndTime.toString());

      logger.info(`Requesting batch: ${format(new Date(batchStartTime * 1000), 'yyyy-MM-dd HH:mm')} to ${format(new Date(currentEndTime * 1000), 'yyyy-MM-dd HH:mm')}`);

      const response = await fetch(url.toString());
      
      if (!response.ok) {
        const errorText = await response.text();
        logger.error(`Delta API Error (${response.status}): ${errorText}`);
        throw new Error(`Delta API Error: ${response.status}`);
      }
      
      const result = await response.json();
      const data = result.result || [];
      
      if (!data || data.length === 0) {
        logger.warning(`No data returned for batch. Stopping fetch.`);
        break;
      }

      logger.success(`Received ${data.length} candles.`);

      const batch: Candle[] = data.map((d: any) => {
        const candle = {
          time: d.time * 1000,
          open: parseFloat(d.open),
          high: parseFloat(d.high),
          low: parseFloat(d.low),
          close: parseFloat(d.close),
          volume: parseFloat(d.volume),
        };
        // Cache the fetched data
        candleCache.set(`${interval}:${candle.time}`, candle);
        return candle;
      });

      allCandles = [...batch, ...allCandles];
      remainingLimit -= batch.length;
      currentEndTime = Math.floor(batch[0].time / 1000) - secondsPerCandle;

      if (batch.length < currentBatchSize / 2 && !startTimeParam) break; 
      if (currentEndTime <= startTime) break;
    }
    
    // Sort by time ascending
    allCandles.sort((a, b) => a.time - b.time);
    
    // Remove duplicates
    const uniqueCandles = allCandles.filter((candle, index, self) =>
      index === self.findIndex((c) => c.time === candle.time)
    );

    logger.success(`Total unique candles fetched and cached: ${uniqueCandles.length}`);
    return uniqueCandles;
  } catch (error) {
    logger.error(`Error fetching BTC data (${interval}): ${error instanceof Error ? error.message : String(error)}`);
    // If fetch fails, return whatever we have in cache for this range
    const fallback: Candle[] = [];
    for (let t = startTime; t <= endTime; t += secondsPerCandle) {
      const cached = candleCache.get(`${interval}:${t * 1000}`);
      if (cached) fallback.push(cached);
    }
    return fallback.sort((a, b) => a.time - b.time);
  }
}
