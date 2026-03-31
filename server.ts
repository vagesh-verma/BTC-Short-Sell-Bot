import express from "express";
import { createServer as createViteServer } from "vite";
import path from "path";
import crypto from "crypto";
import fetch from "node-fetch";
import dotenv from "dotenv";
import { deltaRequest, generateSignature } from "./server/deltaApi";
import { tradingService } from "./server/tradingService";
import { GRUModel } from "./server/modelService";

dotenv.config();

const app = express();
const PORT = process.env.PORT ? Number(process.env.PORT) : 3000;

app.use(express.json({ limit: '50mb' }));

// Trading API Endpoints
app.post("/api/trading/start", async (req, res) => {
  const { settings, isRealTrading } = req.body;
  await tradingService.start(settings, isRealTrading);
  res.json({ success: true });
});

app.post("/api/trading/stop", (req, res) => {
  tradingService.stop();
  res.json({ success: true });
});

app.post("/api/trading/settings", (req, res) => {
  const { settings } = req.body;
  tradingService.updateSettings(settings);
  res.json({ success: true });
});

app.post("/api/trading/mode", (req, res) => {
  const { isRealTrading } = req.body;
  tradingService.setTradingMode(isRealTrading);
  res.json({ success: true });
});

app.get("/api/trading/status", (req, res) => {
  res.json(tradingService.getStatus());
});

app.post("/api/trading/sync-model", async (req, res) => {
  try {
    const { model1hArtifacts, model4hArtifacts, metadata1h, metadata4h } = req.body;
    const model1h = await GRUModel.loadFromArtifacts(model1hArtifacts, metadata1h);
    const model4h = await GRUModel.loadFromArtifacts(model4hArtifacts, metadata4h);
    tradingService.setModels(model1h, model4h);
    res.json({ success: true });
  } catch (err: any) {
    console.error("Model sync failed:", err);
    res.status(500).json({ error: err.message });
  }
});

const DELTA_BASE_URL = "https://api.india.delta.exchange";
let productMap: Record<string, number> = { 'BTCUSD': 1 };

async function fetchProducts() {
  try {
    const result = await deltaRequest("GET", "/v2/products") as any;
    if (result.success && Array.isArray(result.result)) {
      result.result.forEach((p: any) => {
        productMap[p.symbol] = p.id;
      });
      console.log(`[Delta] Loaded ${result.result.length} products.`);
    }
  } catch (error: any) {
    console.warn("[Delta] Could not fetch products:", error.message);
  }
}

function getCredentials() {
  const apiKey = process.env.DELTA_API_KEY?.trim();
  const apiSecret = process.env.DELTA_API_SECRET?.trim();
  if (!apiKey || !apiSecret) {
    throw new Error("Delta API credentials missing. Please set DELTA_API_KEY and DELTA_API_SECRET in Settings.");
  }
  return { apiKey, apiSecret };
}

// API Routes
app.get("/api/health", (req, res) => {
  res.json({ status: "ok" });
});

app.get("/api/wallet", async (req, res) => {
  try {
    const result = await deltaRequest("GET", "/v2/wallet/balances");
    res.json(result);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/order", async (req, res) => {
  try {
    const { symbol, side, order_type, size, limit_price } = req.body;
    
    const productId = productMap[symbol] || 1;
    
    const orderData = {
      product_id: productId,
      side,
      order_type,
      size: Math.floor(size),
    };

    if (order_type === 'limit_order' && limit_price) {
      (orderData as any).limit_price = limit_price.toString();
    }

    // Note: Delta Exchange uses product_id. For BTCUSD it's often 1.
    // We might need to map symbol to product_id.
    
    const result = await deltaRequest("POST", "/v2/orders", orderData);
    res.json(result);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

app.get("/api/status", async (req, res) => {
  try {
    const { apiKey } = getCredentials();
    const maskedKey = apiKey.substring(0, 6) + "..." + apiKey.substring(apiKey.length - 4);
    
    const ipRes = await fetch("https://api.ipify.org?format=json");
    const ipData = await ipRes.json() as { ip: string };
    
    res.json({
      apiKey: maskedKey,
      serverIp: ipData.ip,
      status: "connected"
    });
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

app.get("/api/auth/websocket", (req, res) => {
  try {
    const { apiKey, apiSecret } = getCredentials();
    const method = "GET";
    const timestamp = Math.floor(Date.now() / 1000).toString();
    const path = "/live";
    const signature = generateSignature(apiSecret, method, timestamp, path);
    
    res.json({
      api_key: apiKey,
      signature: signature,
      timestamp: timestamp
    });
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

async function logServerInfo() {
  try {
    const { apiKey } = getCredentials();
    const maskedKey = apiKey.substring(0, 6) + "..." + apiKey.substring(apiKey.length - 4);
    console.log(`[Delta] Using API Key: ${maskedKey}`);
    
    const ipRes = await fetch("https://api.ipify.org?format=json");
    const ipData = await ipRes.json() as { ip: string };
    console.log(`[Delta] Server Public IP: ${ipData.ip}`);
  } catch (error: any) {
    console.warn("[Delta] Could not fetch server info:", error.message);
  }
}

async function startServer() {
  // Perform initialization tasks asynchronously
  logServerInfo().catch(err => console.warn("[Server] Info fetch failed:", err));
  fetchProducts().catch(err => console.warn("[Server] Product fetch failed:", err));

  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  // Start listening at the end to ensure all middleware is ready
  const server = app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://0.0.0.0:${PORT}`);
  });

  process.on('unhandledRejection', (reason, promise) => {
    console.error('Unhandled Rejection at:', promise, 'reason:', reason);
  });

  process.on('uncaughtException', (err) => {
    console.error('Uncaught Exception:', err);
  });
}

startServer();
