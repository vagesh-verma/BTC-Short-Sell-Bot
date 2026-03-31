import express from "express";
import { createServer as createViteServer } from "vite";
import path from "path";
import crypto from "crypto";
import fetch from "node-fetch";
import dotenv from "dotenv";
import { tradingService } from "./server/tradingService";
import { GRUModel } from "./server/modelService";

dotenv.config();

const app = express();
const PORT = Number(process.env.PORT) || 3000;

app.use(express.json({ limit: '50mb' }));

// Trading API Endpoints
app.post("/api/trading/start", async (req, res) => {
  const { settings, isReal } = req.body;
  await tradingService.start(settings, isReal);
  res.json({ success: true });
});

app.post("/api/trading/stop", (req, res) => {
  tradingService.stop();
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

function generateSignature(secret: string, method: string, nonce: string, endpoint: string, body: string = "") {
  const payload = method + nonce + endpoint + body;
  return crypto.createHmac("sha256", secret).update(payload).digest("hex");
}

let lastRestNonce = 0;
function getNextRestNonce() {
  let now = Date.now();
  if (now <= lastRestNonce) {
    now = lastRestNonce + 1;
  }
  lastRestNonce = now;
  return now.toString();
}

async function deltaRequest(method: string, endpoint: string, data: any = null) {
  const { apiKey, apiSecret } = getCredentials();

  const timestamp = Math.floor(Date.now() / 1000).toString();
  
  // Ensure the body is stringified ONCE and used for both signature and request
  const body = data ? JSON.stringify(data) : "";
  
  const signaturePayload = method + timestamp + endpoint + body;
  const signature = crypto.createHmac("sha256", apiSecret).update(signaturePayload).digest("hex");

  console.log(`[Delta] Request: ${method} ${endpoint}`);
  console.log(`[Delta] Payload: ${signaturePayload}`);

  const headers = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "api-key": apiKey,
    "signature": signature,
    "timestamp": timestamp,
    "User-Agent": "node-js-client"
  };

  const options: any = {
    method,
    headers,
    body: data ? body : undefined
  };

  console.log(`[Delta] Headers:`, JSON.stringify({ ...headers, "api-key": "MASKED" }, null, 2));

  const response = await fetch(`${DELTA_BASE_URL}${endpoint}`, options);
  const result = await response.json() as any;

  if (!response.ok) {
    console.error(`Delta API Error (${method} ${endpoint}):`, JSON.stringify(result, null, 2));
    throw new Error(JSON.stringify(result.error || result));
  }

  return result;
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
  // Start listening immediately to satisfy Cloud Run startup probes
  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });

  // Perform initialization tasks
  try {
    await logServerInfo();
    await fetchProducts();
  } catch (err) {
    console.error("[Server] Initialization failed:", err);
  }

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
}

startServer();
