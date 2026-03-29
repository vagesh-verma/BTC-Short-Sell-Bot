import express from "express";
import { createServer as createViteServer } from "vite";
import path from "path";
import crypto from "crypto";
import fetch from "node-fetch";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = 3000;

app.use(express.json());

const DELTA_BASE_URL = "https://api.india.delta.exchange";

function getCredentials() {
  const apiKey = process.env.DELTA_API_KEY;
  const apiSecret = process.env.DELTA_API_SECRET;
  if (!apiKey || !apiSecret) {
    throw new Error("Delta API credentials missing. Please set DELTA_API_KEY and DELTA_API_SECRET in Settings.");
  }
  return { apiKey, apiSecret };
}

function generateSignature(secret: string, method: string, nonce: string, path: string, query: string = "", body: string = "") {
  const payload = method + nonce + path + query + body;
  return crypto.createHmac("sha256", secret).update(payload).digest("hex");
}

async function deltaRequest(method: string, endpoint: string, data: any = null) {
  const { apiKey, apiSecret } = getCredentials();

  const nonce = Math.floor(Date.now() / 1000).toString();
  const body = data ? JSON.stringify(data) : "";
  const signature = generateSignature(apiSecret, method, nonce, endpoint, "", body);
// ... existing headers logic ...

  const headers = {
    "Content-Type": "application/json",
    "api-key": apiKey,
    "api-signature": signature,
    "api-nonce": nonce,
  };

  const options: any = {
    method,
    headers,
  };

  if (data) {
    options.body = body;
  }

  const response = await fetch(`${DELTA_BASE_URL}${endpoint}`, options);
  const result = await response.json();

  if (!response.ok) {
    console.error("Delta API Error:", result);
    const errorMsg = (result as any).error?.message || "Delta API Error";
    throw new Error(errorMsg);
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
    // symbol should be 'BTCUSD' or similar
    // side: 'buy' or 'sell'
    // order_type: 'market' or 'limit'
    
    const orderData = {
      product_id: 1, // BTCUSD on Delta India is usually product_id 1, but we should fetch it or use symbol
      side,
      order_type,
      size: Math.floor(size),
    };

    if (order_type === 'limit' && limit_price) {
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
  await logServerInfo();
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

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
