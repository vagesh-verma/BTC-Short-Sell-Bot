import fetch from 'node-fetch';
import crypto from 'crypto';

const DELTA_BASE_URL = "https://api.india.delta.exchange";

function getCredentials() {
  const apiKey = process.env.DELTA_API_KEY?.trim();
  const apiSecret = process.env.DELTA_API_SECRET?.trim();
  if (!apiKey || !apiSecret) {
    throw new Error("Delta API credentials missing. Please set DELTA_API_KEY and DELTA_API_SECRET in Settings.");
  }
  return { apiKey, apiSecret };
}

export async function deltaRequest(method: string, endpoint: string, data: any = null) {
  const { apiKey, apiSecret } = getCredentials();
  const timestamp = Math.floor(Date.now() / 1000).toString();
  const body = data ? JSON.stringify(data) : "";
  const signaturePayload = method + timestamp + endpoint + body;
  const signature = crypto.createHmac("sha256", apiSecret).update(signaturePayload).digest("hex");

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

  const response = await fetch(`${DELTA_BASE_URL}${endpoint}`, options);
  const result = await response.json() as any;

  if (!response.ok) {
    throw new Error(JSON.stringify(result.error || result));
  }

  return result;
}

export function generateSignature(secret: string, method: string, nonce: string, endpoint: string, body: string = "") {
  const payload = method + nonce + endpoint + body;
  return crypto.createHmac("sha256", secret).update(payload).digest("hex");
}
