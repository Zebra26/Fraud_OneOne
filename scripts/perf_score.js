import http from 'k6/http';
import { check, sleep } from 'k6';
import { uuidv4 } from 'https://jslib.k6.io/k6-utils/1.4.0/index.js';
import crypto from 'k6/crypto';
import encoding from 'k6/encoding';

const BACKEND = __ENV.BACKEND_URL || 'http://127.0.0.1:8000';
const JWT_SECRET = __ENV.JWT_SECRET_KEY || '';
const API_HMAC = __ENV.API_HMAC_KEY || '';

function b64url(s){return s.replace(/\+/g,'-').replace(/\//g,'_').replace(/=+$/,'');}
function decodeKey(k){ try{ return encoding.b64decode(k,'std','s'); } catch(e){ return new TextEncoder().encode(k);} }
function hs256(payload){
  const header={alg:'HS256',typ:'JWT'};
  const head=b64url(encoding.b64encode(JSON.stringify(header)));
  const body=b64url(encoding.b64encode(JSON.stringify(payload)));
  const sigRaw=crypto.hmac('sha256', `${head}.${body}`, decodeKey(JWT_SECRET), 'binary');
  const sig=b64url(encoding.b64encode(sigRaw));
  return `${head}.${body}.${sig}`;
}
function hmacSig(method,path,ts,nonce,body){
  const data = `${method}|${path}|${ts}|${nonce}|` + (body||'');
  const raw = crypto.hmac('sha256', data, decodeKey(API_HMAC), 'binary');
  return encoding.b64encode(raw);
}

export const options = {
  scenarios: {
    rps_1000: { executor:'constant-arrival-rate', rate:1000, timeUnit:'1s', duration:'30s', preAllocatedVUs:100, maxVUs:1000 },
  },
};

export default function(){
  const path='/predictions/score';
  const url=`${BACKEND}${path}`;
  const jwt=hs256({sub:'loadtest',roles:['service'],iss:'fraud_k',aud:'fraud_api',iat:Math.floor(Date.now()/1000),exp:Math.floor(Date.now()/1000)+300,jti:uuidv4()});
  const ts=String(Math.floor(Date.now()/1000)); const nonce=uuidv4();
  const payload = JSON.stringify({
    transaction_id:`txn_${uuidv4()}`, account_id:`acc_${uuidv4()}`, amount:500, currency:'EUR', channel:'mobile',
    features:{ transaction_amount:500, transaction_time_seconds:1, is_weekend:0, hour_of_day:12, is_round_amount:0,
      unique_receivers_24h:1, vpn_detected:0, location_risk_score:0.1, transaction_frequency_30min:1, login_ip_changed_last_hour:0,
      avg_transaction_amount_24h:120, time_since_last_tx:60, ip_risk_score:0.05, transactions_last_24h:3, customer_segment:'standard' }
  });
  const sig=hmacSig('POST', path, ts, nonce, payload);
  const headers={ 'Authorization':`Bearer ${jwt}`, 'Content-Type':'application/json', 'X-Request-Timestamp':ts, 'X-Request-Nonce':nonce, 'X-Request-Signature':sig, 'Idempotency-Key':uuidv4(), 'X-Device-ID':'device123' };
  const res=http.post(url, payload, { headers });
  check(res, { '200|503': r => r.status===200 || r.status===503 });
  sleep(0.001);
}

export function handleSummary(data){
  const p95 = data.metrics.http_req_duration && data.metrics.http_req_duration.percentiles ? data.metrics.http_req_duration.percentiles['p(95)'] : null;
  const summary = { event:'k6_summary', p95_ms: p95 };
  console.log(JSON.stringify(summary));
  return { 'k6_summary.json': JSON.stringify(summary, null, 2) };
}

