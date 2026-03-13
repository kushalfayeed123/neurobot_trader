import asyncio
import joblib
import pandas as pd
import numpy as np
import logging
import httpx  
import sys  
import os
import threading
import time
from flask import Flask
from deriv_api import DerivAPI
from dotenv import load_dotenv

# --- WEB SERVER & TELEGRAM POLLING ---
app = Flask(__name__)
@app.route('/')
def health_check(): return "Active", 200

def run_web():
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

threading.Thread(target=run_web, daemon=True).start()

# --- ENGINE ---
os.environ['PYTHONUNBUFFERED'] = '1'
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class ProductionEngine:
    def __init__(self):
        self.token = os.getenv('DERIV_TOKEN')
        self.app_id = os.getenv('DERIV_APP_ID')
        self.tg_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        # Load AI
        self.model = joblib.load('trading_model.joblib')
        
        # State
        self.buffer = pd.DataFrame()
        self.current_contract = None
        self.session_profit = 0.0
        self.last_tick_time = time.time()
        
        # Diagnostics
        self.last_prob = 0.5
        self.last_rsi = 50.0
        self.last_trend_gap = 0.0
        self.shadow_wins = 0
        self.shadow_losses = 0
        self.last_update_id = 0

    async def start(self):
        while True:
            api = None
            try:
                api = DerivAPI(app_id=self.app_id)
                authorize = await api.authorize(self.token)
                self.start_balance = float(authorize['authorize']['balance'])
                
                logging.info(f"Connected. Balance: ${self.start_balance}")
                
                # Background tasks
                asyncio.create_task(self.periodic_status_report())
                asyncio.create_task(self.telegram_command_listener())

                tick_observable = await api.subscribe({'ticks': 'R_10'})
                tick_observable.subscribe(lambda msg: self.safe_handle_tick(msg, api))

                while True:
                    await asyncio.sleep(15)
                    if time.time() - self.last_tick_time > 45:
                        logging.error("Watchdog triggered. Reconnecting...")
                        break 

            except Exception as e:
                logging.error(f"Main Loop Error: {e}")
            finally:
                if api: await api.disconnect()
                await asyncio.sleep(5)

    def safe_handle_tick(self, msg, api):
        try:
            asyncio.create_task(self.handle_tick(msg, api))
        except Exception: pass

    async def handle_tick(self, msg, api):
        if not msg or 'tick' not in msg: return
        self.last_tick_time = time.time()
        
        price = msg['tick']['quote']
        self.buffer = pd.concat([self.buffer, pd.DataFrame([{'price': price}])], ignore_index=True).tail(60)
        
        if len(self.buffer) < 30: return

        # Indicators
        prices = self.buffer['price']
        sma_s = prices.rolling(5).mean().iloc[-1]
        sma_l = prices.rolling(20).mean().iloc[-1]
        self.last_trend_gap = ((sma_s / sma_l) - 1) * 100
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).tail(14).mean()
        loss = (-delta.where(delta < 0, 0)).tail(14).mean()
        rs = gain / loss if loss != 0 else 0
        self.last_rsi = 100 - (100 / (1 + rs))

        # AI
        feat = pd.DataFrame([{
            'returns': prices.pct_change().iloc[-1],
            'volatility': prices.pct_change().rolling(14).std().iloc[-1],
            'rsi': self.last_rsi,
            'momentum': prices.iloc[-1] - prices.iloc[-10],
            'sma_signal': 1 if sma_s > sma_l else 0
        }]).fillna(0)

        self.last_prob = self.model.predict_proba(feat)[0][1]

        # Sniper Logic
        if not self.current_contract:
            if self.last_prob > 0.98 and self.last_trend_gap > 0.02 and self.last_rsi < 60:
                await self.place_trade(api, "CALL")
            elif self.last_prob < 0.02 and self.last_trend_gap < -0.02 and self.last_rsi > 50:
                await self.place_trade(api, "PUT")
            elif 0.85 < self.last_prob < 0.98 or 0.02 < self.last_prob < 0.15:
                asyncio.create_task(self.shadow_monitor(price, "CALL" if self.last_prob > 0.5 else "PUT"))

    async def shadow_monitor(self, entry_price, direction):
        entry_rsi = self.last_rsi
        entry_gap = self.last_trend_gap
        entry_prob = self.last_prob
        
        await asyncio.sleep(12)
        exit_price = self.buffer['price'].iloc[-1]
        win = (exit_price > entry_price) if direction == "CALL" else (exit_price < entry_price)
        if win: self.shadow_wins += 1
        else: self.shadow_losses += 1
        
        # Post-Mortem Report for Shadow Trades
        status = "✅ SHADOW WIN" if win else "❌ SHADOW LOSS"
        msg = (
            f"{status} ({direction})\n"
            f"--- ---\n"
            f"🧠 Prob: `{entry_prob:.2%}`\n"
            f"📊 Gap: `{entry_gap:.4f}%`\n"
            f"📉 RSI: `{entry_rsi:.1f}`"
        )
        await self.send_telegram_msg(msg)

    async def place_trade(self, api, direction):
        if self.current_contract or self.session_profit <= -1.50: return
        params = {
            "buy": 1, "price": 1.0,
            "parameters": {
                "amount": 1.0, "basis": "stake", "contract_type": direction,
                "currency": "USD", "duration": 12, "duration_unit": "t", "symbol": "R_10"
            }
        }
        try:
            resp = await api.buy(params)
            if 'buy' in resp:
                self.current_contract = resp['buy']['contract_id']
                await self.send_telegram_msg(f"🎯 *LIVE ENTRY: {direction}*\nProb: `{self.last_prob:.2%}`\nGap: `{self.last_trend_gap:.4f}%`")
                asyncio.create_task(self.monitor_contract(api, self.current_contract))
        except: pass

    async def monitor_contract(self, api, contract_id):
        try:
            status_obs = await api.subscribe({"proposal_open_contract": 1, "contract_id": contract_id})
            finished = asyncio.get_event_loop().create_future()
            def handle_update(msg):
                c = msg.get('proposal_open_contract', {})
                if c.get('is_sold'):
                    profit = float(c.get('profit', 0))
                    self.session_profit += profit
                    asyncio.create_task(self.send_telegram_msg(f"🏁 *RESULT: {'WIN' if profit > 0 else 'LOSS'}*\nProfit: `${profit}`\nSession: `${self.session_profit:.2f}`"))
                    if not finished.done(): finished.set_result(True)
            sub = status_obs.subscribe(handle_update)
            await asyncio.wait_for(finished, timeout=60)
            sub.dispose()
        finally: self.current_contract = None

    async def get_diagnostic_text(self):
        sentiment = "🐂 BULL" if self.last_prob > 0.70 else "🐻 BEAR" if self.last_prob < 0.30 else "😐 NEUT"
        return (
            f"📡 *Sniper Diagnostic*\n"
            f"💰 P/L: `${self.session_profit:.2f}`\n"
            f"👻 Shadow: `{self.shadow_wins}W - {self.shadow_losses}L`\n"
            f"--- ---\n"
            f"🧠 AI: `{self.last_prob:.2%}` ({sentiment})\n"
            f"📉 RSI: `{self.last_rsi:.1f}`\n"
            f"📊 Gap: `{self.last_trend_gap:+.4f}%` (Min ±0.02)\n"
            f"⏱ Stream: `{time.strftime('%H:%M:%S')}`\n"
            f"--- ---\n"
            f"_{'OPEN POSITION' if self.current_contract else 'WAITING'}_"
        )

    async def periodic_status_report(self):
        while True:
            await asyncio.sleep(300)
            await self.send_telegram_msg(await self.get_diagnostic_text())

    async def telegram_command_listener(self):
        """Polls Telegram for /status commands."""
        while True:
            try:
                async with httpx.AsyncClient() as client:
                    url = f"https://api.telegram.org/bot{self.tg_token}/getUpdates"
                    params = {"offset": self.last_update_id + 1, "timeout": 30}
                    resp = await client.get(url, params=params)
                    if resp.status_code == 200:
                        data = resp.json()
                        for update in data.get("result", []):
                            self.last_update_id = update["update_id"]
                            msg = update.get("message", {})
                            text = msg.get("text", "")
                            if text == "/status":
                                await self.send_telegram_msg(await self.get_diagnostic_text())
            except: pass
            await asyncio.sleep(1)

    async def send_telegram_msg(self, message):
        url = f"https://api.telegram.org/bot{self.tg_token}/sendMessage"
        async with httpx.AsyncClient() as client:
            try: await client.post(url, json={"chat_id": self.chat_id, "text": message, "parse_mode": "Markdown"})
            except: pass

if __name__ == "__main__":
    asyncio.run(ProductionEngine().start())