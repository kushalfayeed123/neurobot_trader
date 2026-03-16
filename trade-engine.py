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
import json

# --- WEB SERVER (For Deployment Health Checks) ---
app = Flask(__name__)
@app.route('/')
def health_check(): return "Active", 200


def run_web():
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)


threading.Thread(target=run_web, daemon=True).start()

# --- ENGINE CONFIG ---
os.environ['PYTHONUNBUFFERED'] = '1'
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


class ProductionEngine:
    def __init__(self):
        self.token = os.getenv('DERIV_TOKEN')
        self.app_id = os.getenv('DERIV_APP_ID')
        self.tg_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')

        self.model = joblib.load('trading_model.joblib')

        # State
        self.buffer = pd.DataFrame()
        self.current_contract = None
        self.session_profit = 0.0
        self.last_tick_time = time.time()

        # Diagnostics & Safety
        self.last_prob = 0.5
        self.last_rsi = 50.0
        self.last_trend_gap = 0.0
        self.shadow_wins = 0
        self.shadow_losses = 0
        self.last_update_id = 0
        self.cooldown_until = 0
        self.is_shadow_active = False
        self.max_drawdown = -3.00  
        self.stats_file = "trading_stats.json"
        self.load_stats() # Load wins/losses on startup

    async def start(self):
        while True:
            api = None
            try:
                api = DerivAPI(app_id=self.app_id)
                authorize = await api.authorize(self.token)
                self.start_balance = float(authorize['authorize']['balance'])
                logging.info(f"Connected. Balance: ${self.start_balance}")

                # DEFINED BACKGROUND TASKS
                asyncio.create_task(self.periodic_status_report())
                asyncio.create_task(self.telegram_command_listener())

                tick_observable = await api.subscribe({'ticks': 'R_10'})
                tick_observable.subscribe(
                    lambda msg: self.safe_handle_tick(msg, api))

                while True:
                    await asyncio.sleep(15)
                    if time.time() - self.last_tick_time > 45:
                        logging.error("Watchdog triggered. Reconnecting...")
                        break
            except Exception as e:
                logging.error(f"Main Loop Error: {e}")
            finally:
                if api:
                    await api.disconnect()
                await asyncio.sleep(5)

    def safe_handle_tick(self, msg, api):
        try:
            asyncio.create_task(self.handle_tick(msg, api))
        except:
            pass
    
    def load_stats(self):
        """Load shadow stats from a local file."""
        try:
            if os.path.exists(self.stats_file):
                with open(self.stats_file, 'r') as f:
                    data = json.load(f)
                    self.shadow_wins = data.get('shadow_wins', 0)
                    self.shadow_losses = data.get('shadow_losses', 0)
                    self.session_profit = data.get('session_profit', 0.0)
            else:
                self.shadow_wins = 0
                self.shadow_losses = 0
        except:
            self.shadow_wins = 0
            self.shadow_losses = 0

    def save_stats(self):
        """Save shadow stats to a local file."""
        try:
            with open(self.stats_file, 'w') as f:
                json.dump({
                    'shadow_wins': self.shadow_wins,
                    'shadow_losses': self.shadow_losses,
                    'session_profit': self.session_profit
                }, f)
        except:
            pass

    async def handle_tick(self, msg, api):
        if not msg or 'tick' not in msg:
            return

        # 1. COOLDOWN & EQUITY GATES
        now = time.monotonic()
        if now < self.cooldown_until or self.session_profit <= self.max_drawdown:
            return

        self.last_tick_time = time.time()
        price = msg['tick']['quote']
        self.buffer = pd.concat([self.buffer, pd.DataFrame(
            [{'price': price}])], ignore_index=True).tail(60)

        if len(self.buffer) < 30:
            return

        # Indicators
        prices = self.buffer['price']
        sma_s = prices.rolling(5).mean().iloc[-1]
        sma_l = prices.rolling(20).mean().iloc[-1]
        self.last_trend_gap = ((sma_s / sma_l) - 1) * 100

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).tail(14).mean()
        loss = (-delta.where(delta < 0, 0)).tail(14).mean()
        self.last_rsi = 100 - (100 / (1 + (gain/loss))) if loss != 0 else 100

        # AI
        feat = pd.DataFrame([{
            'returns': prices.pct_change().iloc[-1],
            'volatility': prices.pct_change().rolling(14).std().iloc[-1],
            'rsi': self.last_rsi,
            'momentum': prices.iloc[-1] - prices.iloc[-10],
            'sma_signal': 1 if sma_s > sma_l else 0
        }]).fillna(0)
        self.last_prob = self.model.predict_proba(feat)[0][1]

       
            # --- ACTIVE SNIPER EXECUTION ---
        if not self.current_contract:
            # CALL ZONE: Lowered threshold to 0.90 to catch the wins in your logs
            if 20 < self.last_rsi < 65:
                # We trade at 90% IF the Trend Gap is positive and moving
                if self.last_prob > 0.90 and self.last_trend_gap > 0.005:
                    await self.place_trade(api, "CALL")
                    return 

            # PUT ZONE: Lowered threshold to 0.10
            elif 35 < self.last_rsi < 80:
                # We trade at 10% IF the Trend Gap is negative
                if self.last_prob < 0.10 and self.last_trend_gap < -0.005:
                    await self.place_trade(api, "PUT")
                    return

            # SHADOW TRADE
            if not self.is_shadow_active:
                if (0.88 < self.last_prob < 0.94) or (0.06 < self.last_prob < 0.12):
                    if (20 < self.last_rsi < 65 and self.last_prob > 0.5) or \
                       (35 < self.last_rsi < 80 and self.last_prob < 0.5):
                        asyncio.create_task(self.shadow_monitor(
                            price, "CALL" if self.last_prob > 0.5 else "PUT"))

    async def shadow_monitor(self, entry_price, side):
        self.is_shadow_active = True
        e_rsi, e_gap, e_prob = self.last_rsi, self.last_trend_gap, self.last_prob
        await asyncio.sleep(12)
        exit_p = self.buffer['price'].iloc[-1]
        win = (exit_p > entry_price) if side == "CALL" else (
            exit_p < entry_price)

        if win:
            self.shadow_wins += 1
            res_txt = "✅ SHADOW WIN"
        else:
            self.shadow_losses += 1
            res_txt = "❌ SHADOW LOSS"
            self.cooldown_until = time.monotonic() + 300
            await self.send_telegram_msg("⚠️ *SHADOW LOSS:* 5-min Cooldown Active.")
        self.save_stats()
        await self.send_telegram_msg(f"{res_txt}\nProb: `{e_prob:.2%}`\nGap: `{e_gap:.4f}%`\nRSI: `{e_rsi:.1f}`")
        self.is_shadow_active = False

    async def periodic_status_report(self):
        """Pushes the diagnostic every 5 minutes automatically."""
        while True:
            await asyncio.sleep(300)
            try:
                await self.send_telegram_msg(await self.get_diagnostic_text())
            except Exception as e:
                logging.error(f"Status Report Error: {e}")

    async def place_trade(self, api, direction):
        if self.current_contract:
            return
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
                await self.send_telegram_msg(f"🎯 *LIVE ENTRY: {direction}*")
                asyncio.create_task(self.monitor_contract(
                    api, self.current_contract))
        except:
            pass

    async def monitor_contract(self, api, contract_id):
        try:
            status_obs = await api.subscribe({"proposal_open_contract": 1, "contract_id": contract_id})
            finished = asyncio.get_event_loop().create_future()

            def handle_update(msg):
                c = msg.get('proposal_open_contract', {})
                if c.get('is_sold'):
                    profit = float(c.get('profit', 0))
                    self.session_profit += profit
                    self.save_stats() # Save live profit updates
                    asyncio.create_task(self.send_telegram_msg(
                        f"🏁 *RESULT: {'WIN' if profit > 0 else 'LOSS'}*\nSession: `${self.session_profit:.2f}`"))
                    if not finished.done():
                        finished.set_result(True)

            sub = status_obs.subscribe(handle_update)
            await asyncio.wait_for(finished, timeout=60)
            sub.dispose()
        finally:
            self.current_contract = None

    async def get_diagnostic_text(self):
        sentiment = "🐂 BULL" if self.last_prob > 0.70 else "🐻 BEAR" if self.last_prob < 0.30 else "😐 NEUT"
        cd_left = max(0, int(self.cooldown_until - time.monotonic()))
        cd_status = f"\n⏱ *Cooldown:* `{cd_left}s`" if cd_left > 0 else ""
        return (
            f"📡 *Sniper Diagnostic*\n💰 P/L: `${self.session_profit:.2f}`\n👻 Shadow: `{self.shadow_wins}W - {self.shadow_losses}L`"
            f"\n--- ---\n🧠 AI: `{self.last_prob:.2%}` ({sentiment})\n📉 RSI: `{self.last_rsi:.1f}`\n📊 Gap: `{self.last_trend_gap:+.4f}%`"
            f"{cd_status}\n--- ---\n_{'OPEN POSITION' if self.current_contract else 'WAITING'}_"
        )

    async def telegram_command_listener(self):
        while True:
            try:
                async with httpx.AsyncClient() as client:
                    url = f"https://api.telegram.org/bot{self.tg_token}/getUpdates"
                    params = {"offset": self.last_update_id + 1, "timeout": 30}
                    resp = await client.get(url, params=params)
                    if resp.status_code == 200:
                        for update in resp.json().get("result", []):
                            self.last_update_id = update["update_id"]
                            text = update.get("message", {}).get("text", "")
                            if text == "/status":
                                await self.send_telegram_msg(await self.get_diagnostic_text())
                            elif text == "/reset":
                                self.cooldown_until = 0
                                await self.send_telegram_msg("🔄 *Cooldown Reset.* Sniper ready.")
            except:
                pass
            await asyncio.sleep(1)

    async def send_telegram_msg(self, message):
        url = f"https://api.telegram.org/bot{self.tg_token}/sendMessage"
        async with httpx.AsyncClient() as client:
            try:
                await client.post(url, json={"chat_id": self.chat_id, "text": message, "parse_mode": "Markdown"})
            except:
                pass


if __name__ == "__main__":
    asyncio.run(ProductionEngine().start())
