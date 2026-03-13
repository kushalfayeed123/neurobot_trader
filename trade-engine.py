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

# --- WEB SERVER ---
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
        
        self.model = joblib.load('trading_model.joblib')
        self.buffer = pd.DataFrame()
        self.current_contract = None
        self.session_profit = 0.0
        self.last_tick_time = time.time() # WATCHDOG
        self.last_prob = 0.5
        self.shadow_wins = 0
        self.shadow_losses = 0

    async def start(self):
        while True:
            api = None
            try:
                api = DerivAPI(app_id=self.app_id)
                authorize = await api.authorize(self.token)
                self.start_balance = float(authorize['authorize']['balance'])
                
                logging.info(f"Connected. Balance: ${self.start_balance}")
                asyncio.create_task(self.periodic_status_report())

                # Subscribe to R_10
                tick_observable = await api.subscribe({'ticks': 'R_10'})
                tick_observable.subscribe(lambda msg: self.safe_handle_tick(msg, api))

                # WATCHDOG LOOP: Reconnect if no ticks for 45 seconds
                while True:
                    await asyncio.sleep(15)
                    if time.time() - self.last_tick_time > 45:
                        logging.error("Watchdog: No ticks detected. Reconnecting...")
                        break 

            except Exception as e:
                logging.error(f"Main Loop Error: {e}")
            finally:
                if api: await api.disconnect()
                await asyncio.sleep(5)

    def safe_handle_tick(self, msg, api):
        """Wrapper to prevent the subscription from dying on error."""
        try:
            asyncio.create_task(self.handle_tick(msg, api))
        except Exception as e:
            logging.error(f"Handler Error: {e}")

    async def handle_tick(self, msg, api):
        if not msg or 'tick' not in msg: return
        self.last_tick_time = time.time() # Reset Watchdog
        
        price = msg['tick']['quote']
        new_row = pd.DataFrame([{'price': price}])
        self.buffer = pd.concat([self.buffer, new_row], ignore_index=True).tail(60)
        
        if len(self.buffer) < 30: return # Larger buffer for better SMAs

        # TECHNICAL ANALYSIS
        prices = self.buffer['price']
        sma_short = prices.rolling(5).mean().iloc[-1]
        sma_long = prices.rolling(20).mean().iloc[-1]
        
        # RSI Calculation
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).tail(14).mean()
        loss = (-delta.where(delta < 0, 0)).tail(14).mean()
        rs = gain / loss if loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))

        # AI INFERENCE
        feat = pd.DataFrame([{
            'returns': prices.pct_change().iloc[-1],
            'volatility': prices.pct_change().rolling(14).std().iloc[-1],
            'rsi': rsi,
            'momentum': prices.iloc[-1] - prices.iloc[-10], # 10-tick momentum
            'sma_signal': 1 if sma_short > sma_long else 0
        }]).fillna(0)

        self.last_prob = self.model.predict_proba(feat)[0][1]

        # REFINED SNIPER LOGIC (Ultra-Conservative)
        if not self.current_contract:
            # CALL: 98% AI Conf + Strong Momentum + Not Overbought
            if self.last_prob > 0.98 and sma_short > (sma_long * 1.0002) and rsi < 60:
                await self.place_trade(api, "CALL")
            
            # PUT: 2% AI Conf + Strong Momentum + Not Oversold
            elif self.last_prob < 0.02 and sma_short < (sma_long * 0.9998) and rsi > 40:
                await self.place_trade(api, "PUT")
            
            # Shadow monitor "Good" but not "Great" signals
            elif 0.85 < self.last_prob < 0.98 or 0.02 < self.last_prob < 0.15:
                asyncio.create_task(self.shadow_monitor(price, "CALL" if self.last_prob > 0.5 else "PUT"))

    async def shadow_monitor(self, entry_price, direction):
        await asyncio.sleep(12) # Wait for ~10-12 ticks
        exit_price = self.buffer['price'].iloc[-1]
        win = (exit_price > entry_price) if direction == "CALL" else (exit_price < entry_price)
        if win: self.shadow_wins += 1
        else: self.shadow_losses += 1

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
                await self.send_telegram_msg(f"🎯 *LIVE ENTRY: {direction}* (Conf: {self.last_prob:.1%})")
                asyncio.create_task(self.monitor_contract(api, self.current_contract))
        except Exception as e:
            logging.error(f"Trade Execution Failed: {e}")

    async def monitor_contract(self, api, contract_id):
        try:
            # Simple polling for larger account safety or stay with subscription
            status_obs = await api.subscribe({"proposal_open_contract": 1, "contract_id": contract_id})
            
            # Using a future to block until finished
            finished = asyncio.get_event_loop().create_future()

            def handle_update(msg):
                c = msg.get('proposal_open_contract', {})
                if c.get('is_sold'):
                    profit = float(c.get('profit', 0))
                    self.session_profit += profit
                    asyncio.create_task(self.send_telegram_msg(f"🏁 *RESULT: {'WIN' if profit > 0 else 'LOSS'}*\nProfit: ${profit}\nSession: ${self.session_profit:.2f}"))
                    if not finished.done(): finished.set_result(True)

            sub = status_obs.subscribe(handle_update)
            await asyncio.wait_for(finished, timeout=60)
            sub.dispose()
        finally:
            self.current_contract = None

    async def periodic_status_report(self):
        while True:
            await asyncio.sleep(300) # Increased to 5 mins to reduce noise
            status = (f"🤖 *Sniper Heartbeat*\n"
                      f"💰 P/L: `${self.session_profit:.2f}`\n"
                      f"👻 Shadow: `{self.shadow_wins}W - {self.shadow_losses}L`")
            await self.send_telegram_msg(status)

    async def send_telegram_msg(self, message):
        url = f"https://api.telegram.org/bot{self.tg_token}/sendMessage"
        async with httpx.AsyncClient() as client:
            try: await client.post(url, json={"chat_id": self.chat_id, "text": message, "parse_mode": "Markdown"})
            except: pass

if __name__ == "__main__":
    asyncio.run(ProductionEngine().start())