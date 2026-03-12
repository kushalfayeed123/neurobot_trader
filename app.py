import asyncio
import joblib
import pandas as pd
import numpy as np
import logging
import httpx  
import sys  
import os
from deriv_api import DerivAPI
from dotenv import load_dotenv
import threading
from flask import Flask

# 1. Hugging Face Health Check Server
app = Flask(__name__)
@app.route('/')
def health(): return "✅ Bot is Running"

def run_health_server():
    app.run(host='0.0.0.0', port=7860)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

class ProductionEngine:
    def __init__(self):
        self.token = os.getenv('DERIV_TOKEN')
        self.app_id = os.getenv('DERIV_APP_ID')
        self.tg_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')

        if not all([self.token, self.app_id, self.tg_token, self.chat_id]):
            print("CRITICAL ERROR: Missing environment variables")
            exit(1)
            
        try:
            self.model = joblib.load('trading_model.joblib')
            print("Model loaded successfully.")
        except Exception as e:
            print(f"CRITICAL: Could not load model: {e}")
            exit()

        self.buffer = pd.DataFrame()
        self.current_contract = None
        self.session_profit = 0.0
        self.max_loss = -3.0
        self.tick_count = 0
        
        # Performance Tracking for Reports
        self.shadow_wins = 0
        self.shadow_losses = 0
        self.start_balance = 0.0
        self.last_prob = 0.5

    async def start(self):
        print(f"Attempting to connect with AppID: {self.app_id}...")
        while True:
            api = None
            try:
                api = DerivAPI(app_id=self.app_id)
                authorize = await api.authorize(self.token)
                self.start_balance = float(authorize['authorize']['balance'])
                
                print(f"SUCCESS! Connected. Balance: ${self.start_balance}")
                
                # START PERIODIC UPDATES (Every 2 Minutes)
                asyncio.create_task(self.periodic_status_report())

                print("Subscribing to R_10 ticks...")
                tick_observable = await api.subscribe({'ticks': 'R_10'})
                tick_observable.subscribe(
                    lambda msg: asyncio.create_task(self.handle_tick(msg, api))
                )

                print("--- BOT IS NOW LIVE AND LISTENING ---")

                while True:
                    await asyncio.sleep(10)

            except Exception as e:
                print(f"\nCONNECTION ERROR: {e}")
                if api: await api.disconnect()
                await asyncio.sleep(10)

    async def periodic_status_report(self):
        """Sends a Telegram update every 2 minutes."""
        while True:
            await asyncio.sleep(120) # 120 seconds = 2 minutes
            current_total = self.start_balance + self.session_profit
            status_msg = (
                "🤖 *R10 Sniper: 2-Min Status*\n"
                f"💰 Balance: `${current_total:.2f}`\n"
                f"📈 P/L: `${self.session_profit:.2f}`\n"
                f"📊 Shadow: `{self.shadow_wins}W - {self.shadow_losses}L`\n"
                f"🕒 Ticks: `{self.tick_count}`\n"
                f"⚡ Confidence: `{self.last_prob:.2%}`"
            )
            await self.send_telegram_msg(status_msg)

    async def handle_tick(self, msg, api):
        if not msg or 'tick' not in msg: return
        price = msg['tick']['quote']
        self.tick_count += 1
        
        new_row = pd.DataFrame([{'price': price}])
        self.buffer = pd.concat([self.buffer, new_row], ignore_index=True).tail(50)

        if len(self.buffer) < 20: return

        # Feature Engineering
        latest_prices = self.buffer['price']
        delta = latest_prices.diff()
        gain = (delta.where(delta > 0, 0)).tail(14).mean()
        loss = (-delta.where(delta < 0, 0)).tail(14).mean()
        rs = gain / loss if loss != 0 else 0
        current_rsi = 100 - (100 / (1 + rs))
        mom_short = latest_prices.iloc[-1] - latest_prices.iloc[-3]
        sma_5 = latest_prices.rolling(5).mean().iloc[-1]
        sma_10 = latest_prices.rolling(10).mean().iloc[-1]
        sma_signal = 1 if sma_5 > sma_10 else 0

        feat = pd.DataFrame([{
            'returns': latest_prices.pct_change().iloc[-1],
            'volatility': latest_prices.pct_change().rolling(14).std().iloc[-1],
            'rsi': current_rsi,
            'momentum': latest_prices.iloc[-1] - latest_prices.iloc[-5],
            'sma_signal': sma_signal
        }]).fillna(0)

        self.last_prob = self.model.predict_proba(feat)[0][1]

        # Dashboard Output
        sys.stdout.write(f"\rPrice: {price:.2f} | Prob: {self.last_prob:.2%} | RSI: {current_rsi:.1f}")
        sys.stdout.flush()

        if not self.current_contract:
            # Signal Logic
            if self.last_prob > 0.86:
                if current_rsi >= 60 or mom_short <= 0.10:
                    asyncio.create_task(self.shadow_monitor(price, "CALL"))
                else:
                    await self.place_trade(api, "CALL", self.last_prob)
            elif self.last_prob < 0.14:
                if current_rsi <= 40 or mom_short >= -0.10:
                    asyncio.create_task(self.shadow_monitor(price, "PUT"))
                else:
                    await self.place_trade(api, "PUT", self.last_prob)

    async def shadow_monitor(self, entry_price, direction):
        start_count = self.tick_count
        while self.tick_count < start_count + 5:
            await asyncio.sleep(1)
        
        exit_price = self.buffer['price'].iloc[-1]
        win = (exit_price > entry_price) if direction == "CALL" else (exit_price < entry_price)
        
        if win: self.shadow_wins += 1
        else: self.shadow_losses += 1
        logging.info(f" [SHADOW] {direction} {'WON' if win else 'LOST'}")

    async def place_trade(self, api, direction, prob):
        if self.current_contract: return
        stake = 1.0
        trade_params = {
            "buy": 1, "price": stake,
            "parameters": {
                "amount": stake, "basis": "stake", "contract_type": direction,
                "currency": "USD", "duration": 5, "duration_unit": "t", "symbol": "R_10"
            }
        }
        try:
            response = await api.buy(trade_params)
            if 'error' in response:
                logging.error(f"Trade Rejected: {response['error']['message']}")
                return
            self.current_contract = response['buy']['contract_id']
            await self.send_telegram_msg(f"🚀 *LIVE TRADE*\nDir: {direction}\nProb: {prob:.2%}")
            asyncio.create_task(self.monitor_contract(api, self.current_contract))
        except Exception as e:
            logging.error(f"Buy Error: {e}")

    async def monitor_contract(self, api, contract_id):
        finished_signal = asyncio.get_event_loop().create_future()
        async def handle_update(msg):
            contract = msg.get('proposal_open_contract', {})
            if contract.get('contract_id') == contract_id and contract.get('is_sold'):
                profit = float(contract.get('profit', 0))
                self.session_profit += profit
                await self.send_telegram_msg(f"🏁 *RESULT*\nProfit: ${profit}\nSession: ${self.session_profit:.2f}")
                if not finished_signal.done(): finished_signal.set_result(True)

        status_obs = await api.subscribe({"proposal_open_contract": 1, "contract_id": contract_id})
        subscription = status_obs.subscribe(handle_update)
        await finished_signal
        subscription.dispose()
        self.current_contract = None

    async def send_telegram_msg(self, message):
        url = f"https://api.telegram.org/bot{self.tg_token}/sendMessage"
        
        # Retry logic for DNS issues
        for attempt in range(3):
            async with httpx.AsyncClient(timeout=15.0) as client:
                try:
                    response = await client.post(
                        url, 
                        json={"chat_id": self.chat_id, "text": message, "parse_mode": "Markdown"}
                    )
                    response.raise_for_status()
                    return # Success!
                except (httpx.ConnectError, httpx.HTTPStatusError, httpx.RequestError) as e:
                    logging.warning(f"TG Attempt {attempt+1} failed: {e}")
                    if attempt < 2:
                        await asyncio.sleep(5)
                    else:
                        logging.error("TG Final attempt failed. Check Space internet/DNS.")

if __name__ == "__main__":
    # Force logs to flush immediately in Docker/Hugging Face
    os.environ['PYTHONUNBUFFERED'] = '1'
    
    threading.Thread(target=run_health_server, daemon=True).start()
    
    bot = ProductionEngine()
    print("--- STARTING BOT ENGINE ---", flush=True)
    asyncio.run(bot.start())

if __name__ == "__main__":
    threading.Thread(target=run_health_server, daemon=True).start()
    asyncio.run(ProductionEngine().start())