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

# Force logs to show up immediately in Render/Cloud consoles
os.environ['PYTHONUNBUFFERED'] = '1'

load_dotenv()

# Professional Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_log.txt', mode='a'),
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
            logging.error("CRITICAL: Missing environment variables!")
            sys.exit(1)
            
        try:
            self.model = joblib.load('trading_model.joblib')
            logging.info("AI Model loaded successfully.")
        except Exception as e:
            logging.error(f"CRITICAL: Model load failed: {e}")
            sys.exit(1)

        # State Management
        self.buffer = pd.DataFrame()
        self.current_contract = None
        self.session_profit = 0.0
        self.tick_count = 0
        self.start_balance = 0.0
        self.last_prob = 0.5
        
        # Strategy Metrics
        self.shadow_wins = 0
        self.shadow_losses = 0
        self.max_session_loss = -1.50  # Hard stop to protect the $5.99

    async def start(self):
        logging.info(f"Connecting to Deriv (AppID: {self.app_id})...")
        while True:
            api = None
            try:
                api = DerivAPI(app_id=self.app_id)
                authorize = await api.authorize(self.token)
                self.start_balance = float(authorize['authorize']['balance'])
                
                logging.info(f"✅ Connection Established. Balance: ${self.start_balance}")
                await self.send_telegram_msg(f"🔋 *Bot Online*\nBalance: `${self.start_balance}`")
                
                # Start Background Tasks
                asyncio.create_task(self.periodic_status_report())

                # Subscribe to R_10
                tick_observable = await api.subscribe({'ticks': 'R_10'})
                tick_observable.subscribe(
                    lambda msg: asyncio.create_task(self.handle_tick(msg, api))
                )

                while True:
                    await asyncio.sleep(10)

            except Exception as e:
                logging.error(f"Connection lost. Retrying in 10s... Error: {e}")
                if api: await api.disconnect()
                await asyncio.sleep(10)

    async def periodic_status_report(self):
        """Sends a Telegram update every 2 minutes."""
        while True:
            await asyncio.sleep(120) 
            current_total = self.start_balance + self.session_profit
            status_msg = (
                "📊 *R10 Sniper: 2-Min Report*\n"
                f"💰 Total: `${current_total:.2f}`\n"
                f"📈 Session P/L: `${self.session_profit:.2f}`\n"
                f"👻 Shadow: `{self.shadow_wins}W - {self.shadow_losses}L`\n"
                f"⚡ AI Conf: `{self.last_prob:.2%}`"
            )
            await self.send_telegram_msg(status_msg)

    async def handle_tick(self, msg, api):
        if not msg or 'tick' not in msg: return
        price = msg['tick']['quote']
        self.tick_count += 1
        
        # 1. Update Data Buffer
        new_row = pd.DataFrame([{'price': price}])
        self.buffer = pd.concat([self.buffer, new_row], ignore_index=True).tail(50)
        if len(self.buffer) < 20: return

        # 2. Technical Analysis (Triple Confluence)
        prices = self.buffer['price']
        
        # RSI
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).tail(14).mean()
        loss = (-delta.where(delta < 0, 0)).tail(14).mean()
        rs = gain / loss if loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        
        # SMA Trend
        sma_5 = prices.rolling(5).mean().iloc[-1]
        sma_10 = prices.rolling(10).mean().iloc[-1]

        # 3. AI Inference
        feat = pd.DataFrame([{
            'returns': prices.pct_change().iloc[-1],
            'volatility': prices.pct_change().rolling(14).std().iloc[-1],
            'rsi': rsi,
            'momentum': prices.iloc[-1] - prices.iloc[-5],
            'sma_signal': 1 if sma_5 > sma_10 else 0
        }]).fillna(0)

        self.last_prob = self.model.predict_proba(feat)[0][1]

        # 4. SNIPER LOGIC (Win Optimization)
        if not self.current_contract:
            # CALL: AI Conf > 94% AND SMA Trending Up AND Not Overbought
            if self.last_prob > 0.94 and sma_5 > sma_10 and rsi < 65:
                await self.place_trade(api, "CALL", self.last_prob)
            
            # PUT: AI Conf < 6% AND SMA Trending Down AND Not Oversold
            elif self.last_prob < 0.06 and sma_5 < sma_10 and rsi > 35:
                await self.place_trade(api, "PUT", self.last_prob)
            
            # Shadow Monitoring (Record high confidence skips)
            elif self.last_prob > 0.90 or self.last_prob < 0.10:
                asyncio.create_task(self.shadow_monitor(price, "CALL" if self.last_prob > 0.5 else "PUT"))

    async def shadow_monitor(self, entry_price, direction):
        """Simulates the trade in the background to verify filters."""
        start_count = self.tick_count
        while self.tick_count < start_count + 10: # Check at 10 ticks
            await asyncio.sleep(1)
        
        exit_price = self.buffer['price'].iloc[-1]
        win = (exit_price > entry_price) if direction == "CALL" else (exit_price < entry_price)
        
        if win: self.shadow_wins += 1
        else: self.shadow_losses += 1

    async def place_trade(self, api, direction, prob):
        if self.current_contract or self.session_profit <= self.max_session_loss: 
            return

        stake = 1.0
        trade_params = {
            "buy": 1, "price": stake,
            "parameters": {
                "amount": stake, 
                "basis": "stake", 
                "contract_type": direction,
                "currency": "USD", 
                "duration": 10, # Optimized duration for R_10 stability
                "duration_unit": "t", 
                "symbol": "R_10"
            }
        }
        try:
            resp = await api.buy(trade_params)
            if 'error' in resp:
                logging.warning(f"Trade Rejected: {resp['error']['message']}")
                return
            
            self.current_contract = resp['buy']['contract_id']
            await self.send_telegram_msg(f"🎯 *SNIPER ENTRY*\nDir: {direction}\nConf: {prob:.2%}")
            asyncio.create_task(self.monitor_contract(api, self.current_contract))
        except Exception as e:
            logging.error(f"Execution Error: {e}")

    async def monitor_contract(self, api, contract_id):
        finished = asyncio.get_event_loop().create_future()
        
        async def handle_update(msg):
            contract = msg.get('proposal_open_contract', {})
            if contract.get('contract_id') == contract_id and contract.get('is_sold'):
                profit = float(contract.get('profit', 0))
                self.session_profit += profit
                status = "✅ WIN" if profit > 0 else "❌ LOSS"
                
                await self.send_telegram_msg(f"🏁 *TRADE RESULT*\n{status}\nProfit: `${profit}`\nSession: `${self.session_profit:.2f}`")
                if not finished.done(): finished.set_result(True)

        status_obs = await api.subscribe({"proposal_open_contract": 1, "contract_id": contract_id})
        subscription = status_obs.subscribe(handle_update)
        await finished
        subscription.dispose()
        self.current_contract = None

    async def send_telegram_msg(self, message):
        url = f"https://api.telegram.org/bot{self.tg_token}/sendMessage"
        for attempt in range(3):
            async with httpx.AsyncClient(timeout=15.0) as client:
                try:
                    resp = await client.post(url, json={"chat_id": self.chat_id, "text": message, "parse_mode": "Markdown"})
                    if resp.status_code == 200: return
                except Exception:
                    await asyncio.sleep(5)

if __name__ == "__main__":
    bot = ProductionEngine()
    try:
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        logging.info("Bot stopped manually.")