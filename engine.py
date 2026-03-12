import asyncio
import joblib
import pandas as pd
import numpy as np
import logging
import httpx  # Lightweight for async requests
import sys  # Added for immediate flush
from deriv_api import DerivAPI
import os
from dotenv import load_dotenv

load_dotenv()

# Business-Grade Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('business_log.txt', mode='a'),  # 'a' for append
        logging.StreamHandler(sys.stdout)                 # Force to terminal
    ]
)


class ProductionEngine:
    def __init__(self):
        self.token = os.getenv('DERIV_TOKEN')
        self.app_id = os.getenv('DERIV_APP_ID')
        self.tg_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')

        # Verify critical variables are present
        if not all([self.token, self.app_id, self.tg_token, self.chat_id]):
            print("CRITICAL ERROR: Missing environment variables in .env file")
            exit(1)
        try:
            self.model = joblib.load('trading_model.joblib')
            print("Model loaded successfully.")
        except Exception as e:
            print(f"CRITICAL: Could not load model file: {e}")
            exit()
        self.model = joblib.load('trading_model.joblib')
        self.buffer = pd.DataFrame()
        self.current_contract = None
        self.session_profit = 0.0
        self.max_loss = -3.0

    async def start(self):
        print(f"Attempting to connect with AppID: {self.app_id}...")
        while True:
            api = None
            try:
                api = DerivAPI(app_id=self.app_id)

                # Test the connection before authorizing
                print("Sending Authorize request...")
                authorize = await api.authorize(self.token)

                balance = authorize['authorize']['balance']
                print(f"SUCCESS! Connected to Account. Balance: ${balance}")
                logging.info(f"Session started. Balance: ${balance}")

                # 2. Setup the Reactive Subscription
                print("Subscribing to R_10 ticks...")
                tick_observable = await api.subscribe({'ticks': 'R_10'})

                # Attach the handler
                tick_observable.subscribe(
                    lambda msg: asyncio.create_task(self.handle_tick(msg, api))
                )

                print("--- BOT IS NOW LIVE AND LISTENING ---")

                while True:
                    await asyncio.sleep(10)
                    # This print confirms the script hasn't frozen
                    print("Heartbeat: Monitoring stream...", end='\r')

            except Exception as e:
                print(f"\nCONNECTION ERROR: {e}")
                logging.error(f"Stream Error: {e}")
                if api:
                    try:
                        await api.disconnect()
                    except:
                        pass
                await asyncio.sleep(10)

    async def handle_tick(self, msg, api):
        if not msg or 'tick' not in msg:
            return

        price = msg['tick']['quote']

        # 1. Buffer Management
        new_row = pd.DataFrame([{'price': price}])
        self.buffer = pd.concat([self.buffer, new_row],
                                ignore_index=True).tail(50)

        # Wait for buffer to fill
        if len(self.buffer) < 20:
            print(
                f"Tick: {price} | Filling Buffer: {len(self.buffer)}/20...", end='\r')
            return

        # 2. Feature Engineering
        latest_prices = self.buffer['price']
        delta = latest_prices.diff()
        gain = (delta.where(delta > 0, 0)).tail(14).mean()
        loss = (-delta.where(delta < 0, 0)).tail(14).mean()
        rs = gain / loss if loss != 0 else 0
        current_rsi = 100 - (100 / (1 + rs))

        # Momentum calculations
        # For trade confirmation
        mom_short = latest_prices.iloc[-1] - latest_prices.iloc[-3]
        mom_long = latest_prices.iloc[-1] - \
            latest_prices.iloc[-5]   # For AI feature

        # Calculate 5-period and 10-period SMA
        sma_5 = latest_prices.rolling(5).mean().iloc[-1]
        sma_10 = latest_prices.rolling(10).mean().iloc[-1]

        # New Feature: Is it a bullish or bearish cross?
        sma_signal = 1 if sma_5 > sma_10 else 0

        feat = pd.DataFrame([{
            'returns': latest_prices.pct_change().iloc[-1],
            'volatility': latest_prices.pct_change().rolling(14).std().iloc[-1],
            'rsi': current_rsi,
            'momentum': mom_long,
            'sma_signal': sma_signal
        }])

        try:
            # 3. AI Inference
            prediction_data = feat.fillna(0)
            prob = self.model.predict_proba(prediction_data)[0][1]

            # --- LIVE DASHBOARD (This shows you why it's NOT trading) ---
            sys.stdout.write(
                f"\rPrice: {price:.2f} | Prob: {prob:.2%} | RSI: {current_rsi:.1f} | Mom: {mom_short:+.2f} | Active: {self.current_contract is not None}")
            sys.stdout.flush()

            if not self.current_contract:
                # 1. Potential CALL Signal
                if prob > 0.86:
                    if current_rsi >= 60 or mom_short <= 0.10:
                        reason = "RSI High" if current_rsi >= 60 else "Weak Mom"
                        logging.info(f" [SHADOW] Skipping CALL: {reason}")
                        # Start Shadow Monitor to see if we WERE right to skip
                        asyncio.create_task(self.shadow_monitor(price, "CALL"))
                    else:
                        await self.place_trade(api, "CALL", prob)

                # 2. Potential PUT Signal
                elif prob < 0.14:
                    if current_rsi <= 40 or mom_short >= -0.10:
                        reason = "RSI Low" if current_rsi <= 40 else "Weak Mom"
                        logging.info(f" [SHADOW] Skipping PUT: {reason}")
                        asyncio.create_task(self.shadow_monitor(price, "PUT"))
                    else:
                        await self.place_trade(api, "PUT", prob)
            # # 4. Refined Logic Check (The "Shielded" Version)
            # # Check for CALL: Higher confidence + tighter RSI window
            # if prob > 0.82 and 40 < current_rsi < 68:
            #     if mom_short > 0.05 and mom_long > 0:  # Both agree + minimum move
            #         if not self.current_contract:
            #             await self.place_trade(api, "CALL", prob)

            # # Check for PUT: Higher confidence + tighter RSI window
            # elif prob < 0.18 and 32 < current_rsi < 60:
            #     if mom_short < -0.05 and mom_long < 0:
            #         if not self.current_contract:
            #             await self.place_trade(api, "PUT", prob)
                else:
                    pass

        except Exception as e:
            logging.error(f"AI Inference Error: {e}")

    async def shadow_monitor(self, entry_price, direction):
        """Waits 5 ticks and checks if the skipped trade would have won."""
        start_buffer_len = len(self.buffer)
        # Wait until 5 more ticks have passed in the buffer
        while len(self.buffer) < start_buffer_len + 5:
            await asyncio.sleep(1)

        exit_price = self.buffer['price'].iloc[-1]
        was_correct_to_skip = False

        if direction == "CALL":
            win = exit_price > entry_price
        else:
            win = exit_price < entry_price

        result_str = "WIN" if win else "LOSS"
        # If it would have been a LOSS, our skip was "CORRECT"
        advice = "GOOD SKIP" if not win else "MISSED PROFIT"

        logging.info(
            f" [RESULT] Shadow {direction} would have {result_str}. Verdict: {advice}")

    async def place_trade(self, api, direction, prob):
        if self.current_contract:
            return

        # 1. Increase stake to $1.00 (Standard Minimum)
        # 2. Add 'proposal' check to see if the trade is even allowed
        stake_amount = 1.0

        logging.info(
            f"==> SIGNAL: {direction} (Confidence: {prob:.2%}) | Stake: ${stake_amount}")
        print(
            f"==> SIGNAL: {direction} (Confidence: {prob:.2%}) | Stake: ${stake_amount}")

        trade_params = {
            "buy": 1,
            "price": stake_amount,
            "parameters": {
                "amount": stake_amount,
                "basis": "stake",
                "contract_type": direction,
                "currency": "USD",
                "duration": 5,
                "duration_unit": "t",
                "symbol": "R_10"
            }
        }

        try:
            response = await api.buy(trade_params)

            # If it still fails, the error message will tell us the exact limit
            if 'error' in response:
                error_msg = response['error'].get('message', 'Unknown Error')
                logging.error(f"Execution Rejected: {error_msg}")

                # If the error is about the stake being too LOW, we'll know here
                return

            self.current_contract = response['buy']['contract_id']
            logging.info(f"Trade Success! ID: {self.current_contract}")
            print(f"Trade Success! ID: {self.current_contract}")
            # Inside place_trade
            await self.send_telegram_msg(f"🚀 *TRADE PLACED*\nDirection: {direction}\nProb: {prob:.2%}")
            asyncio.create_task(self.monitor_contract(
                api, self.current_contract))

        except Exception as e:
            logging.error(f"Network error during buy: {e}")

    async def monitor_contract(self, api, contract_id):
        """
        Watches the contract using a reactive subscription.
        """
        # We use a Future to 'wait' for the reactive stream to finish
        finished_signal = asyncio.get_event_loop().create_future()

        async def handle_update(msg):
            try:
                contract = msg.get('proposal_open_contract', {})
                # Only process if this is the contract we are looking for
                if contract.get('contract_id') == contract_id and contract.get('is_sold'):
                    status = contract.get('status', 'unknown')
                    profit = contract.get('profit', 0)
                    self.session_profit += float(profit)
                    logging.info(f"Session P/L: ${self.session_profit:.2f}")

                    if self.session_profit <= self.max_loss:
                        logging.critical(
                            "Stop Loss hit. Shutting down to protect balance.")
                        print(
                            "Stop Loss hit. Shutting down to protect balance.")
                        exit()
                    logging.info(
                        f"--- TRADE FINISHED: {status.upper()} (Profit: ${profit}) ---")
                    print(
                        f"--- TRADE FINISHED: {status.upper()} (Profit: ${profit}) ---")
                    await self.send_telegram_msg(f"🏁 *TRADE RESULT*\nStatus: {status}\nProfit: ${profit}\nBalance: ${self.session_profit}")

                    # Signal that we are done
                    if not finished_signal.done():
                        finished_signal.set_result(True)
            except Exception as e:
                logging.error(f"Error in monitor callback: {e}")

        try:
            # 1. Get the Observable for this specific contract
            status_observable = await api.subscribe({"proposal_open_contract": 1, "contract_id": contract_id})

            # 2. Attach the handler
            subscription = status_observable.subscribe(handle_update)

            # 3. Wait here until the 'finished_signal' is triggered by the handler
            await finished_signal

            # 4. Clean up the subscription
            subscription.dispose()

        except Exception as e:
            logging.error(f"Monitor Setup Error: {e}")
        finally:
            # Reset state so the bot can trade again
            self.current_contract = None

    async def check_balance_safety(self, auth_data):
        balance = auth_data['authorize']['balance']
        logging.info(f"Connected to Account. Balance: ${balance}")
        if balance < 1.0:
            logging.critical(
                "Balance too low for safe trading. Shutting down.")
            exit()

    async def send_telegram_msg(self, message):
        token = self.tg_token
        chat_id = self.chat_id
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id,
                   "text": message, "parse_mode": "Markdown"}

        async with httpx.AsyncClient() as client:
            try:
                await client.post(url, json=payload)
            except Exception as e:
                logging.error(f"Telegram Failed: {e}")


if __name__ == "__main__":
    bot = ProductionEngine()
    asyncio.run(bot.start())
