import asyncio
import pandas as pd
import numpy as np
import joblib
import logging
from xgboost import XGBClassifier
from deriv_api import DerivAPI
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


async def train_new_model(api_token, app_id):
    api = DerivAPI(app_id=app_id)
    try:
        logging.info("Connecting to Deriv for R_10 training data...")
        await api.authorize(api_token)

        history = await api.ticks_history({
            "ticks_history": "R_10",  # Switched to Volatility 10
            "adjust_start_time": 1,
            "count": 50000,
            "end": "latest",
            "style": "ticks"
        })

        df = pd.DataFrame({'price': history['history']['prices']})

        logging.info(
            "Engineering features: Returns, Volatility, RSI, Momentum, and SMA...")

        # 1. Returns & Volatility
        df['returns'] = df['price'].pct_change()
        df['volatility'] = df['returns'].rolling(14).std()

        # 2. RSI
        df['rsi'] = calculate_rsi(df['price'], period=14)

        # 3. Momentum (5-tick)
        df['momentum'] = df['price'].diff(5)

        # 4. SMA Crossover (New Trend Feature)
        df['sma_5'] = df['price'].rolling(5).mean()
        df['sma_10'] = df['price'].rolling(10).mean()
        # 1 if short trend is above long trend, else 0
        df['sma_signal'] = (df['sma_5'] > df['sma_10']).astype(int)

        # Target: Price higher in 5 ticks?
        df['target'] = (df['price'].shift(-5) > df['price']).astype(int)

        df = df.dropna().copy()

        # UPDATED FEATURE LIST
        features = ['returns', 'volatility', 'rsi', 'momentum', 'sma_signal']
        X = df[features]
        y = df['target']

        logging.info(f"Training XGBoost on {len(X)} samples...")
        model = XGBClassifier(
            n_estimators=200,  # Increased for the extra feature
            max_depth=6,
            learning_rate=0.05,
            objective='binary:logistic',
            random_state=42
        )
        model.fit(X, y)

        joblib.dump(model, 'trading_model.joblib')
        logging.info("SUCCESS: R_10 Model with SMA saved.")

    except Exception as e:
        logging.error(f"Training failed: {e}")
    finally:
        await api.disconnect()

if __name__ == "__main__":
    MY_TOKEN = os.getenv('DERIV_TOKEN')
    MY_APP_ID = os.getenv('DERIV_APP_ID')
    asyncio.run(train_new_model(MY_TOKEN, MY_APP_ID))
