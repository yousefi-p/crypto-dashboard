import requests
import pandas as pd
from datetime import datetime
from database_connection import connect_db


def get_historical_data(symbol, interval, limit):

    # Binance API endpoint for historical Kline (candlestick) data
    url = "https://api.binance.com/api/v3/klines"

    # Parameters for the request
    params = {
        "symbol": symbol.upper() + "USDT",          # Trading pair (e.g., Bitcoin to USD)
        "interval": interval,            # Interval (daily data)
        "limit": limit                   # Number of data points
    }
    try:
        # Send GET request to Binance API
        response = requests.get(url, params=params)
        data = response.json()

        # Extract relevant data and convert to DataFrame
        historical_data = []
        for candle in data:
            historical_data.append({
                "date": datetime.fromtimestamp(candle[0] / 1000).strftime('%Y-%m-%d'),
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume": float(candle[5])
            })

        df = pd.DataFrame(historical_data)
        # print(df.head())  # Show the first few rows

        # Connect to PostgreSQL database
        conn = connect_db()
        cur = conn.cursor()

        # Create table if not exists
        cur.execute(f'''
            CREATE TABLE IF NOT EXISTS {symbol.lower().strip()}_usdt_data (
                date DATE PRIMARY KEY,
                open FLOAT,
                high FLOAT,
                low FLOAT,
                close FLOAT,
                volume FLOAT
            )
        ''')

        # Insert data into table
        for _, row in df.iterrows():
            cur.execute(f'''
                INSERT INTO {symbol.lower().strip()}_usdt_data (date, open, high, low, close, volume)'''+
                '''VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (date) DO NOTHING
            ''',(row['date'], row['open'], row['high'], row['low'], row['close'], row['volume']))

        # Commit and close connection   
        conn.commit()
        cur.close()
        conn.close()
        return (df.head())
    except Exception as e:
        print(f"An error occurred: {e}")
