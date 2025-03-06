import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import psycopg2
from database_connection import connect_db


def prediction_function(symbol):

    # Load data from PostgreSQL
    def load_data(symbol):
        conn = connect_db()
        query = f"SELECT * FROM {symbol.lower().strip()}_usdt_data"
        df = pd.read_sql(query, conn)
        conn.close()
        print("load data successful")
        return df

    # Feature engineering
    def add_features(df):
        df['price_change'] = df['close'].pct_change()
        df['vol_change'] = df['volume'].pct_change()
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['rsi'] = calculate_rsi(df['close'])
        df.dropna(inplace=True)
        print("add features successful")
        return df

    # RSI calculation
    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        print("calculate rsi successful")
        return 100 - (100 / (1 + rs))

    # Labeling: 1 = Buy, -1 = Sell, 0 = Hold
    def label_data(df):
        df['signal'] = np.where(df['price_change'] > 0.01, 1, np.where(df['price_change'] < -0.01, -1, 0))
        print("label data successful")
        return df

    # Train model
    def train_model(df):
        X = df[['price_change', 'vol_change', 'sma_5', 'sma_10', 'rsi']]
        y = df['signal']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(classification_report(y_test, y_pred))
        print("train model successful") 
        return model

    # Store predictions in database
    def store_predictions(df, predictions, symbol):
        conn = connect_db()
        cur = conn.cursor()

        cur.execute(f'''
            CREATE TABLE IF NOT EXISTS {symbol.lower().strip()}_predictions (
                date DATE PRIMARY KEY,
                prediction INT
            )
        ''')

        for date, pred in zip(df['date'], predictions):
            cur.execute(f'''
                INSERT INTO {symbol.lower().strip()}_predictions (date, prediction)
                VALUES (%s, %s)
                ON CONFLICT (date) DO UPDATE SET prediction = EXCLUDED.prediction
            ''', (date, int(pred)))

        conn.commit()
        cur.close()
        conn.close()
        print("store predictions successful")



    data = load_data(symbol)
    data = add_features(data)
    data = label_data(data)
    model = train_model(data)
    predictions = model.predict(data[['price_change', 'vol_change', 'sma_5', 'sma_10', 'rsi']])
    store_predictions(data, predictions, symbol)
    print("Predictions stored successfully!")