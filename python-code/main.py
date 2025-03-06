from get_data import get_historical_data
from database_connection import connect_db
from prediction_model import prediction_function



crypto_currencies_name = ["TRX","BTC","ETH","XRP","SOL","DOGE"]

if __name__ == "__main__":

    for currency in crypto_currencies_name:
        prediction_function(currency)


    