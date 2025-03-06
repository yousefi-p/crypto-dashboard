import psycopg2
from psycopg2 import sql

def connect_db():
    conn = psycopg2.connect(
        dbname="binancedb",
        user="postgres",
        password="P@ssw0rd",
        host="localhost",
        port="5432"
    )
    return conn