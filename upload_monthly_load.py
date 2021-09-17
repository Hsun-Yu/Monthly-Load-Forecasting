# %%
import pandas as pd
import os
import database as db
from dotenv import load_dotenv
import sqlalchemy

load_dotenv()
username = os.getenv("load_db_username")
password = os.getenv("load_db_password")
host = os.getenv("load_db_host")
database = os.getenv("load_db_dbname")

conn = sqlalchemy.engine.URL.create(
    "mssql+pymssql",
    username=username,
    password=password,
    host=host,
    database=database,
)
conn = sqlalchemy.create_engine(conn, echo=False)

data = pd.read_csv('data/monthly_load.csv')
monthly_load = pd.DataFrame(columns=['year', 'month', 'type_id', 'value'])
d = pd.DataFrame({'year': data['年'], 'month': data['月'], 'type_id': '1', 'value': data['住宅部門售電量(度)']})
monthly_load = monthly_load.append(d)
d = pd.DataFrame({'year': data['年'], 'month': data['月'], 'type_id': '2', 'value': data['服務業部門(含包燈)(度)']})
monthly_load = monthly_load.append(d)
d = pd.DataFrame({'year': data['年'], 'month': data['月'], 'type_id': '3', 'value': data['農林漁牧售電量(度)']})
monthly_load = monthly_load.append(d)
d = pd.DataFrame({'year': data['年'], 'month': data['月'], 'type_id': '4', 'value': data['工業部門售電量(度)']})
monthly_load = monthly_load.append(d)
d = pd.DataFrame({'year': data['年'], 'month': data['月'], 'type_id': '5', 'value': data['合計售電量(度)']})
monthly_load = monthly_load.append(d)

db.upload_monthly_load(db_conn=conn, data=monthly_load)

print('done')