# %%
from numpy import isnan
import pandas as pd
import os
import database as db
from dotenv import load_dotenv
import sqlalchemy
import numpy as np

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

data = pd.read_csv('data/migrants.csv')
db.upload_migrants(db_conn=conn, data=data)

print('done')
# %%
