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

data = pd.read_csv('data/population.csv')
population = pd.DataFrame(columns=['year', 'month', 'location_id', 'value'])
for m in range(1, 13):
    d = pd.DataFrame({'year': data['year'], 'month': str(m), 'location_id': data['location_id'], 'value': data[str(m)]})
    population = population.append(d)

population = population[population['value'].notnull()]
db.upload_population(db_conn=conn, data=population)

print('done')