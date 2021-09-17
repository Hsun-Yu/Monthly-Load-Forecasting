# %%
from os import name
from sqlalchemy.engine.base import Engine
import pandas as pd
import datetime

def get_monthly_load(db_conn: Engine, from_date: datetime.datetime, to_date: datetime.datetime, area=0) -> pd.DataFrame:
    sql = """
        SELECT [date]
            ,[year]
            ,[month]
            ,[area]
            ,[value]
        FROM [Monthly_Power_Load]
        WHERE [date] between '{0}' and '{1}' and [area]='{2}'
        ORDER BY [date]
    """.format(from_date.strftime("%Y-%m-01"), to_date.strftime("%Y-%m-01"), area)
    return pd.read_sql(sql, con=db_conn)

def upload_monthly_load(db_conn: Engine, data: pd.DataFrame):
    data = data.astype({
        'year': int,
        'month': int,
        'type_id': int,
        'value': float
    })
    print(data)
    print(data.info())
    print(data.head(n=100))
    data.to_sql(name='monthly_load', con=db_conn, if_exists='append', index=False)

def upload_population(db_conn: Engine, data: pd.DataFrame):
    data = data.astype({
        'year': int,
        'month': int,
        'location_id': int,
        'value': float
    })
    print(data)
    print(data.info())
    print(data.head(n=100))
    data.to_sql(name='population', con=db_conn, if_exists='append', index=False)

def upload_migrants(db_conn: Engine, data: pd.DataFrame):
    data = data.astype({
        'year': int,
        'location_id': int,
        'in': float,
        'out': float
    })
    print(data)
    print(data.info())
    print(data.head(n=100))
    data.to_sql(name='migrants', con=db_conn, if_exists='append', index=False)