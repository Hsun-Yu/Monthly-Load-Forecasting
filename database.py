# %%
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