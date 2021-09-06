# %%
from datetime import datetime
from matplotlib.pyplot import plot
import method as md
import sqlalchemy
import pandas as pd
from copy import deepcopy
import os
from dotenv import load_dotenv

class Experimenter:
    def run(self, pred_year_from: int, pred_year_to: int, training_year_range=5):
        for y in range(pred_year_from, pred_year_to + 1):
            m = self.method
            m.performance_test(pred_from_date=datetime(y, 1, 1), 
                                pred_to_date=datetime(y, 12, 1), 
                                train_from_date=datetime(y-1-training_year_range, 1, 1), 
                                train_to_date=datetime(y-1, 12, 1))

    def __init__(self, db_conn, method:md.MidTermForecastMethod) -> None:
        self.db_conn = db_conn
        self.method = method


def run_seasonal_ARIMA_experiment(db_conn, pred_year_from:int, pred_year_to:int):
    for ar_lag in range(4):
        for ma_lag in range(4):
            for diff in range(3):
                method = md.SeasonalARIMA(db_conn=db_conn, AR_lag=ar_lag, MA_lag=ma_lag, differencing_order=diff, seasonal_period=12)
                ex = Experimenter(db_conn=db_conn, method=method)
                ex.run(pred_year_from=pred_year_from, pred_year_to=pred_year_to)

def run_trend_residual_ARIMA_experiment(db_conn, pred_year_from:int, pred_year_to:int):
    for t_ar_lag in range(4):
        for t_ma_lag in range(4):
            for t_diff in range(3):
                for r_ar_lag in range(4):
                    for r_ma_lag in range(4):
                        for r_diff in range(3):
                            method = md.TrendResidualARIMA(db_conn=db_conn, trend_AR_lag=t_ar_lag, trend_MA_lag=t_ma_lag, trend_differencing_order=t_diff, 
                                                                            resid_AR_lag=r_ar_lag, resid_MA_lag=r_ma_lag, resid_differencing_order=r_diff, seasonal_period=12)
                            ex = Experimenter(db_conn=db_conn, method=method)
                            ex.run(pred_year_from=pred_year_from, pred_year_to=pred_year_to)


if __name__ == "__main__":
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
    # run_seasonal_ARIMA_experiment(db_conn=conn, pred_year_from=2016, pred_year_to=2021)
    # run_trend_residual_ARIMA_experiment(db_conn=conn, pred_year_from=2016, pred_year_to=2021)
    # method = md.TrendResidualARIMA(db_conn=conn, trend_AR_lag=2, trend_MA_lag=2, trend_differencing_order=1, 
    #                                             resid_AR_lag=2, resid_MA_lag=2, resid_differencing_order=1, seasonal_period=12)
    # method.performance_test(pred_from_date=datetime(2021, 1, 1), pred_to_date=datetime(2021, 12, 1), 
    #                         train_from_date=datetime(2012, 1, 1), train_to_date=datetime(2017, 12, 1))

    # method = md.SeasonalARIMA(db_conn=conn, AR_lag=12, MA_lag=1, differencing_order=1, seasonal_period=12)
    # ex = Experimenter(db_conn=conn, method=method)
    # ex.run(pred_year_from=2016, pred_year_to=2019)

    method = md.JustAverage(db_conn=conn)
    ex = Experimenter(db_conn=conn, method=method)
    ex.run(pred_year_from=2016, pred_year_to=2021)


# %%
