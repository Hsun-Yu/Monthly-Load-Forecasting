# %%
from datetime import datetime
from matplotlib.pyplot import plot
from statsmodels.tsa.statespace.tools import diff
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

def run_seasonal_ARIMA_experiment(db_conn, pred_year_from=2016, pred_year_to=2021):
    for ar_lag in range(4):
        for ma_lag in range(4):
            for diff in range(3):
                for s_ar_lag in range(4):
                    for s_ma_lag in range(4):
                        for s_diff in range(3):
                            method = md.SeasonalARIMA(db_conn=db_conn, AR_lag=ar_lag, MA_lag=ma_lag, differencing_order=diff, 
                                                                        seasonal_AR_lag=s_ar_lag, seasonal_MA_lag=s_ma_lag, seasonal_differencing_order=s_diff, seasonal_period=12)
                            ex = Experimenter(db_conn=db_conn, method=method)
                            ex.run(pred_year_from=pred_year_from, pred_year_to=pred_year_to)

def run_trend_residual_mix_ARIMA_experiment(db_conn, pred_year_from=2016, pred_year_to=2021):
    for ar_lag in range(4):
        for ma_lag in range(4):
            for diff in range(3):
                method = md.TrendResidualMixARIMA(db_conn=db_conn, AR_lag=ar_lag, MA_lag=ma_lag, differencing_order=diff, seasonal_period=12)
                ex = Experimenter(db_conn=db_conn, method=method)
                ex.run(pred_year_from=pred_year_from, pred_year_to=pred_year_to)

def run_trend_residual_ARIMA_experiment(db_conn, pred_year_from=2016, pred_year_to=2021):
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

def run_just_average(db_conn, pred_year_from=2016, pred_year_to=2021):
    method = md.JustAverage(db_conn=conn)
    ex = Experimenter(db_conn=conn, method=method)
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
    run_seasonal_ARIMA_experiment(db_conn=conn)


# %%
