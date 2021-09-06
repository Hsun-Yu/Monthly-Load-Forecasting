from math import isnan
import mlflow
import numpy as np
import database as db
import pandas as pd
import time_series as ts
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
from dotenv import load_dotenv
import os

class MidTermForecastMethod:
    def get_training_data(self, from_date: datetime, to_date: datetime):
        pass

    def get_testing_input(self, from_date: datetime, to_date: datetime):
        pass

    def train(self, from_date: datetime, to_date: datetime):
        mlflow.log_param("Training set from", from_date.strftime("%Y-%m-%d"))
        mlflow.log_param("Training set to", to_date.strftime("%Y-%m-%d"))
        print("Training finished!!")
    
    def plot_result(self):
        pass
    
    def eval_metrics(self, actual, pred):
        ac = []
        pre = []
        for i in actual.index:
            if not np.isnan(actual["value"][i]) and not np.isnan(pred["value"][i]):
                ac.append(actual["value"][i])
                pre.append(pred["value"][i])
        if len(ac) == 0 and len(pre) == 0:
            return np.nan, np.nan, np.nan

        rmse = np.sqrt(mean_squared_error(ac, pre))
        mae = mean_absolute_error(ac, pre)
        mape = mean_absolute_percentage_error(ac, pre)
        return rmse, mae, mape

    def performance_test(self, pred, test_from_date: datetime, test_to_date: datetime, train_from_date: datetime, train_to_date: datetime):
        target_load = db.get_monthly_load(db_conn=self.db_conn, from_date=test_from_date, to_date=test_to_date)
        
        # resize target length to be same as pred
        padding_target = np.pad(target_load['value'].values, (0, pred.shape[0] - target_load.shape[0]), 'constant', constant_values=(None, None))
        target_load = pd.DataFrame({'value': padding_target}, index=pred.index)

        target_load.index = pd.to_datetime(target_load.index)
        rmse, mae, mape = self.eval_metrics(actual=target_load, pred=pred)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mape", mape)
        mlflow.log_param("testing_set_from", test_from_date.strftime("%Y-%m-%d"))
        mlflow.log_param("testing_set_to", test_to_date.strftime("%Y-%m-%d"))
        mlflow.log_param("training_set_from", train_from_date.strftime("%Y-%m-%d"))
        mlflow.log_param("training_set_to", train_to_date.strftime("%Y-%m-%d"))
        fig, ax = plt.subplots()
        ax.plot(target_load, label='actual')
        ax.plot(pred, label='forecast')
        ax.set_title("Final Forecast Usage")
        mlflow.log_figure(fig, "load.png")
        for m in range(1, 13):
            d = pred[pred.index.month == m]
            d_hat = target_load[target_load.index.month == m]
            rmse, mae, mape = self.eval_metrics(d_hat, d)        
            mlflow.log_metric(str(m) + "_rmse", rmse)
            mlflow.log_metric(str(m) + "_mae", mae)
            mlflow.log_metric(str(m) + "_mape", mape)
        dt = pd.DataFrame({'actual': target_load['value'].values, 'predict': pred['value'].values}, index=target_load.index)
        dt.to_csv("output.csv")
        mlflow.log_artifact("output.csv")
        print("Performance test finished!!")

    def create_new_run(self):
        # self.run = mlflow.start_run()
        pass

    def forecast(self):
        pass
    
    def __del__(self) -> None:
        pass
        # print("Finished run_id: {}".format(self.run.info.run_id))
        # mlflow.end_run()

    def __init__(self, db_conn) -> None:
        self.db_conn = db_conn

class SeasonalARIMA(MidTermForecastMethod):
    def get_training_data(self, from_date: datetime, to_date: datetime):
        load = db.get_monthly_load(db_conn=self.db_conn, from_date=from_date, to_date=to_date)
        training_data = pd.DataFrame({'value': load['value'].values}, index=load['date'])
        return training_data

    def get_testing_input(self, from_date: datetime, to_date: datetime):
        return None

    def train(self, from_date: datetime, to_date: datetime):
        training_data = self.get_training_data(from_date=from_date, to_date=to_date)
        self.model = ARIMA(training_data, seasonal_order=(self.AR_lag, self.differencing_order, self.MA_lag, self.seasonal_period), enforce_stationarity=False)
        self.model = self.model.fit()
        # super().train(from_date=from_date, to_date=to_date)

    def forecast(self, from_date: datetime, to_date: datetime):
        pred = self.model.predict(from_date, to_date)
        pred = pd.DataFrame({'value': pred}, index=pred.index)
        return pred

    def performance_test(self, pred_from_date: datetime, pred_to_date: datetime, train_from_date: datetime, train_to_date: datetime):
        tracking_uri = os.getenv("mlflow_tracking_uri")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("/seasonal_ARIMA")
        mlflow.start_run()
        mlflow.log_param("method_name", "Seasonal ARIMA")
        mlflow.log_param("AR_lag", self.AR_lag)
        mlflow.log_param("MA_lag", self.MA_lag)
        mlflow.log_param("differencing_order", self.differencing_order)
        mlflow.log_param("seasonal_period", self.seasonal_period)
        
        self.train(from_date=train_from_date, to_date=train_to_date)
        pred = self.forecast(from_date=pred_from_date, to_date=pred_to_date)
        super().performance_test(pred=pred, test_from_date=pred_from_date, test_to_date=pred_to_date, train_from_date=train_from_date, train_to_date=train_to_date)
        mlflow.end_run()

    def __init__(self, db_conn, AR_lag: int, MA_lag: int, differencing_order: int, seasonal_period=12) -> None:
        super().__init__(db_conn=db_conn)
        self.AR_lag = AR_lag
        self.MA_lag = MA_lag
        self.differencing_order = differencing_order
        self.seasonal_period = seasonal_period
        self.db_conn = db_conn
        # mlflow.statsmodels.autolog()

class TrendResidualARIMA(MidTermForecastMethod):
    def get_training_data(self, from_date: datetime, to_date: datetime):
        load = db.get_monthly_load(db_conn=self.db_conn, from_date=from_date, to_date=to_date)
        training_data = pd.DataFrame({'value': load['value'].values}, index=load['date'])
        return training_data

    def get_testing_input(self, from_date: datetime, to_date: datetime):
        return None

    def train(self, from_date: datetime, to_date: datetime):
        training_data = self.get_training_data(from_date=from_date, to_date=to_date)
        train_decomp = ts.X11(time_series=training_data, period=self.seasonal_period)
        self.trend = train_decomp.trend.fillna(0)
        self.seasonal = train_decomp.seasonal.fillna(0)
        self.residual = train_decomp.resid.fillna(0)
        self.trend_model = ARIMA(self.trend, order=(self.trend_AR_lag, self.trend_differencing_order, self.trend_MA_lag))
        self.trend_model = self.trend_model.fit()
        self.resid_model = ARIMA(self.residual, order=(self.resid_AR_lag, self.resid_differencing_order, self.resid_MA_lag))
        self.resid_model = self.resid_model.fit()

    def forecast(self, from_date: datetime, to_date: datetime):
        trend_pred = self.trend_model.predict(from_date, to_date)
        resid_pred = self.resid_model.predict(from_date, to_date)
        pred = pd.DataFrame({'value': resid_pred + trend_pred + self.seasonal[:12].values}, index=trend_pred.index)
        return pred

    def performance_test(self, pred_from_date: datetime, pred_to_date: datetime, train_from_date: datetime, train_to_date: datetime):
        load_dotenv()
        tracking_uri = os.getenv("mlflow_tracking_uri")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("/trend_residual_ARIMA")
        mlflow.start_run()
        mlflow.log_param("method_name", "Trend Residual ARIMA")
        mlflow.log_param("trend_AR_lag", self.trend_AR_lag)
        mlflow.log_param("trend_MA_lag", self.trend_MA_lag)
        mlflow.log_param("trend_differencing_order", self.trend_differencing_order)
        mlflow.log_param("resid_AR_lag", self.resid_AR_lag)
        mlflow.log_param("resid_MA_lag", self.resid_MA_lag)
        mlflow.log_param("resid_differencing_order", self.resid_differencing_order)
        mlflow.log_param("seasonal_period", self.seasonal_period)
        self.train(from_date=train_from_date, to_date=train_to_date)
        pred = self.forecast(from_date=pred_from_date, to_date=pred_to_date)
        super().performance_test(pred=pred, test_from_date=pred_from_date, test_to_date=pred_to_date, train_from_date=train_from_date, train_to_date=train_to_date)
        mlflow.end_run()

    # def create_new_run(self):
    #     self.__del__()
    #     self.__init__(db_conn=self.db_conn, AR_lag=self.AR_lag, MA_lag=self.MA_lag, differencing_order=self.differencing_order, seasonal_period=self.seasonal_period)


    def __init__(self, db_conn, trend_AR_lag: int, trend_MA_lag: int, trend_differencing_order: int, 
                                resid_AR_lag: int, resid_MA_lag: int, resid_differencing_order: int, seasonal_period: int) -> None:
        super().__init__(db_conn=db_conn)
        self.trend_AR_lag = trend_AR_lag
        self.trend_MA_lag = trend_MA_lag
        self.trend_differencing_order = trend_differencing_order
        self.resid_AR_lag = resid_AR_lag
        self.resid_MA_lag = resid_MA_lag
        self.resid_differencing_order = resid_differencing_order
        self.seasonal_period = seasonal_period
        
        self.db_conn = db_conn

class JustAverage(MidTermForecastMethod):
    def get_training_data(self, from_date: datetime, to_date: datetime):
        load = db.get_monthly_load(db_conn=self.db_conn, from_date=from_date, to_date=to_date)
        training_data = pd.DataFrame({'value': load['value'].values}, index=load['date'])
        return training_data

    def get_testing_input(self, from_date: datetime, to_date: datetime):
        return None

    def train(self, from_date: datetime, to_date: datetime):
        training_data = self.get_training_data(from_date=from_date, to_date=to_date)
        self.average = np.average(training_data["value"])

    def forecast(self, from_date: datetime, to_date: datetime):
        dates = pd.date_range(start=from_date, end=to_date, freq='MS')
        pred = np.full((len(dates)), self.average)
        pred = pd.DataFrame({'value': pred}, index=dates)
        return pred

    def performance_test(self, pred_from_date: datetime, pred_to_date: datetime, train_from_date: datetime, train_to_date: datetime):
        tracking_uri = os.getenv("mlflow_tracking_uri")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("/just_average")
        mlflow.start_run()
        mlflow.log_param("method_name", "Just Average")
        
        self.train(from_date=train_from_date, to_date=train_to_date)
        pred = self.forecast(from_date=pred_from_date, to_date=pred_to_date)
        super().performance_test(pred=pred, test_from_date=pred_from_date, test_to_date=pred_to_date, train_from_date=train_from_date, train_to_date=train_to_date)
        mlflow.end_run()

    def __init__(self, db_conn) -> None:
        super().__init__(db_conn)
