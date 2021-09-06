# %%
from datetime import datetime
from statsmodels.base.model import Model
import os
import database as db
import time_series as ts
import sqlalchemy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error
from dotenv import load_dotenv

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
# %%
# Original as input
train_load = db.get_monthly_load(db_conn=conn, from_date=datetime(2010, 1, 1), to_date=datetime(2018, 12, 1))

# target_load = target_load.set_index(["date"])
x = pd.DataFrame({'value': train_load['value'].values}, index=train_load['date'])

train_decomp = ts.X11(time_series=x)
train_trend = train_decomp.trend.fillna(0)
train_seasonal = train_decomp.seasonal.fillna(0)
train_residual = train_decomp.resid.fillna(0)

x = pd.DataFrame({'value': train_trend.values + train_residual.values}, index=train_load['date'])    
result = adfuller(x.value.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

result = adfuller(x.value.diff().dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

result = adfuller(x.value.diff().dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

# Original Series
fig, axes = plt.subplots(3, 3, sharex='col')
axes[0, 0].plot(x.value); axes[0, 0].set_title('Original Series')
plot_acf(x, ax=axes[0, 1])
plot_pacf(x, ax=axes[0, 2])
# 1st Differencing
axes[1, 0].plot(x.value.diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(x.value.diff().dropna(), ax=axes[1, 1])
plot_pacf(x.value.diff().dropna(), ax=axes[1, 2])
# 2nd Differencing
axes[2, 0].plot(x.value.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(x.value.diff().diff().dropna(), ax=axes[2, 1])
plot_pacf(x.value.diff().dropna(), ax=axes[2, 2])
fig.set_size_inches(18.5, 10.5)
plt.show()
# %%
# residual and trend predict
# for year in range(2015, 2019):
train_load = db.get_monthly_load(db_conn=conn, from_date=datetime(2010, 1, 1), to_date=datetime(2018, 12, 1))
# target_load = target_load.set_index(["date"])
x = pd.DataFrame({'value': train_load['value'].values}, index=train_load['date'])

train_decomp = ts.X11(time_series=x)
train_trend = train_decomp.trend.fillna(0)
train_seasonal = train_decomp.seasonal.fillna(0)
train_residual = train_decomp.resid.fillna(0)
t = pd.DataFrame({'value': train_trend.values}, index=train_load['date'])
result = adfuller(train_residual)
print('Residual ADF Statistic: %f' % result[0])
print('Residual p-value: %f' % result[1])

result = adfuller(train_residual.diff().dropna())
print('Residual diff ADF Statistic: %f' % result[0])
print('Residual diff p-value: %f' % result[1])

result = adfuller(train_trend)
print('Trend ADF Statistic: %f' % result[0])
print('Trend p-value: %f' % result[1])

result = adfuller(train_trend.diff().dropna())
print('Trend diff ADF Statistic: %f' % result[0])
print('Trend diff p-value: %f' % result[1])
# Original Series
fig, axes = plt.subplots(4, 3, sharex='col')
axes[0, 0].plot(train_residual); axes[0, 0].set_title('Residual Series')
plot_acf(train_residual, ax=axes[0, 1])
plot_pacf(train_residual, ax=axes[0, 2])

axes[1, 0].plot(train_residual.diff().dropna()); axes[1, 0].set_title('Residual diff Series')
plot_acf(train_residual.diff().dropna(), ax=axes[1, 1])
plot_pacf(train_residual.diff().dropna(), ax=axes[1, 2])

axes[2, 0].plot(train_trend); axes[2, 0].set_title('Trend Series')
plot_acf(train_trend, ax=axes[2, 1])
plot_pacf(train_trend, ax=axes[2, 2])

axes[3, 0].plot(train_trend.diff().dropna()); axes[3, 0].set_title('Trend diff Series')
plot_acf(train_trend.diff().dropna(), ax=axes[3, 1])
plot_pacf(train_trend.diff().dropna(), ax=axes[3, 2])

fig.set_size_inches(18.5, 10.5)
plt.show()
# %%
