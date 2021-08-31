from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

def X11(time_series, period=12, plot=False):
    decomposition = seasonal_decompose(x=time_series, period=6, extrapolate_trend='freq')
    if plot:
        decomposition.plot()
        plt.show()
    return decomposition