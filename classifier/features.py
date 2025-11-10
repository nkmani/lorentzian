import pandas as pd
import numpy as np

from ta.momentum import rsi, stoch_signal
from ta.trend import cci, adx, ema_indicator as ema, sma_indicator as sma
from sklearn.preprocessing import MinMaxScaler


def normalize(src: np.array, range_min=0, range_max=1) -> np.array:
    """
    function Rescales a source value with an unbounded range to a bounded range
    param src: <np.array> The input series
    param range_min: <float> The minimum value of the unbounded range
    param range_max: <float> The maximum value of the unbounded range
    returns <np.array> The normalized series
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    return range_min + (range_max - range_min) * scaler.fit_transform(src.reshape(-1,1))[:,0]


def rescale(src: np.array, old_min, old_max, new_min=0, new_max=1) -> np.array:
    """
    function Rescales a source value with a bounded range to anther bounded range
    param src: <np.array> The input series
    param old_min: <float> The minimum value of the range to rescale from
    param old_max: <float> The maximum value of the range to rescale from
    param new_min: <float> The minimum value of the range to rescale to
    param new_max: <float> The maximum value of the range to rescale to
    returns <np.array> The rescaled series
    """
    rescaled_value = new_min + (new_max - new_min) * (src - old_min) / max(old_max - old_min, 10e-10)
    return rescaled_value


def n_rsi(src: pd.Series, n1, n2) -> np.array:
    """
    function Returns the normalized RSI ideal for use in ML algorithms
    param src: <np.array> The input series
    param n1: <int> The length of the RSI
    param n2: <int> The smoothing length of the RSI
    returns <np.array> The normalized RSI
    """
    return rescale(ema(rsi(src, n1), n2).values, 0, 100)


def n_cci(high_src: pd.Series, low_src: pd.Series, close_src: pd.Series, n1, n2) -> np.array:
    """
    function Returns the normalized CCI ideal for use in ML algorithms
    param high_src: <np.array> The input series for the high price
    param low_src: <np.array> The input series for the low price
    param close_src: <np.array> The input series for the close price
    param n1: <int> The length of the CCI
    param n2: <int> The smoothing length of the CCI
    returns <np.array> The normalized CCI
    """
    return normalize(ema(cci(high_src, low_src, close_src, n1), n2).values)


def n_wt(src: pd.Series, n1=10, n2=11) -> np.array:
    """
    function Returns the normalized WaveTrend Classic series ideal for use in ML algorithms
    param src: <np.array> The input series
    param n1: <int> The first smoothing length for WaveTrend Classic
    param n2: <int> The second smoothing length for the WaveTrend Classic
    returns <np.array> The normalized WaveTrend Classic series
    """
    ema1 = ema(src, n1)
    ema2 = ema(abs(src - ema1), n1)
    ci = (src - ema1) / (0.015 * ema2)
    wt1 = ema(ci, n2)  # tci
    wt2 = sma(wt1, 4)
    return normalize((wt1 - wt2).values)


def n_adx(high_src: pd.Series, low_src: pd.Series, close_src: pd.Series, n1) -> np.array:
    """
    function Returns the normalized ADX ideal for use in ML algorithms
    param high_src: <np.array> The input series for the high price
    param low_src: <np.array> The input series for the low price
    param close_src: <np.array> The input series for the close price
    param n1: <int> The length of the ADX
    """
    return rescale(adx(high_src, low_src, close_src, n1).values, 0, 100)
    # TODO: Replicate ADX logic from jdehorty


def n_stoch(high_src: pd.Series, low_src: pd.Series, close_src: pd.Series, n1, n2) -> np.array:
    """
    function Returns the normalized stochastic oscillator signal for use in ML algorithms
    param high_src: <np.array> The input series for the high price
    param low_src: <np.array> The input series for the low price
    param close_src: <np.array> The input series for the close price
    param n1: <int> The length of the stochastic
    param n2: <int> The smoothing length of the stochastic
    returns <np.array> The stochastic (which should already be normalized)
    """
    return stoch_signal(high_src, low_src, close_src, n1, n2).values
