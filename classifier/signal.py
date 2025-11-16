import math
import pandas as pd
import numpy as np
from ta.volatility import average_true_range as atr
from ta.trend import cci, adx, ema_indicator as ema, sma_indicator as sma
from .settings import Settings, Direction


def bars_since(s: np.array):
    val = np.array([0.0] * s.size)
    c = math.nan
    for i in range(s.size):
        if s[i]:
            c = 0
            continue
        if c >= 0:
            c += 1
        val[i] = c
    return val


def slope(s: np.array):
    val = np.array([0.0] * s.size)
    c = math.nan
    for i in range(s.size):
        if s[i]:
            c = 0
            continue
        if c >= 0:
            c += 1
        val[i] = c
    return val


def shift(arr, idx, fill_value=0.0):
    return np.pad(arr, (idx,), mode='constant', constant_values=(fill_value,))[:arr.size]


def gaussian(src: pd.Series, lookback: int, start_at_bar: int):
    """
    vectorized calculate for gaussian curve
    :param src:
    :param lookback:
    :param start_at_bar:
    :return:
    """
    current_weight = [0.0]*len(src)
    cumulative_weight = 0.0
    for i in range(start_at_bar + 2):
        y = src.shift(i, fill_value=0.0)
        w = math.exp(-(i ** 2) / (2 * lookback ** 2))
        current_weight += y.values * w
        cumulative_weight += w
    val = current_weight / cumulative_weight
    val[:start_at_bar + 1] = 0.0

    return val


def rational_quadratic(src: pd.Series, lookback: int, relative_weight: float, start_at_bar: int):
    """
    vectorized calculate for rational quadratic curve
    :param src:
    :param lookback:
    :param relative_weight:
    :param start_at_bar:
    :return:
    """
    current_weight = [0.0] * len(src)
    cumulative_weight = 0.0
    for i in range(start_at_bar + 2):
        y = src.shift(i, fill_value=0.0)
        w = (1 + (i ** 2 / (lookback ** 2 * 2 * relative_weight))) ** -relative_weight
        current_weight += y.values * w
        cumulative_weight += w
    val = current_weight / cumulative_weight
    val[:start_at_bar + 1] = 0.0

    return val


def regime(src: pd.Series, high: pd.Series, low: pd.Series) -> np.array:
    # @njit(parallel=True, cache=True)
    def klmf(src: np.array, high: np.array, low: np.array):
        value1 = np.array([0.0]*len(src))
        value2 = np.array([0.0]*len(src))
        klmf = np.array([0.0]*len(src))

        for i in range(len(src)):
            if (high[i] - low[i]) == 0: continue
            value1[i] = 0.2 * (src[i] - src[i - 1 if i >= 1 else 0]) + 0.8 * value1[i - 1 if i >= 1 else 0]
            value2[i] = 0.1 * (high[i] - low[i]) + 0.8 * value2[i - 1 if i >= 1 else 0]

        with np.errstate(divide='ignore',invalid='ignore'):
            omega = np.nan_to_num(np.abs(np.divide(value1, value2)))
        alpha = (-(omega ** 2) + np.sqrt((omega ** 4) + 16 * (omega ** 2))) / 8

        for i in range(len(src)):
            klmf[i] = alpha[i] * src[i] + (1 - alpha[i]) * klmf[i - 1 if i >= 1 else 0]

        return klmf

    abs_curve_slope = np.abs(np.diff(klmf(src.values, high.values, low.values), prepend=0.0))
    exponential_average_abs_curve_slope = ema(pd.Series(abs_curve_slope), 200).values
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized_slope_decline = (abs_curve_slope - exponential_average_abs_curve_slope) / exponential_average_abs_curve_slope
    return normalized_slope_decline


def filter_regime(src: pd.Series, high: pd.Series, low: pd.Series, use_regime, threshold):
    """
    regime_filter
    param src: <np.array> The source series
    param high: <np.array> The input series for the high price
    param low: <np.array> The input series for the low price
    param useRegimeFilter: <bool> Whether to use the regime filter
    param threshold: <float> The threshold
    returns <np.array> Boolean indicating whether or not to let the signal pass through the filter
    """
    if not use_regime:
        return np.array([True] * len(src))

    _filter = np.array([False] * len(src))
    normalized_slope_decline = regime(src, high, low)
    flags = normalized_slope_decline >= threshold
    _filter[(len(_filter) - len(flags)):] = flags
    return _filter


def filter_adx(src: pd.Series, high: pd.Series, low: pd.Series, use_adx, adx_threshold, length=14):
    """
    function filter_adx
    param src: <np.array> The source series
    param high: <np.array> The input series for the high price
    param low: <np.array> The input series for the low price
    param use_adx: <bool> Whether to use the ADX filter
    param adx_threshold: <int> The ADX threshold
    param length: <int> The length of the ADX
    returns <np.array> Boolean indicating whether or not to let the signal pass through the filter
    """
    if not use_adx:
        return np.array([True]*len(src))
    return adx(high, low, src, length).values > adx_threshold


def filter_volatility(high, low, close, use_volatility_filter, min_length=1, max_length=10):
    """
    function filter_volatility
    param high: <np.array> The input series for the high price
    param low: <np.array> The input series for the low price
    param close: <np.array> The input series for the close price
    param useVolatilityFilter: <bool> Whether to use the volatility filter
    param minLength: <int> The minimum length of the ATR
    param maxLength: <int> The maximum length of the ATR
    returns <np.array> Boolean indicating whether or not to let the signal pass through the filter
    """
    if not use_volatility_filter:
        return np.array([True] * len(close))
    recent_atr = atr(high, low, close, min_length).values
    historical_atr = atr(high, low, close, max_length).values
    return recent_atr > historical_atr


def generate_signals(settings: Settings, df: pd.DataFrame) -> pd.DataFrame:
    src = df[settings.source]

    yhat1 = np.array
    yhat2 = np.array

    if settings.use_kernel:
        yhat1 = rational_quadratic(src, settings.kernel_lookback, settings.kernel_relative_weight, settings.max_bars_back_index())
        df['yhat1'] = yhat1

        yhat2 = gaussian(src, settings.kernel_lookback - settings.kernel_crossover_lag, settings.kernel_regression_level)
        df['yhat2'] = yhat2

    is_ema_uptrend = np.where(settings.use_ema, (df["close"] > ema(df["close"], settings.ema_period)), True)
    is_ema_downtrend = np.where(settings.use_ema, (df["close"] < ema(df["close"], settings.ema_period)), True)
    is_sma_uptrend = np.where(settings.use_sma, (df["close"] > sma(df["close"], settings.sma_period)), True)
    is_sma_downtrend = np.where(settings.use_sma, (df["close"] < sma(df["close"], settings.sma_period)), True)

    # User Defined Filters: Used for adjusting the frequency of the ML Model's prediction
    _volatility = filter_volatility(df['high'], df['low'], df['close'], settings.use_volatility, settings.volatility_min_period, settings.volatility_max_period)
    _regime = filter_regime(df['close'], df['high'], df['low'], settings.use_regime, settings.regime_threshold)
    _adx = filter_adx(df['close'], df['high'], df['low'], settings.use_adx, settings.adx_threshold, settings.adx_length)
    filter_all = _volatility & _regime & _adx

    prediction = df['predictions']
    signal = np.where(((prediction > 0) & filter_all), Direction.LONG, np.where(((prediction < 0) & filter_all), Direction.SHORT, None))
    signal[0] = (0 if signal[0] is None else signal[0])
    for i in np.where(signal is None)[0]:
        signal[i] = signal[i - 1 if i >= 1 else 0]

    change = lambda ser, idx: (shift(ser, idx, fill_value=ser[0]) != shift(ser, idx + 1, fill_value=ser[0]))

    # Bar-Count Filters: Represents strict filters based on a pre-defined holding period of 4 bars
    bars_held = []
    is_different_signal_type = (signal != shift(signal, 1, fill_value=float(signal[0])))
    _sigFlip = np.where(is_different_signal_type)[0].tolist()
    if not (len(is_different_signal_type) in _sigFlip):
        _sigFlip.append(len(is_different_signal_type))
    for i, x in enumerate(_sigFlip):
        if i > 0:
            bars_held.append(0)
        bars_held += range(1, x - (-1 if i == 0 else _sigFlip[i - 1]))
    is_held_four_bars = (pd.Series(bars_held) == 4).tolist()
    is_held_less_than_four_bars = (pd.Series(bars_held) < 4).tolist()

    is_early_signal_flip = (change(signal, 0) & change(signal, 1) & change(signal, 2) & change(signal, 3))
    is_buy_signal = ((signal == Direction.LONG) & is_ema_uptrend & is_sma_uptrend)
    is_sell_signal = ((signal == Direction.SHORT) & is_ema_downtrend & is_sma_downtrend)
    is_last_signal_buy = (shift(signal, settings.sample_interval) == Direction.LONG) & shift(is_ema_uptrend, settings.sample_interval) & shift(is_sma_uptrend, settings.sample_interval)
    is_last_signal_sell = (shift(signal, settings.sample_interval) == Direction.SHORT) & shift(is_ema_downtrend, settings.sample_interval) & shift(is_sma_downtrend, settings.sample_interval)
    is_new_buy_signal = (is_buy_signal & is_different_signal_type)
    is_new_sell_signal = (is_sell_signal & is_different_signal_type)

    # Kernel Rates of Change
    was_bearish_rate = np.where(shift(yhat1, 2) > shift(yhat1, 1), True, False)
    was_bullish_rate = np.where(shift(yhat1, 2) < shift(yhat1, 1), True, False)
    is_bearish_rate = np.where(shift(yhat1, 1) > yhat1, True, False)
    is_bullish_rate = np.where(shift(yhat1, 1) < yhat1, True, False)
    is_bearish_change = is_bearish_rate & was_bullish_rate
    is_bullish_change = is_bullish_rate & was_bearish_rate

    crossover = lambda s1, s2: (s1 > s2) & (shift(s1, 1) < shift(s2, 1))
    crossunder = lambda s1, s2: (s1 < s2) & (shift(s1, 1) > shift(s2, 1))

    is_bullish_cross_alert = crossover(yhat2, yhat1)
    is_bearish_cross_alert = crossunder(yhat2, yhat1)

    is_bullish_smooth = (yhat2 >= yhat1)
    is_bearish_smooth = (yhat2 <= yhat1)

    alert_bullish = np.where(settings.kernel_use_smoothing, is_bullish_cross_alert, is_bullish_change)
    alert_bearish = np.where(settings.kernel_use_smoothing, is_bearish_cross_alert, is_bearish_change)

    is_bullish = np.where(settings.use_kernel, np.where(settings.kernel_use_smoothing, is_bullish_smooth, is_bullish_rate), True)
    is_bearish = np.where(settings.use_kernel, np.where(settings.kernel_use_smoothing, is_bearish_smooth, is_bearish_rate), True)

    start_long_trade = is_new_buy_signal & is_bullish & is_ema_uptrend & is_sma_uptrend
    start_short_trade = is_new_sell_signal & is_bearish & is_ema_downtrend & is_sma_downtrend

    bars_since_red_entry = bars_since(start_short_trade)
    bars_since_red_exit = bars_since(alert_bullish)
    bars_since_green_entry = bars_since(start_long_trade)
    bars_since_green_exit = bars_since(alert_bearish)
    is_valid_short_exit = bars_since_red_exit > bars_since_red_entry
    is_valid_long_exit = bars_since_green_exit > bars_since_green_entry
    end_long_trade_dynamic = is_bearish_change & shift(is_valid_long_exit, 1)
    end_short_trade_dynamic = is_bullish_change & shift(is_valid_short_exit, 1)

    # Fixed Exit Conditions: Booleans for ML Model Position Exits based on Bar-Count Filters
    end_long_trade_strict = ((is_held_four_bars & is_last_signal_buy) | (is_held_less_than_four_bars & is_new_sell_signal & is_last_signal_buy)) & shift(start_long_trade, settings.sample_interval)
    end_short_trade_strict = ((is_held_four_bars & is_last_signal_sell) | (is_held_less_than_four_bars & is_new_buy_signal & is_last_signal_sell)) & shift(start_short_trade, settings.sample_interval)
    is_dynamic_exit_valid = (not settings.use_ema) & (not settings.use_sma) & (not settings.kernel_use_smoothing)
    end_long_trade = settings.use_dynamic_exits & is_dynamic_exit_valid & end_long_trade_dynamic
    end_short_trade = settings.use_dynamic_exits & is_dynamic_exit_valid & end_short_trade_dynamic

    df["bars_held"] = bars_held
    df["is_early_signal_flip"] = is_early_signal_flip
    df["is_last_signal_buy"] = is_last_signal_buy
    df["is_last_signal_sell"] = is_last_signal_sell
    df["is_new_buy_signal"] = is_new_buy_signal
    df["is_new_sell_signal"] = is_new_sell_signal

    df["start_long_trade"] = np.where(start_long_trade, df['low'], np.nan)
    df["start_short_trade"] = np.where(start_short_trade, df['high'], np.nan)

    df["end_long_trade"] = np.where(end_long_trade, df['high'], np.nan)
    df["end_short_trade"] = np.where(end_short_trade, df['low'], np.nan)
    df["end_long_trade_strict"] = np.where(end_long_trade_strict, df['high'], np.nan)
    df["end_short_trade_strict"] = np.where(end_short_trade_strict, df['low'], np.nan)
    df["signal"] = signal

    return df
