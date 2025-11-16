import logging
import json

from abc import abstractmethod
from ta.volatility import average_true_range
from ta.trend import ema_indicator as EMA, adx as ADX
from ta.momentum import KAMAIndicator as KAMA
import numpy as np
import pandas as pd

from common.types import Signal, NpEncoder
from typing import Protocol, Tuple

logger = logging.getLogger("indicators")
logger.setLevel(logging.INFO)


class Indicator(Protocol):
    name: str
    config: dict = {}
    df: pd.DataFrame
    last_action: str = ""

    def __init__(self, config: dict, df: pd.DataFrame):
        self.config = config
        self.df = df

    def _is_indicator_enabled(self, indicator_name: str) -> bool:
        indicator_data = self.config.get('indicators', {}).get(indicator_name, {})
        return indicator_data.get('enabled') == 1

    def _get_indicator_value(self, indicator_name: str):
        params = self._get_indicator_params(indicator_name)
        if indicator_name == "ADX":
            return self.adx(params)
        elif indicator_name == "ATR":
            return self.atr(params)
        elif indicator_name == "ROC":
            return self.roc(params)
        elif indicator_name == "KAMA":
            return self.kama(params)
        elif indicator_name == "KAMA_ROC":
            return self.kama_roc(params)
        elif indicator_name == "KAMA_TREND":
            return self.kama_trend(params)
        return None

    def _get_indicator_params(self, indicator_name: str) -> dict:
        indicator_data = self.config.get('indicators', {}).get(indicator_name, {})
        return indicator_data.get('params', {}) if indicator_data.get('enabled') == 1 else {}

    def adx(self, params):
        self.df['adx'] = ADX(self.df['high'], self.df['low'], self.df['close'], params.get('window', 14))
        return self.df['adx'].tail(1).values[0]

    def kama(self, params):
        self.df['kama'] = KAMA(self.df['close'], params.get("window", 10), params.get("pow1", 2), params.get("pow2", 30)).kama().values
        return self.df['kama'].tail(1).values[0]

    def kama_roc(self, params):
        self.df['kama_roc'] = self.df['kama'].pct_change(params.get("window", 2)) * 100
        return self.df['kama_roc'].tail(1).values[0]

    def kama_trend(self, params):
        self.df['kama_trend'] = np.where((self.df['kama_roc'].abs().gt(params.get("pow1", 0.002))), 1, 0)
        return self.df['kama_trend'].tail(1).values[0]

    def _andean(self, length, signal_length):
        """
        Inputs
        ------
        close : Closing price (Array)
        open  : Opening price (Array)

        Settings
        --------
        length        : Indicator period (float)
        signal_length : Signal line period (float)

        Returns
        -------
        Bull   : Bullish component (Array)
        Bear   : Bearish component (Array)
        Signal : Signal line (Array)

        Example
        -------
        bull,bear,signal = AndeanOsc(close,open,14,9)
        """
        close = self.df['close']
        open = self.df['open']

        n = len(close)

        alpha = 2 / (length + 1)
        alpha_signal = 2 / (signal_length + 1)

        up1, up2, dn1, dn2, bull, bear, signal = np.zeros((7, n))

        up1[0] = dn1[0] = signal[0] = close[0]
        up2[0] = dn2[0] = close[0] ** 2

        for i in range(1, n):
            up1[i] = max(close[i], open[i], up1[i - 1] - alpha * (up1[i - 1] - close[i]))
            dn1[i] = min(close[i], open[i], dn1[i - 1] + alpha * (close[i] - dn1[i - 1]))

            up2[i] = max(close[i] ** 2, open[i] ** 2, up2[i - 1] - alpha * (up2[i - 1] - close[i] ** 2))
            dn2[i] = min(close[i] ** 2, open[i] ** 2, dn2[i - 1] + alpha * (close[i] ** 2 - dn2[i - 1]))

            bull[i] = np.sqrt(dn2[i] - dn1[i] ** 2)
            bear[i] = np.sqrt(up2[i] - up1[i] ** 2)

            signal[i] = signal[i - 1] + alpha_signal * (np.maximum(bull[i], bear[i]) - signal[i - 1])

        return bull, bear, signal

    def andean(self):
        params = self._get_indicator_params("ANDEAN")
        self.df['ao_bull'],  self.df['ao_bear'], self.df['ao_signal'] = self._andean(params.get("length", 50), params.get("signal_length", 9))

    def atr(self, params) -> float:
        v = average_true_range(self.df['high'], self.df['low'], self.df['close'], params.get("window", 4))
        return v.tail(1).values[0]

    def roc(self, params) -> float:
        cols = ['yhat1', 'ma_fast']
        for col in cols:
            if col in self.df.columns:
                v = self.df[col].pct_change(params.get("window", 1)) * 100
                return v.tail(1).values[0]
        return 0

    def regime(self, params):
        src = self.df['close']
        high = self.df['high']
        low = self.df['low']

        # @njit(parallel=True, cache=True)
        def klmf(_src, _high, _low):
            value1 = np.array([0.0] * len(_src))
            value2 = np.array([0.0] * len(_src))
            _klmf = np.array([0.0] * len(_src))

            for i in range(len(_src)):
                if (_high[i] - _low[i]) == 0: continue
                value1[i] = 0.2 * (_src[i] - _src[i - 1 if i >= 1 else 0]) + 0.8 * value1[i - 1 if i >= 1 else 0]
                value2[i] = 0.1 * (_high[i] - _low[i]) + 0.8 * value2[i - 1 if i >= 1 else 0]

            with np.errstate(divide='ignore', invalid='ignore'):
                omega = np.nan_to_num(np.abs(np.divide(value1, value2)))
            alpha = (-(omega ** 2) + np.sqrt((omega ** 4) + 16 * (omega ** 2))) / 8

            for i in range(len(_src)):
                _klmf[i] = alpha[i] * _src[i] + (1 - alpha[i]) * _klmf[i - 1 if i >= 1 else 0]

            return _klmf

        abs_curve_slope = np.abs(np.diff(klmf(src.values, high.values, low.values), prepend=0.0))
        exponential_average_abs_curve_slope = EMA(pd.Series(abs_curve_slope), params.get("window", 200)).values
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized_slope_decline = (abs_curve_slope - exponential_average_abs_curve_slope) / exponential_average_abs_curve_slope

        self.df["regime"] = normalized_slope_decline

    def limit_data(self, limit: tuple) -> pd.DataFrame:
        _start, _end = limit

        df_ni = self.df.copy(True)
        df_ni['ts_date'] = pd.to_datetime(df_ni['ts_event'])

        # use numeric index to filter out corresponding data from yhat1 and yhat2
        df_ni = df_ni.reset_index()

        q = f"ts_event >= {_start.value} and ts_event < {_end.value}"
        df_ni = df_ni.query(q)
        return df_ni

    def get_column(self, column: str):
        if column in self.df.columns:
            return self.df[column].tail(1).values[0]
        return None

    def internals(self) -> str:
        val = self.df.tail(1)
        d = dict()
        idx = val.index[0]
        for col in self.df.columns:
            col = col.strip()
            d[col] = val[col][idx]
            # if type(d[col]) == numpy.bool_:
            #     d[col] = bool(d[col])
        return json.dumps(d, cls=NpEncoder)

    def run(self) -> Tuple[Signal, float]:
        _signal, _elapsed = self.process()

        # calculate the indicators and append to signal
        for _indicator in self.config.get('indicators', {}).keys():
            if self._is_indicator_enabled(_indicator):
                _signal.indicator_data[_indicator.lower()] = self._get_indicator_value(_indicator)

        return _signal, _elapsed

    @abstractmethod
    def process(self) -> Tuple[Signal, float]:
        pass

    @abstractmethod
    def dump(self, name: str, limit: tuple | None = None):
        pass
