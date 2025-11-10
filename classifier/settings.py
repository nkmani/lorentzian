import pandas as pd
import numpy as np

import json
import itertools
from typing import List
from enum import IntEnum
from .features import n_rsi, n_cci, n_wt, n_adx, n_stoch
from constants import *

# Label Object: Used for classifying historical data as training data for the ML Model
class Direction(IntEnum):
    LONG = 1
    SHORT = -1
    NEUTRAL = 0


class Filters:
    use_sma: bool
    sma_period: int

    use_ema: bool
    ema_period: int

    use_volatility: bool
    volatility_min_period: int
    volatility_max_period: int

    use_regime: bool
    regime_threshold: float

    use_adx: bool
    adx_threshold: float
    adx_length: int

    use_kernel: bool
    kernel_lookback: int
    kernel_relative_weight: float
    kernel_regression_level: int
    kernel_crossover_lag: int
    kernel_use_smoothing: bool

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class Feature:
    name: str
    param1: int
    param2: int

    def __init__(self, name, param1, param2):
        self.name = name
        self.param1 = param1
        self.param2 = param2


class Settings:
    source: str = "close"
    data: pd.DataFrame = None
    sample_interval: int = 4
    neighbors_count: int = 8
    max_bars_back: int = 2000
    distance_algo_type: int = 1
    use_dynamic_exits: bool = True

    stop_loss: float = 0.5
    take_profit: float = 1.5
    point_value: float = 5
    trade_commission: float = 0.8
    
    use_sma: bool = False
    sma_period: int = 200

    use_ema: bool = False
    ema_period: int = 50

    use_volatility: bool = True
    volatility_min_period: int = 1
    volatility_max_period: int = 10

    use_regime: bool = True
    regime_threshold: float = 0.1

    use_adx: bool = False
    adx_threshold: float = 20
    adx_length: int = 14

    use_kernel: bool = True
    kernel_lookback: int = 8
    kernel_relative_weight: float = 8
    kernel_regression_level: int = 25
    kernel_crossover_lag: int = 2
    kernel_use_smoothing: bool = False

    feature_list: List = []
    features: List = []
    debug_flags: int = 0
    iteration: int = 0

    def __init__(self, **kwargs):
        self.default_filters()
        self.feature_list = default_features()
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.y_train = None
        self.src = None
        self.iteration = 0

    def init(self, df: pd.DataFrame):
        self.data = df
        src = self.data[self.source]

        self.y_train = np.where(src.shift(self.sample_interval) < src.shift(0), Direction.SHORT, np.where(src.shift(self.sample_interval) > src.shift(0), Direction.LONG, Direction.NEUTRAL))
        self.src = src

        self.features = []
        for feature in self.feature_list:
            if type(feature) == Feature:
                self.features.append(series_from(df, feature))
            elif type(feature) == list and len(feature) == 3:
                self.features.append(series_from(df, Feature(feature[0], feature[1], feature[2])))


    def get_changed_settings(self):
        # Get instance attributes
        instance_attributes = vars(self)
        # Get class attributes
        class_attributes = vars(type(self))

        changed_settings = {}
        excluded_attributes = ['data', 'filters', 'features', 'iteration']
        for key, value in instance_attributes.items():
            # Check if the attribute is a setting we want to track (e.g., starts with a letter)
            # and if it differs from the class default
            if not key.startswith('__') and not key in excluded_attributes and key in class_attributes:
                if key == 'feature_list':
                    if value != default_features():
                        changed_settings[key] = value
                elif value != class_attributes[key]:
                    changed_settings[key] = value

        return changed_settings

    def default_filters(self):
        self.use_sma=False
        self.sma_period=200

        self.use_ema=False
        self.ema_period=50

        self.use_regime=True
        self.regime_threshold=0.1

        self.use_volatility=True
        self.volatility_min_period=1
        self.volatility_max_period=10

        self.use_kernel=True
        self.kernel_lookback=8
        self.kernel_relative_weight=8
        self.kernel_regression_level=25
        self.kernel_crossover_lag=2
        self.kernel_use_smoothing=False

        self.use_adx=False
        self.adx_threshold=20
        self.adx_length=14

    def max_bars_back_index(self) -> int:
        return (len(self.data.index) - self.max_bars_back) if (len(self.data.index) >= self.max_bars_back) else 0


def generate_combinations(options: dict):
    combinations = []
    keys = list(options.keys())
    value_combinations = itertools.product(*map(options.get, keys))
    for values in value_combinations:
        combinations.append(dict(zip(keys, values)))
    return combinations


class SettingsIterator:
    def __init__(self, config_file: str = None):
        if config_file is not None:
            with open(config_file, "r") as f:
                self.options = json.load(f)
        else:
            self.options = {
                'sample_interval': [4],
                'neighbors_count': [8],
            }
        self.combinations = generate_combinations(self.options)
        self.current = 0
        self.limit = len(self.combinations)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < self.limit:
            settings = default_settings()
            for key, value in self.combinations[self.current].items():
                setattr(settings, key, value)
            self.current += 1
            settings.iteration = self.current
            return settings
        else:
            raise StopIteration


def series_from(data: pd.DataFrame, feature: Feature):
    match feature.name.lower():
        case "rsi":
            return n_rsi(data['close'], feature.param1, feature.param2)
        case "wt":
            hlc3 = (data['high'] + data['low'] + data['close']) / 3
            return n_wt(hlc3, feature.param1, feature.param2)
        case "cci":
            return n_cci(data['high'], data['low'], data['close'], feature.param1, feature.param2)
        case "stoch":
            return n_stoch(data['high'], data['low'], data['close'], feature.param1, feature.param2)
        case "adx":
            return n_adx(data['high'], data['low'], data['close'], feature.param1)
    return None


# def default_filters() -> Filters:
#     return Filters(
#         use_sma=False,
#         sma_period=200,
#
#         use_ema=False,
#         ema_period=50,
#
#         use_regime=True,
#         regime_threshold=0.1,
#
#         use_volatility=True,
#         volatility_min_period=1,
#         volatility_max_period=10,
#
#         use_kernel=True,
#         kernel_lookback=8,
#         kernel_relative_weight=8,
#         kernel_regression_level=25,
#         kernel_crossover_lag=2,
#         kernel_use_smoothing=False,
#
#         use_adx=False,
#         adx_threshold=20,
#         adx_length=14,
#     )


def default_features() -> list:
    # return [
    #     series_from(data, Feature("RSI", 14, 1)),  # f1
    #     series_from(data, Feature("WT", 10, 11)),  # f2
    #     series_from(data, Feature("CCI", 20, 1)),  # f3
    #     series_from(data, Feature("ADX", 20, 2)),  # f4
    #     series_from(data, Feature("RSI", 9, 1)),  # f5
    # ]
    # return [
    #     Feature('rsi', 14, 1), # f1
    #     Feature('wt', 10, 11), # f2
    #     Feature('cci', 20, 1), # f3
    #     Feature('adx', 20, 2), # f4
    #     Feature('rsi', 9, 1),  # f5
    # ]
    return [
            ['rsi', 14, 1], # f1
            ['wt', 10, 11], # f2
            ['cci', 20, 1], # f3
            ['adx', 20, 2], # f4
            ['rsi', 9, 1],  # f5
        ]

def default_settings() -> Settings:
    return Settings(
        source="close",
        sample_interval=4,
        neighbors_count=8,
        max_bars_back=2000,
        distance_algo_type=1,
        use_dynamic_exists=True,
        stop_loss=0.5,
        take_profit=1.5,
        point_value=5,
        trade_commission=0.8,
        feature_list=default_features(),
    )
