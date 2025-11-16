import logging

from typing import Tuple

from common.types import Signal
from indicators.indicator import Indicator

from classifier.settings import *
from classifier.distance import get_lorentzian_predictions
from classifier.signal import generate_signals

import time


class Lorentzian(Indicator):
    def __init__(self, config: dict, df: pd.DataFrame):
        super().__init__(config, df)
        self.name = "Lorentzian"
        self.settings = config['settings']
        self.settings.init(df)

    def action(self) -> str:
        action = "none"
        if pd.notna(self.get_column('start_long_trade')):
            action = "long"
        if pd.notna(self.get_column('start_long_trade')):
            action = "short"
        if 'end_long_trade' in self.df.columns and 'end_short_trade' in self.df.columns:
            if pd.notna(self.get_column('end_long_trade')) or pd.notna(self.get_column('end_short_trade')):
                action = "flat"
        self.last_action = action
        return action

    def dump(self, name: str, limit: tuple | None = None):
        df_ni = self.limit_data(limit)

        try:
            if len(df_ni):
                # yhat1 = self.df['yhat1'][df_ni.index[0]:df_ni.index[-1] + 1]
                # yhat2 = self.df['yhat2'][df_ni.index[0]:df_ni.index[-1] + 1]

                # drawings = [
                #     {'type': 'line', 'data': yhat1, 'color': 'blue', 'size': 1, 'alpha': 1.0},
                #     {'type': 'line', 'data': yhat2, 'color': 'gray', 'size': 1, 'alpha': 1.0},
                # ]

                # self.draw_plot(name, drawings, df_ni)

                # dump_trades(df_ni, f"{name}-trades.txt")
                df_ni.to_csv(f"{name}-trades.csv")
        except Exception as e:
            print(f"Error dumping data - {e}")

    def process(self) -> Tuple[Signal, float]:
        start = time.time_ns()
        self.df['predictions'] = get_lorentzian_predictions(self.settings)
        elapsed_time = time.time_ns() - start

        self.df = generate_signals(self.settings, self.df)

        indicator_data = {
            "estimate1": float(self.get_column('yhat1')),
            "estimate2": float(self.get_column('yhat2')),
            "signal": self.get_column('signal'),
            "prediction": self.get_column('prediction'),
        }

        # study, symbol, schema is often set by the calling context (e.g., a strategy runner)
        # For now, passing them as empty strings as per original.
        # elapsed is typically updated by process() or caller. ts() from lc is used for 'time'.
        return Signal(study="", symbol="", schema="",
                      action=self.action(),
                      price=float(self.get_column('close')),
                      elapsed=0,  # Placeholder, typically updated by process() or caller
                      time=int(self.get_column('ts_event')),
                      rows=len(self.df),
                      internals=self.internals(),
                      indicator_data=indicator_data), elapsed_time
