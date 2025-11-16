import datetime
from dataclasses import dataclass, field
from enum import IntEnum

import pandas as pd
import databento as db
from typing import Callable, List
import json
import os
import numpy as np


@dataclass
class Ohlcv:
    ts_event: np.int64
    open: float
    high: float
    low: float
    close: float
    volume: int

    def __init__(self, _d: dict | None = None):
        if _d is not None:
            self.ts_event = np.int64(_d['ts_event'])
            self.open = float(_d['open'])
            self.high = float(_d['high'])
            self.low = float(_d['low'])
            self.close = float(_d['close'])
            self.volume = int(float(_d['volume']))
        else:
            self.ts_event = np.int64(0)
            self.open = 0
            self.high = 0
            self.low = 0
            self.close = 0
            self.volume = 0

    def to_dict(self) -> dict:
        return {
            "ts_event": self.ts_event,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


class SignalCode(IntEnum):
    NONE = 0
    LONG = 1
    SHORT = 2
    FLAT = 3

    def __str__(self):
        return str(self.name)


def signal_code(_action: str) -> SignalCode | None:
    try:
        return {"none": SignalCode.NONE, "long": SignalCode.LONG, "short": SignalCode.SHORT, "flat": SignalCode.FLAT}[_action.lower()]
    except KeyError:
        # logger.error("Unknown signal code: {}".format(_action))
        return None


def signal_from_json(json_str: str):
    try:
        data = json.loads(json_str)
        signal = Signal(
            study=data["study"],
            symbol=data["symbol"],
            schema=data["schema"],
            action=data["action"],
            price=data["price"],
            indicator_data=data.get("indicator_data", {}),
            elapsed=data["elapsed"],
            time=data["time"],
            rows=data["rows"],
            internals=data.get("internals", "")
        )
        return signal
    except json.decoder.JSONDecodeError:
        raise ValueError("Invalid JSON format")


def signal_from_message(data: dict):
    try:
        signal = Signal(
            study=data["study"],
            symbol=data["symbol"],
            schema=data["schema"],
            action=data["action"],
            price=data["price"],
            indicator_data=data.get("indicator_data", {}),
            elapsed=data["elapsed"],
            time=data["time"],
            rows=data["rows"],
            internals=data.get("internals", "")
        )
        return signal
    except KeyError:
        raise KeyError("Message is missing some required fields")


@dataclass
class Signal:
    study: str
    symbol: str
    schema: str
    action: str
    price: float
    elapsed: float
    time: int
    rows: int
    internals: str
    indicator_data: dict = field(default_factory=dict)

    def __repr__(self):
        return f"Signal(action={self.action} price={self.price} indicator_data={self.indicator_data} stats[elapsed={self.elapsed} time={self.time} rows={self.rows} internals={self.internals}])"

    def code(self) -> SignalCode | None:
        return signal_code(self.action)

    def to_dict(self) -> dict:
        return {
            "study": self.study,
            "symbol": self.symbol,
            "schema": self.schema,
            "action": self.action,
            "price": self.price,
            "indicator_data": self.indicator_data,
            "elapsed": self.elapsed,
            "time": int(self.time),
            "rows": self.rows,
            "internals": self.internals,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), cls=NpEncoder)


SignalCallback = Callable[[Signal], int]


# @dataclass
# class SymbolInfo:
#     symbol: str
#     schema: str
#     agg_schema: str
#     folder: str
#     prefix: str
#     status: str
#     config: dict
#
#
# def si_default() -> SymbolInfo:
#     return SymbolInfo("MES.c.0", "ohlcv-1s", "ohlcv-1m", "data/mes", "mes-intra", "data/mes/mes.json", {})
#
#
# def si_from_args(_symbol, _schema, _agg_schema, _folder, _prefix, _status_file) -> SymbolInfo:
#     return SymbolInfo(_symbol, _schema, _agg_schema, _folder, _prefix, _status_file, {})
#
#
# def si_from_config(_config, _status_file) -> SymbolInfo:
#     __data = _config['data']
#     return SymbolInfo(__data['symbol'], __data['schema'], __data['agg_schema'], __data['folder'], __data['prefix'],
#                       _status_file, _config)

@dataclass
class AggSchema:
    freq: str  # (e-g 5s, 10s, 15s, 30s, 1min, 5min)
    schema: str
    file: str | None = None
    ts: pd.Timestamp | None = None
    count: int = 0
    a_record: Ohlcv | None = None
    record: db.OHLCVMsg | None = None


@dataclass
class DbConfig:
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    file_date: datetime.date
    file: str
    last_ts_event: int
    config: dict
    seq_no: int = 0
    aggregates: List[AggSchema] = field(default_factory=list)

    def setup_filenames(self, date):
        if isinstance(date, datetime.date):
            __file_date = str(date)
        else:
            __file_date = date
        common = f"{self.config['data']['folder']}/{__file_date}/{self.config['data']['prefix']}-{__file_date}"
        self.file_date = __file_date
        self.file = f"{common}-{self.config['data']['schema']}.csv"
        os.makedirs(os.path.dirname(self.file), exist_ok=True)

    def __init__(self, start, end, _config: dict):
        self.start_date = pd.Timestamp(start)
        self.end_date = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=1)
        self.config = _config
        self.setup_filenames(self.start_date.date())


class GrState(IntEnum):
    Flat = 0
    Green = 1
    Red = 2


def gr_color(gr) -> str:
    if gr == GrState.Flat:
        return 'Flat'
    elif gr == GrState.Green:
        return 'Green'
    else:
        return 'Red'


def gr_state(prediction) -> GrState:
    if prediction > 0:
        return GrState.Green
    elif prediction < 0:
        return GrState.Red
    else:
        return GrState.Flat


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NpEncoder, self).default(obj)