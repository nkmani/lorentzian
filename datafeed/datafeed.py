from abc import abstractmethod, ABCMeta, ABC

import os
import copy
import time
import json
from enum import IntEnum
from typing import Tuple

from common import *
from Types import Ohlcv
import numpy as np


logger = logging.getLogger('datafeed')
logger.setLevel(logging.INFO)


class DataSource(IntEnum):
    FILE = 0
    DB = 1  # Databento
    SC = 2  # SierraCharts

    def __str__(self):
        return str(self.name)


class BarType(IntEnum):
    OPEN = 0
    CLOSED = 1

    def __str__(self):
        return str(self.name)


class AggOhlcv(Ohlcv):
    freq: str = None
    type: BarType = None
    rec_ts: pd.Timestamp = None

    def __init__(self, _freq: str, _dict: dict | None):
        super().__init__(_dict)
        self.freq = _freq
        self.ts = pd.to_datetime(0, utc=True).ceil(freq=self.freq)

    def new(self, record: Ohlcv, norm_ts: pd.Timestamp):
        self.ts = norm_ts
        self.rec_ts = pd.to_datetime(np.int64(record.ts_event), utc=True)
        self.patch(record, norm_ts)
        self.type = BarType.OPEN
        logger.debug(f"New {self.freq} time={norm_ts} freq={self.freq} type={self.type} record={self.to_dict()}")

    def aggregate(self, record: Ohlcv):
        if self.high < record.high:
            self.high = record.high
        if self.low > record.low:
            self.low = record.low
        self.volume += record.volume
        self.close = record.close
        self.rec_ts = pd.to_datetime(np.int64(record.ts_event), utc=True)

    def patch(self, record: Ohlcv, ts: pd.Timestamp | None = None):
        self.ts_event = ts.value if ts else record.ts_event
        self.open = record.open
        self.close = record.close
        self.high = record.high
        self.low = record.low
        self.volume = record.volume

    def add(self, record: Ohlcv):
        self.rec_ts = pd.to_datetime(np.int64(record.ts_event), utc=True)
        norm_ts = pd.to_datetime(np.int64(record.ts_event), utc=True).ceil(freq=self.freq)
        logger.debug(f"Adding {self.freq} {self.rec_ts} {pd.to_datetime(np.int64(record.ts_event), utc=True)} --> {record}")
        if self.ts.value == 0:
            self.new(record, norm_ts)
            return None
        elif norm_ts == self.ts:
            self.aggregate(record)
            logger.debug(f"Aggregated {self.freq} record={record} --> {self.to_dict()}")
            return None
        elif norm_ts != self.ts:
            _current = copy.copy(self)
            _current.type = BarType.CLOSED
            logger.debug(f"Closing {self.freq} record: {pd.to_datetime(_current.ts_event, utc=True)} --> {_current.to_dict()}")

            # start a new aggregation
            self.new(record, norm_ts)
            return _current
        return None

    def __str__(self):
        return f"freq={self.freq} bar={str(self.type)} ohlcv={self.to_dict()}"


class Datafeed(metaclass=ABCMeta):
    source: DataSource = DataSource.FILE

    HEADER:str = f"ts_event,open,high,low,close,volume"

    def __init__(self, config, ds: DataSource, data_freq='1min', ltf_freq='1min', agg_freq='1min'):
        self.initialized = False

        self.config = config

        self.data_freq = data_freq
        self.data_freq_seconds = pd.Timedelta(self.data_freq).total_seconds()

        self.ltf_freq = ltf_freq
        self.ltf_freq_seconds = pd.Timedelta(ltf_freq).total_seconds()

        self.next_ltf_update = pd.to_datetime(0, utc=True).ceil(freq=self.ltf_freq).value

        self.agg_freq = agg_freq
        self.agg_freq_seconds = pd.Timedelta(agg_freq).total_seconds()

        self.ltf_updates = self.agg_freq_seconds > self.ltf_freq_seconds

        if self.ltf_freq_seconds % self.data_freq_seconds != 0:
            logger.fatal(f"LTF {self.ltf_freq_seconds} frequency is not divisible by Data frequency {self.data_freq_seconds}")
            return
        elif self.agg_freq_seconds % self.ltf_freq_seconds != 0:
            logger.fatal(f"Aggregate {self.agg_freq_seconds} frequency is not divisible by LTF frequency {self.ltf_freq_seconds}")
            return

        # is aggregation required
        self.aggregation_not_required = self.data_freq_seconds == self.agg_freq_seconds

        if self.agg_freq_seconds % self.data_freq_seconds != 0:
            logger.fatal(f"Aggregate {self.agg_freq_seconds} frequency is not divisible by Data frequency {self.data_freq_seconds}")
            return

        # message sequence no
        self.seq = 0

        self.last_ts_event = 0

        if self.ltf_updates:
            self.ltf_records = []
            self.ltf_record = AggOhlcv(ltf_freq, None)

        self.record = AggOhlcv(agg_freq, None)

        self.source = ds
        self.df = None

        self.wait_period = 50  # milliseconds
        self.timeout_period = 2000  # milliseconds

    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the data feed with the preloading of any historical data available.
        :return: True if initialization was successful, False otherwise.
        """

    @abstractmethod
    def eof(self):
        """
        Check if we are at the end of the feed (applies to file sources only)
        :return: bool
        """

    @abstractmethod
    def next(self) -> Tuple[int, str]:
        """
        Return the next available record, as a sequence number and json response tuple.
        :return: Tuple[int, str]
        """

    def process(self) -> AggOhlcv | Ohlcv | None:
        wait = 0
        while wait < self.timeout_period:
            self.seq, msg = self.next()
            if msg is not None:
                try:
                    if len(msg) == 0:
                        return None
                    _d = json.loads(msg)
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON {msg}: {e}")
                    return None

                if self.data_freq not in _d:
                    logging.warning(f"Could not find {self.data_freq} in data")
                    return None

                _d = Ohlcv(_d[self.data_freq])

                if self.aggregation_not_required:
                    ret = AggOhlcv(self.agg_freq, _d.to_dict())
                    ret.rec_ts = pd.to_datetime(np.int64(_d.ts_event), utc=True)
                    ret.ts = _d.ts_event
                    ret.type = BarType.CLOSED
                    return ret

                ltf_ret = None
                ret = self.record.add(_d)

                if self.ltf_updates:
                    ltf_ret = self.ltf_record.add(_d)

                if ret is not None and ret.type == BarType.CLOSED:
                    logger.debug(f"Returning closed bar {ret.ts} --> {ret}")
                    # print("---------------")
                    return ret

                if self.ltf_updates and ltf_ret is not None and ltf_ret.type == BarType.CLOSED:
                    logger.debug(f"Returning open bar {self.record.ts} --> {self.record}")
                    # print("---------------")
                    return self.record

                # ret = None
                # if self.ltf_updates:
                #     ltf_ret = self.ltf_record.add(_d)
                #     if ltf_ret is not None and ltf_ret.type == BarType.CLOSED:
                #         ret = self.record.add(ltf_ret)
                #         if ret is None:
                #             logger.debug(f"Returning open bar {self.record.ts} --> {self.record}")
                #             print("---------------")
                #             return self.record
                # else:
                #     ret = self.record.add(_d)
                #
                # if ret is not None and ret.type == BarType.CLOSED:
                #     logger.debug(f"Returning closed bar {ret.ts} --> {ret}")
                #     print("---------------")
                #     return ret

                # if self.ltf_updates and _d.ts_event > self.next_ltf_update:
                #     logger.debug(f"next ltf update will be: {pd.to_datetime(_d.ts_event, utc=True).ceil(freq=self.ltf_freq)}")
                #     self.next_ltf_update = pd.to_datetime(_d.ts_event, utc=True).ceil(freq=self.ltf_freq).value
                #     logger.debug(f"Returning open bar {self.record.ts} --> {self.record}")
                #     print("---------------")
                #     return self.record

            time.sleep(self.wait_period/1000)
            wait += self.wait_period

        logging.debug(f"Timed out waiting next record - seq={self.seq}")
        return None

    def write_header(self, file):
        record = f"{self.HEADER}\n"
        if not os.path.exists(file):
            with open(file, "w") as _f:
                _f.write(record)
        if os.path.getsize(file) == 0:
            with open(file, "a") as _fo:
                _fo.write(record)

    # def write_data(self, record: Ohlcv):
    #     if is_next_day(record.ts_event) or self.data_feed_file is None:
    #         __file_date = str(pd.Timestamp(record.ts_event).date())
    #
    #         __symbol = self.config['data']['symbol']
    #         __schema = self.config['data']['schema']
    #
    #         __folder = self.config['data'][__symbol]['folder']
    #         __prefix = self.config['data'][__symbol]['prefix']
    #
    #         self.data_feed_file = f"{__folder}/{__file_date}/{__prefix}-{__file_date}-{__schema}.csv"
    #         os.makedirs(os.path.dirname(self.data_feed_file), exist_ok=True)
    #
    #         self.write_header(self.data_feed_file)
    #
    #         logger.debug("Rolling over to {self.data_feed_file}")
    #
    #     with open(self.data_feed_file, "a") as f:
    #         f.write(f"{record.ts_event},{record.open},{record.high},{record.low},{record.close},{record.volume}\n")
    #         self.last_ts_event = record.ts_event

class FileDatafeed(Datafeed):
    def __init__(self, config: dict):
        super().__init__(config, DataSource.FILE, ltf_freq='1min', agg_freq='1min')
        self.data_feed_file = config['data']['feed_file']
        self.num_records = 0
        self.wait_period = 1
        logger.info(f"Datafeed: source: {str(self.source)} data_freq: {self.data_freq} ltf_freq: {self.ltf_freq} agg_freq={self.agg_freq} ds_file={self.data_feed_file} num_records={self.num_records}")

    def initialize(self):
        self.df = pd.read_csv(self.data_feed_file)
        self.num_records = len(self.df)

    def eof(self):
        return self.seq >= self.num_records

    def next(self) -> Tuple[int, str]:
        msg = None
        if self.seq < self.num_records:
            msg = json.dumps({self.data_freq: self.df.iloc[self.seq].to_dict()})
            self.seq += 1
        return self.seq, msg

class SierraChartDatafeed(Datafeed):
    def __init__(self, config: dict):
        data_freq = config['data']['data_freq']
        super().__init__(config, DataSource.SC, data_freq=data_freq, ltf_freq=data_freq, agg_freq=data_freq)
        self.last_ts_event = pd.Timestamp.now(tz="UTC").value
        self.shm = config["data"][data_freq]["shm"]
        self.lock = config["data"][data_freq]["lock"]
        self.timeout_period = 5000
        logger.info(f"Datafeed: source: {str(self.source)} data_freq: {self.data_freq} ltf_freq: {self.ltf_freq} agg_freq={self.agg_freq} shm={self.shm}")

    def initialize(self):
        _, msg = self.next()
        if msg is not None:
            data = json.loads(msg)
            if "datafeed_file" in data:
                self.df = pd.read_csv(data["datafeed_file"])
                logger.info(f"Historical data {data['datafeed_file']} with {len(self.df)} records loaded")
                return True
            else:
                raise Exception(f"Could not find {self.data_freq} in data {msg}")
        return False

    def eof(self):
        return False

    def next(self) -> Tuple[int, str]:
        self.seq, msg = mmap_read(self.shm, self.lock, self.seq)
        return self.seq, msg

class DatabentoDatafeed(Datafeed):
    def __init__(self, config: dict):
        super().__init__(config, DataSource.DB, agg_freq='1min')
        self.shm = config["data"]["shm"]
        self.lock = config["data"]["lock"]
        self.timeout_period = 5000
        logger.info(f"Datafeed: source: {str(self.source)} data_freq: {self.data_freq} ltf_freq: {self.ltf_freq} agg_freq={self.agg_freq} shm={self.shm}")

    def initialize(self):
        pass

    def eof(self):
        return False

    def next(self) -> Tuple[int, str]:
        self.seq, msg = shm_read_data(self.shm, self.lock, self.seq)
        return self.seq, msg

def create_datafeed(config: dict):
    if config['data']['source'] == 'sc':
        return SierraChartDatafeed(config)
    elif config['data']['source'] == 'db':
        return DatabentoDatafeed(config)
    elif config['data']['source'] == 'file':
        return FileDatafeed(config)
    else:
        raise Exception(f"Unknown source: {config['data']['source']}")