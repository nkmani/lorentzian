import argparse
import json
import os
import platform
from multiprocessing import Lock
from multiprocessing.shared_memory import SharedMemory
import mmap
import pandas as pd
from datetime import datetime, timezone, timedelta
import logging
import logging.handlers
from classifier import settings


MAGIC = 0x59415453  # yats


def is_windows():
    return platform.system() == "Windows"


def get_data(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if 'ts_date' not in df.columns:
        df['ts_date'] = pd.to_datetime(df['ts_event'], utc=True)
        df.set_index('ts_date', inplace=True)
    return df


# Return standard US market session hours (6.30am to 1pm PDT)
def get_session_start_and_end_times() -> (pd.Timestamp, pd.Timestamp):
    _now = pd.Timestamp.now(tz=timezone.utc).normalize()
    # TODO: do this based on DST in effect or not
    return _now.replace(hour=13, minute=30), _now.replace(hour=20, minute=0)


def get_session_times_from_date(_date: str) -> (pd.Timestamp, pd.Timestamp):
    _d = pd.Timestamp(_date, tz=timezone.utc)
    return _d.replace(hour=13, minute=30), _d.replace(hour=18, minute=0)


def get_session_times_from_date_time(_date: str, _start_time: str, _end_time: str) -> (pd.Timestamp, pd.Timestamp):
    return pd.Timestamp(f"{_date} {_start_time}", tz=timezone.utc), pd.Timestamp(f"{_date} {_end_time}", tz=timezone.utc)


def get_session_data(df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    df_ni = df.reset_index()
    q = f"ts_event >= {start_date.value} and ts_event < {end_date.value}"
    df_ni = df_ni.query(q)
    return df_ni


def set_logger(config, log_file, verbose) -> None:
    root = logging.getLogger()

    _format = "%(asctime)s|%(levelname)s|%(name)s|%(message)s"
    if 'log' in config and 'format' in config['log']:
        _format = config['log']['format']

    _log_level = logging.DEBUG
    if 'log' in config and 'level' in config['log']:
        _log_level = config['log']['level']
        if _log_level == 'debug':
            _log_level = logging.DEBUG
        elif _log_level == 'info':
            _log_level = logging.INFO
        elif _log_level == 'warning':
            _log_level = logging.WARNING
        elif _log_level == 'error':
            _log_level = logging.ERROR

    if log_file is not None:
        logging.basicConfig(filename=log_file, format=_format, level=_log_level)
        handler = logging.handlers.TimedRotatingFileHandler(filename=log_file, utc=True, when='midnight', interval=1, backupCount=0)
        root.addHandler(handler)

    if verbose or log_file is None:
        import sys
        console = logging.StreamHandler(stream=sys.stdout)
        console.setFormatter(logging.Formatter(fmt=_format))
        console.setLevel(_log_level)
        root.addHandler(console)

    root.setLevel(_log_level)


def is_next_day(ts1, ts2) -> bool:
    return pd.Timestamp(ts2).date() == pd.Timestamp(ts1).date() + timedelta(days=1)


if is_windows():
    import win32event


    def open_mutex(_mutex_name):
        return win32event.OpenMutex(0x00100000 + 0x00020000 + 0x00001, False, _mutex_name)


    def create_mutex(_mutex_name):
        return win32event.CreateMutex(None, False, _mutex_name)


    def create_or_attach_mmap(_shm_name: str, _shm_size: int, _study: str):
        return mmap.mmap(-1, _shm_size, _shm_name, mmap.ACCESS_WRITE)


    def mmap_write(_mmap, _mutex, _seq: int, _msg: str) -> None:
        win32event.WaitForSingleObject(_mutex, win32event.INFINITE)

        _magic = MAGIC
        _mmap.seek(0)
        _mmap.write(_magic.to_bytes(4, byteorder="little"))
        _mmap.seek(4)
        _mmap.write(_seq.to_bytes(4, byteorder="little"))
        _mmap.seek(8)
        _l = len(_msg)
        _mmap.write(_l.to_bytes(4, byteorder="little"))
        _mmap.seek(12)
        _mmap.write(_msg.encode("ascii"))

        win32event.ReleaseMutex(_mutex)


    def mmap_read(_mmap, _mutex, _next: int) -> (int, str):
        logger = logging.getLogger()

        result = win32event.WaitForSingleObject(_mutex, 100)

        if result == win32event.WAIT_OBJECT_0:
            _mmap.seek(0)
            _magic = int.from_bytes(_mmap.read(4), byteorder="little")
            if _magic != MAGIC:
                logger.error(f"Expected magic '{MAGIC}' got {_magic}")
                win32event.ReleaseMutex(_mutex)
                return -1, None

            _mmap.seek(4)
            _seq = int.from_bytes(_mmap.read(4), byteorder="little")
            if _seq == _next:
                win32event.ReleaseMutex(_mutex)
                return _seq, None

            _mmap.seek(8)
            _l = int.from_bytes(_mmap.read(4), byteorder="little")

            _mmap.seek(12)
            _msg = str(_mmap.read(_l).decode("utf-8"))

            win32event.ReleaseMutex(_mutex)
            return _seq, _msg

        elif result == win32event.WAIT_TIMEOUT:
            win32event.ReleaseMutex(_mutex)
            return -1, None

        elif result == win32event.WAIT_ABANDONED:
            win32event.CloseHandle(_mutex)
            return -2, None


def setup_data_shm_and_lock(_config: dict) -> None:
    logger = logging.getLogger()

    freqs = _config['data']['freqs']
    for freq in freqs:
        _shm_name = f"{_config['data']['shm_name']}-{_config['data']['symbol']}-{freq}"
        _shm_size = _config['data']['shm_size']

        _config['data'][freq] = {}
        if is_windows():
            _config['data'][freq]['lock'] = open_mutex(f"{_shm_name}-mutex")
            _config['data'][freq]['shm'] = create_or_attach_mmap(_shm_name, _shm_size, "")
        else:
            _config['data'][freq]['lock'] = Lock()
            _config['data'][freq]['shm'] = create_or_attach_shm(_shm_name, _shm_size)
        logger.debug(f"Data shared memory {_shm_name} size {_shm_size} setup complete")


def get_logger():
    return logging.getLogger()

def setup_signal_shm_and_lock(_config: dict, _study: str) -> None:
    logger = logging.getLogger()

    _shm_name = f"{_config['signal']['shm_name']}-{_study}"
    _shm_size = _config['signal']['shm_size']

    _config['signal'][_study] = {}
    if is_windows():
        mutex_name = f"{_shm_name}-mutex"
        _config['signal'][_study]['lock'] = create_mutex(mutex_name)
        _config['signal'][_study]['shm'] = create_or_attach_mmap(_shm_name, _shm_size, _study)
    else:
        _config['signal'][_study]['lock'] = Lock()
        _config['signal'][_study]['shm'] = create_or_attach_shm(_shm_name, _shm_size)
    logger.debug(f"Signal shared memory {_shm_name} size {_shm_size} setup complete")


def create_or_attach_shm(_name: str, _size: int):
    try:
        _shm = SharedMemory(name=_name, create=False)
    except Exception as e:
        logger.warning(f"Could not attach to memory; try to creat it - {e}")
        _shm = SharedMemory(name=_name, size=_size, create=True)

        # init message
        seq = 0
        msg = "{}"
        _shm.buf[0:4] = MAGIC.to_bytes(4, byteorder="little")
        _shm.buf[4:8] = seq.to_bytes(4, byteorder="little")
        seq = 2
        _shm.buf[8:12] = seq.to_bytes(4, byteorder="little")
        _shm.buf[12:14] = bytearray(msg, encoding="ascii")
    return _shm


def shm_write_data(_shm: SharedMemory, _lock: Lock, _seq: int, _msg: str) -> None:
    logger.debug(f"Writing to shm: {_shm.name} seq: {_seq} msg: {_msg}")

    # Don't do this:
    # with _shm.buf as _buf
    #
    # This will release the shared memory at the end of the with block
    _lock.acquire()
    _shm.buf[0:4] = MAGIC.to_bytes(4, byteorder="little")
    _shm.buf[4:8] = _seq.to_bytes(4, byteorder="little")

    _l = len(_msg)
    _shm.buf[8:12] = _l.to_bytes(4, byteorder="little")

    _shm.buf[12:(_l+12)] = bytearray(_msg, encoding="ascii")
    _lock.release()


def shm_read_data(_shm: SharedMemory, _lock: Lock, _next: int) -> (int, str):
    # see comment in shm_write_data
    # with _shm.buf as _buf:
    _lock.acquire()
    _magic = int.from_bytes(_shm.buf[0:4], byteorder="little")
    if _magic != MAGIC:
        logger.error(f"Expected magic '{MAGIC}' got {_magic}")
        _lock.release()
        return -1, None

    _seq = int.from_bytes(_shm.buf[4:8], byteorder="little")

    if _seq == _next:
        _lock.release()
        return _seq, None

    _l = int.from_bytes(_shm.buf[8:12], byteorder="little")

    _msg = str(_shm.buf[12:(_l + 12)], encoding="ascii")
    _lock.release()

    logger.debug(f"Read from shm: {_shm.name} seq: {_seq} msg: {_msg}")

    return _seq, _msg

def market_closed() -> bool:
    _mc = False
    _now_utc = datetime.now(timezone.utc)
    _wd = _now_utc.weekday()
    if (_wd == 4 and _now_utc.hour >= 22) or (_wd == 5) or (_wd == 6 and _now_utc.hour < 23) or (22 <= _now_utc.hour < 23):
        _mc = True
    return _mc

#
# different type of arg parsers
#
def get_default_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default="config.json")
    parser.add_argument('--symbol', default="mes", help="Symbol to use, e-g mes, m6e")
    parser.add_argument('--log-file', type=str)
    parser.add_argument('-l', '--log-level', default="info")
    parser.add_argument('-v', '--verbose', action="store_true")
    parser.add_argument('--signal-file', type=str)
    return parser

def get_signaling_argparser() -> argparse.ArgumentParser:
    parser = get_default_argparser()
    parser.add_argument('--study', type=str, default="lc-1min")
    parser.add_argument('--datafeed-source', default="sc")
    parser.add_argument('--session-start', default='14:30:00', type=str)
    parser.add_argument('--session-end', default='22:00:00', type=str)
    parser.add_argument("--lorentzian-settings", type=str, default="settings-defaults.json")
    return parser

def get_config(parser_type: str) -> dict:
    if parser_type == "signaling":
        parser = get_signaling_argparser()
    else:
        parser = get_default_argparser()
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f'{args.config} does not exist')
        exit(1)

    config = {}
    with open(args.config, "r") as f:
        config = json.load(f)
        config["args"] = args

    config["log"]["level"] = args.log_level
    set_logger(config, args.log_file, args.verbose)

    return config

def load_study(config: dict, study: str):
    config_signal = config.get('signal', {})
    config_signal_study = config_signal.get('study', {})
    indicators = config_signal_study.get('indicators', {})
    if "LORENTZIAN" in indicators.keys():
        lorentzian = indicators["LORENTZIAN"]
        if os.path.exists(lorentzian["settings_file"]):
            lorentzian["settings"] = settings.generate_settings(lorentzian["settings_file"])
        else:
            lorentzian["settings"] = settings.default_settings()
