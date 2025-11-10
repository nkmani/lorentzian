import pandas as pd
from datetime import datetime, timezone, timedelta
import logging
import logging.handlers


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
