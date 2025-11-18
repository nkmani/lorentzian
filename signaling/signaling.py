import pandas as pd
from common.constants import *

import logging
import traceback

from datafeed.datafeed import Datafeed, create_datafeed
from indicators.indicator import Indicator
from common.functions import setup_data_shm_and_lock, setup_signal_shm_and_lock, is_windows, mmap_write, shm_write_data
from indicators.lorentzian import Lorentzian


class Signaling:

    indicator: Indicator
    df: pd.DataFrame
    datafeed: Datafeed = None

    def __init__(self, config: dict, study: str):
        self.config = config
        self.study = study

        self.config_study = config['signal']['study']
        self.signal_freq = self.config_study['signal_freq']
        self.signal_freq_seconds = pd.Timedelta(self.signal_freq).total_seconds()

        self.ltf_freq = self.config_study['ltf_freq'] if 'ltf_freq' in self.config_study else self.signal_freq
        self.ltf_freq_seconds = pd.Timedelta(self.ltf_freq).total_seconds()

        self.use_ltfs = True if self.signal_freq_seconds != self.ltf_freq_seconds else False

        self.signal_shm = None
        self.signal_lock = None

        self.init_msg = f"signal: {self.signal_freq} data: {self.ltf_freq}\n"
        self.flags = 0

        self.logger = logging.getLogger()

    def _prep_data(self, df: pd.DataFrame) -> pd.DataFrame:
        self.df = df.copy()
        if 'ts_event' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['ts_event'], utc=True)
        self.df.set_index('date', inplace=True)
        return self.df

    def init(self):
        self.logger.debug(f"Initializing signaling {self.study}")
        if not self.flags & DF_SHM_INITIALIZED:
            try:
            # setup shm for data
                setup_data_shm_and_lock(self.config)
                self.datafeed = create_datafeed(self.config)
                self.flags |= DF_SHM_INITIALIZED
            except Exception as e:
                self.logger.warning(f"Failed to initialize datafeed shared memory: {e}")
                return

        if not self.flags & DF_HISTORICAL_DATA_LOADED:
            try:
                if self.datafeed is None:
                    self.datafeed = create_datafeed(self.config)
                if not self.datafeed.initialize():
                    self.logger.warning(f"Failed to setup historical datafeed")
                    return
                else:
                    if self.datafeed.df is not None and len(self.datafeed.df) > 0:
                        self.df = self._prep_data(self.datafeed.df)
                        self.flags |= DF_HISTORICAL_DATA_LOADED
                    else:
                        self.logger.warning("Failed to initialize historical datafeed - no data found")
                        return
            except Exception as e:
                self.logger.warning(f"Failed to initialize datafeed shared memory: {e}")
                return

        if not self.flags & DF_INDICATORS_LOADED:
            try:
                lorentzian_config = self.config_study['indicators']['LORENTZIAN']
                self.indicator = Lorentzian(lorentzian_config)
                signal, elapsed = self.indicator.run(self.df)
                self.logger.info(f"Signaling - ran {self.indicator.name} on initial data signal: {signal} elapsed: {elapsed}")
                self.flags |= DF_INDICATORS_LOADED
            except Exception as e:
                self.logger.warning(f"Failed to initialize indicators: {e}")
                self.logger.warning(traceback.format_exc())
                return

        # setup shm for signal
        if not self.flags & DF_STREAMING_READY:
            try:
                setup_signal_shm_and_lock(self.config, self.study)
                self.signal_shm = self.config['signal'][self.study]['shm']
                self.signal_lock = self.config['signal'][self.study]['lock']
                self.flags |= DF_STREAMING_READY
            except Exception as e:
                self.logger.warning(f"Failed to initialize signal shared memory: {e}")
                return

    def ready_for_streaming(self):
        if self.flags & DF_STREAMING_READY:
            return True
        return False

    def send_signal(self, signal_seq, signal):
        if is_windows():
            mmap_write(self.signal_shm, self.signal_lock, signal_seq, signal.to_json())
        else:
            shm_write_data(self.signal_shm, self.signal_lock, signal_seq, signal.to_json())

    def stream(self):
        self.logger.info(f"Signalling stream for {self.study} signal_freq: {self.signal_freq}")

        signal_seq = 1

        while not self.datafeed.eof():
            rec = self.datafeed.process()
            if rec is not None:
                # make sure to run the analysis on bar close (and if there were signals in the open bar
                # check if the signal concurs with the closed bar)

                # make sure on signal_freq boundaries, the agg records
                # are refreshed

                self.logger.info(f"[{self.study}] Processing record {rec}")

                if self.use_ltfs:
                    ltf_dt = pd.to_datetime(rec.ts_event, utc=True)
                    dt = ltf_dt.ceil(freq=self.signal_freq)
                    ltf_idx = self.ltfs.count - int((dt - ltf_dt).seconds / self.ltf_freq_seconds) - 1
                else:
                    dt = pd.to_datetime(rec.ts_event, utc=True)

                self.df.loc[dt] = rec.to_dict()
                signal, elapsed = self.indicator.run(self.df)

                self.send_signal(signal_seq, signal)
                signal_seq += 1

                self.logger.info(f"[{self.study}] Generated signal {signal}")

