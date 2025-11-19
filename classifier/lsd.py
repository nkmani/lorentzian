from datetime import time
import time as timer
import logging

from classifier.setting import *
from classifier.distance import get_lorentzian_predictions
from classifier.signal import generate_signals

import pandas as pd
import numpy as np
import math

logger = logging.getLogger('classifier')

class LorentzianSpaceDistanceIndictor:
    """Lorentzian Space Distance Indictor

    Calculate the lorentzian space distance of time series market data.

    Background information can be found here (behind pay-wall)
    https://ai-edge.io/docs/category/lorentzian-classification

    Pine script version of the original implementation (open source):
    https://www.tradingview.com/script/WhBzgfDu-Machine-Learning-Lorentzian-Classification/

    Python adaptation of the same:
    https://pypi.org/project/advanced-ta/

    """
    def __init__(self, data: pd.DataFrame, settings: Settings):
        self.df = data.copy()
        self.settings = settings

    def lsd(self) -> pd.DataFrame:
        start_time = timer.time()
        self.df['predictions'] = get_lorentzian_predictions(self.settings)
        elapsed_time = timer.time() - start_time
        logger.debug(f"Evaluation completed in {elapsed_time} seconds.")

        self.df = generate_signals(self.settings, self.df)

        return self.df