import time

import pandas as pd
import numpy as np
from classifier.settings import default_features, default_settings, Settings
from classifier.distance import get_lorentzian_predictions
from common import get_data


def calculate_distance(df, features, idx):
    l = len(df)
    dists = np.array([0.0] * l)
    for i in range(l):
        val = 0.0
        for feature in features:
            val = val + np.log(1 + np.abs(feature[idx] - feature[i]))
        dists[i] = val
    return dists


if __name__ == '__main__':
    _df = get_data("data/2025-10-17.csv")
    _df = _df[:50]
    _settings = Settings()
    _settings.init(_df)
    start_time = time.time()
    _dists = calculate_distance(_df, _settings.features, 25)
    elapsed_time = time.time() - start_time
    print(f"Single Loop distance calculation: Finished in {elapsed_time} -- {len(_dists)} distances")

    start_time = time.time()
    _predictions_v1 = get_lorentzian_predictions(_settings)
    elapsed_time = time.time() - start_time
    print(
        f"Recalc algo v1: Finished in {elapsed_time} -- {len(_predictions_v1)} predictions iterations: {_settings.iteration}")

    _settings.distance_algo_type = 2
    start_time = time.time()
    _predictions = get_lorentzian_predictions(_settings)
    elapsed_time = time.time() - start_time
    print(
        f"Recalc algo v2: Finished in {elapsed_time} -- {len(_predictions)} predictions iterations: {_settings.iteration}")

    print(np.array_equal(_predictions_v1, _predictions))
