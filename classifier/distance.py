from abc import abstractmethod, ABCMeta

import numpy as np
import math
from .settings import Settings, Direction
from constants import *


class Distances(metaclass=ABCMeta):
    max_bars_back: int
    max_bars_back_index: int

    def __init__(self, settings: Settings):
        self.settings = settings
        self.max_bars_back = settings.max_bars_back
        self.max_bars_back_index = settings.max_bars_back_index()
        self.size = (len(settings.data) - self.max_bars_back_index)
        self.features = settings.features


class DistancesV1(Distances):
    batchSize = 50
    lastBatch = 0

    def __init__(self, settings):
        super().__init__(settings)
        self.dists = np.array([[0.0] * self.size] * self.batchSize)
        self.rows = np.array([0.0] * self.batchSize)

    def __getitem__(self, item):
        batch = math.ceil((item + 1) / self.batchSize) * self.batchSize
        if batch > self.lastBatch:
            self.dists.fill(0.0)
            for feature in self.features:
                self.rows.fill(0.0)
                fBatch = feature[(self.max_bars_back_index + self.lastBatch):(self.max_bars_back_index + batch)]
                self.rows[:fBatch.size] = fBatch.reshape(-1, )
                val = np.log(1 + np.abs(self.rows.reshape(-1, 1) - feature[:self.size].reshape(1, -1)))
                self.dists += val
            self.lastBatch = batch

        return self.dists[item % self.batchSize]


class DistancesV2(Distances):

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.bar_index = 0

    def get_distance(self, bar: int, idx: int) -> float:
        val = 0.0
        for feature in self.settings.features:
            # val = val + np.log(1 + np.abs(feature[bar] - feature[idx]))
            val = val + math.log(1 + abs(feature[bar] - feature[idx]))
        return val


def get_lorentzian_predictions(settings: Settings):
    max_bars_back_index = settings.max_bars_back_index()

    predictions = []
    distances = []
    neighbors = []

    def get_lorentzian_prediction_v2(dist: DistancesV2, bar_index: int):
        last_distance = -1.0
        for i in range(max_bars_back_index, bar_index):
            d = dist.get_distance(bar_index, i)
            if d >= last_distance and i % settings.sample_interval:
                last_distance = d
                distances.append(d)
                neighbors.append(i)
                predictions.append(round(settings.y_train[i]))
                if len(predictions) > settings.neighbors_count:
                    last_distance = distances[round(settings.neighbors_count * 3 / 4)]
                    distances.pop(0)
                    predictions.pop(0)
                    neighbors.pop(0)
        with open("v2.log", "a") as f:
            f.write(f"bar_index: {bar_index}, distances: {distances}, neighbors: {neighbors}\n")
        return sum(predictions)

    def get_lorentzian_prediction_v1():
        for bar_index in range(max_bars_back_index):
            yield 0

        for bar_index in range(max_bars_back_index, len(settings.src)):
            last_distance = -1.0
            span = min(settings.max_bars_back, bar_index + 1)
            for i, d in enumerate(dists[bar_index - max_bars_back_index][:span]):
                if d >= last_distance and i % settings.sample_interval:
                    last_distance = d
                    distances.append(d)
                    neighbors.append(i)
                    predictions.append(round(settings.y_train[i]))
                    if len(predictions) > settings.neighbors_count:
                        last_distance = distances[round(settings.neighbors_count * 3 / 4)]
                        distances.pop(0)
                        predictions.pop(0)
                        neighbors.pop(0)
            if settings.debug_flags & DUMP_DISTANCE_DATA:
                with open("v1.log", "a") as dump_file:
                    dump_file.write(f"bar_index: {bar_index}, distances: {distances}, neighbors: {neighbors}\n")
            yield sum(predictions)

    if settings.distance_algo_type == 1:
        if settings.debug_flags & DUMP_DISTANCE_DATA:
            with open("v1.log", "w") as f:
                f.write("\n")
        dists = DistancesV1(settings)
        _predictions = np.array([p for p in get_lorentzian_prediction_v1()])
        return _predictions
    elif settings.distance_algo_type == 2:
        dists = DistancesV2(settings)
        if settings.debug_flags & DUMP_DISTANCE_DATA:
            with open("v2.log", "w") as f:
                f.write("\n")
        _predictions = [0 for _ in range(max_bars_back_index)]
        for idx in range(max_bars_back_index, len(settings.src)):
            _predictions.append(get_lorentzian_prediction_v2(dists, idx))
        return np.array(_predictions)

    return np.array([])
