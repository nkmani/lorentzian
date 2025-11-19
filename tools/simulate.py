import argparse
import csv
import glob
import operator
import os
import sys
import math
import threading
import queue

import pandas as pd
from classifier.setting import *

# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.insert(0, parent_dir)

from common.functions import *

from ta.volatility import average_true_range as atr

import time

from indicators.lorentzian import Lorentzian

logger = get_logger()

class Stats:
    symbol: str
    date: str
    profit: float
    win_trades: int
    loss_trades: int
    iterations: int
    lock: threading.Lock

    def __init__(self, **kwargs):
        self.changed_settings = None
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.profit = -math.inf
        self.iterations = 0
        self.win_trades = 0
        self.loss_trades = 0
        self.results = []
        self.lock = threading.Lock()

    def __str__(self):
        return f"symbol: {self.symbol} date: {self.date} profit: {self.profit:.2f} win_trades: {self.win_trades} loss_trades: {self.loss_trades} iterations: {self.iterations} changed_settings: {self.changed_settings}"
        # for attr, value in self.__dict__.items():
        #     if attr != "results":
        #         s += f"{attr}: {value} "
        # return s

    def update(self, profit, win_trades, loss_trades, settings):
        self.lock.acquire()
        result = [self.symbol, self.date, f"{profit:.2f}", win_trades, loss_trades, str(settings.get_changed_settings())]
        self.results.append(result)
        self.iterations += 1
        if profit > self.profit:
            self.profit = profit
            self.win_trades = win_trades
            self.loss_trades = loss_trades
            self.changed_settings = settings.get_changed_settings()
        logger.debug(f"{str(result)} iteration={self.iterations}")
        self.lock.release()

    def top_n_a(self, n: int):
        sorted_results = sorted(self.results, key=operator.itemgetter(2), reverse=True)
        return sorted_results[:n]

    def top_n(self, n: int):
        sorted_results = sorted(self.results, key=operator.itemgetter(2), reverse=True)
        s = ""
        for i in range(min(len(sorted_results), n)):
            s += f"{str(sorted_results[i])}\n"
        return s

    def dump_stats(self, csv_file):
        open_mode = "a"
        if not os.path.exists(csv_file):
            open_mode = "w"
        with open(csv_file, open_mode, newline='') as f:
            writer = csv.writer(f, delimiter=',')
            if open_mode == "w":
                writer.writerow(['symbol', 'date', 'profit', 'win_trades', 'loss_trades', 'changed_settings'])
            writer.writerows(self.results)

class AggStats:
    records: list

    def __init__(self):
        self.records = []

    def update(self, stats: Stats):
        self.records.extend(stats.top_n_a(10))

    def report(self):
        df = pd.DataFrame(self.records, columns=['symbol', 'date', 'profit', 'win_trades', 'loss_trades', 'changed_settings'])

        # sort by date, profit and assign ranks
        sorted_df = df.sort_values(by=['date', 'profit'], ascending=[True, False])
        for date in sorted_df.date.unique():
            rank = 10
            for index, _ in sorted_df[sorted_df['date'] == date].iterrows():
                sorted_df.loc[index, 'rank'] = rank
                rank -= 1

        agg_ranks = []
        for setting in sorted_df.changed_settings.unique():
            grouped_setting = sorted_df[sorted_df['changed_settings'] == setting]
            agg_rank = 0
            agg_profit = 0
            agg_trades = 0
            count = 0
            for _, row in grouped_setting.iterrows():
                agg_rank += row['rank']
                agg_profit += row['profit']
                agg_trades += row['loss_trades'] + row['win_trades']
                count += 1
            agg_ranks.append([setting, agg_rank, agg_profit, agg_trades, count])

        sorted_agg_ranks = sorted(agg_ranks, key=operator.itemgetter(1), reverse=True)
        logger.info("*" * 64)
        logger.info("Aggregated Ranks: ")
        for row in sorted_agg_ranks:
            logger.info(f"setting: {row[0]} agg-rank: {row[1]} agg-profit: {row[2]} agg-trades: {row[3]} count: {row[4]}")
        logger.info("*" * 64)

def simulate(df: pd.DataFrame, settings: Settings):
    settings.init(df)
    lc = Lorentzian({'settings': settings})
    signal, _ = lc.process(df)
    lc_df = lc.df

    buy_or_sell = lc_df[(lc_df['start_long_trade'].notna() | lc_df['start_short_trade'].notna())]
    if len(buy_or_sell) == 0:
        logger.info(f"No buy or sell trades in the dataset to simulate trading")
        return 0.0, 0, 0

    logger.info(f"No of Buy or sell trades: {len(buy_or_sell)}")

    lc_df['shifted_open'] = lc_df['open'].shift(-1)

    position = 0
    num_trades = 0
    win_trades = 0
    loss_trades = 0
    profit = 0.0
    net_profit = 0.0
    entry_price = 0.0
    exit_price = 0.0
    sl_price = 0.0
    tp_price = 0.0

    stop_loss = settings.stop_loss
    take_profit = settings.take_profit

    point_value = settings.point_value
    trade_commission = settings.trade_commission

    logger.info(f"iteration: {settings.iteration} settings: {settings.get_changed_settings()}")

    start_time, end_time = get_session_times_from_date(str(df.index[-1].date()))
    start_time_ts = start_time.value
    end_time_ts = end_time.value

    index = df.index[-1]
    for index, row in lc_df.iterrows():
        if row.ts_event < start_time_ts:
            continue
        elif row.ts_event > end_time_ts:
            break
        # print(f"[{index}] :: open: {row.open} high: {row.high} low: {row.low} close: {row.close}")
        if position == 0 and (not math.isnan(row.start_long_trade) or not math.isnan(row.start_short_trade)):
            entry_price = row.shifted_open
            if math.isnan(entry_price):
                continue
            num_trades += 1
            if not math.isnan(row.start_long_trade):
                position = position + 1
                sl_price = entry_price - stop_loss
                tp_price = entry_price + take_profit
                logger.debug(f"[{index}] :: Enter Long Trade: entry_price: {entry_price} stop_loss: {sl_price} take_profit: {tp_price} atr: {row.atr}")
            else:
                position = position - 1
                sl_price = entry_price + stop_loss
                tp_price = entry_price - take_profit
                logger.debug(f"[{index}] :: Enter Short Trade: entry_price: {entry_price} stop_loss: {sl_price} take_profit: {tp_price} atr: {row.atr}")
        elif position != 0:
            if position > 0:
                if row.low < sl_price:
                    profit -= (entry_price - sl_price)
                    loss_trades += 1
                    position = position - 1
                    logger.debug(f"[{index}] :: Stop Loss: {row.low} profit: {profit} win_trades: {win_trades} loss_trades: {loss_trades} atr: {row.atr}")
                elif row.high > tp_price:
                    profit += (tp_price - entry_price)
                    win_trades += 1
                    position = position - 1
                    logger.debug(f"[{index}] :: Take Profit: {row.high} profit: {profit} win_trades: {win_trades} loss_trades: {loss_trades} atr: {row.atr}")
            elif position < 0:
                if row.low < tp_price:
                    profit += (entry_price - tp_price)
                    win_trades += 1
                    position = position + 1
                    logger.debug(f"[{index}] :: Take Profit: {row.high} profit: {profit} win_trades: {win_trades} loss_trades: {loss_trades} atr: {row.atr}")
                elif row.high > sl_price:
                    profit -= (sl_price - entry_price)
                    loss_trades += 1
                    position = position + 1
                    logger.debug(f"[{index}] :: Stop Loss: {row.high} profit: {profit} win_trades: {win_trades} loss_trades: {loss_trades} atr: {row.atr}")

    # close any open position
    if position != 0:
        logger.debug(f"Closing open position: {position}")
        last_row = lc_df.loc[index]
        if position > 0:
            if last_row.close > entry_price:
                profit += (last_row.close - entry_price)
                win_trades += 1
            else:
                profit -= (entry_price - last_row.close)
                loss_trades += 1
        elif position < 0:
            if last_row.close < entry_price:
                profit += (entry_price - last_row.close)
                win_trades += 1
            else:
                profit -= (last_row.close - entry_price)
                loss_trades += 1

    if num_trades > 0:
        win_rate = win_trades /num_trades * 100
        net_profit = profit * point_value - trade_commission * num_trades
    else:
        win_rate = 0
    logger.debug(f"Profit: {net_profit:.2f} Points: {profit:.2f} Number of trades: {num_trades} Win Trades: {win_trades} Loss trades: {loss_trades} Win Rate: {win_rate:.2f}")
    return net_profit, win_trades, loss_trades

def worker(q):
    while True:
        try:
            item = q.get()
            if item is None:
                break
            df = item["df"]
            settings = item["settings"]
            stats = item["stats"]
            logger.info(f"Processing: {len(df)} records with {settings.get_changed_settings()} by thread {threading.current_thread().name}" )
            profit, win_trades, loss_trades = simulate(df, settings)
            stats.update(profit, win_trades, loss_trades, settings)
        finally:
            q.task_done()

def grid_search_optimal_settings(symbol: str, data_file: str, config_file: str, result_file: str, num_threads: int = 1) -> Stats | None:
    df = get_data(data_file)
    if len(df) < 1200:
        logger.warning(f"Not enough data found for {data_file} - length: {len(df)}")
        return None

    date = str(df.index[-1].date())

    settings_iterator = SettingsIterator(config_file)

    logger.info("-" * 64)
    logger.info(f"Grid Search Optimal Settings for {data_file} with {len(df)} records and {settings_iterator.limit} iterations")

    stats = Stats(symbol=symbol, date=date)

    df['atr'] = atr(df['high'], df['low'], df['close'], window=7)

    q = queue.Queue()
    for settings in settings_iterator:
        q.put({"df": df, "settings": settings, "stats": stats})

    threads = []
    for i in range(num_threads):
        thread = threading.Thread(target=worker, args=(q,))
        threads.append(thread)
        thread.start()

    q.join()

    for _ in range(num_threads):
        q.put(None)
    for thread in threads:
        thread.join()  # Wait for threads to finish
    logger.info("All tasks completed and threads terminated.")

    if result_file:
        stats.dump_stats(result_file)

    logger.info(f"{stats}\n{stats.top_n(5)}")
    logger.info("=" * 64)

    return stats

def run(args):
    config = {
        "log": {
            "level": args.log_level,
        }
    }
    set_logger(config, args.log_file, args.verbose)

    if args.config_file is not None:
        if not os.path.exists(args.config_file):
            logger.error(f"Config file not found: {args.config_file}")
            return

    start_time = time.time()
    if args.data_file:
        if not os.path.exists(args.data_file):
            logger.error(f"Data file not found: {args.data_file}")
            return
        grid_search_optimal_settings(args.symbol, args.data_file, args.config_file, args.result_file)
    elif args.folder:
        agg_stats = AggStats()
        folders = glob.glob(os.path.join(args.folder, "20*"))
        folders.sort()
        for folder in folders:
            files = glob.glob(os.path.join(folder, "*.csv"))
            for file in files:
                stats = grid_search_optimal_settings(args.symbol, file, args.config_file, args.result_file)
                if stats is not None:
                    agg_stats.update(stats)
        agg_stats.report()

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Function executed in {elapsed_time:.4f} seconds.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-file", help="data file name", type=str)
    parser.add_argument("-r", "--result-file", help="data file name", type=str, default="results.csv")
    parser.add_argument("-l", "--log-file", help="logger file name")
    parser.add_argument("--log-level", help="log level", type=str, default="info")
    parser.add_argument("-f", "--folder", help="data folder", type=str)
    parser.add_argument("-s", "--symbol", help="3 letter symbol name", type=str, required=True)
    parser.add_argument("-v", "--verbose", help="verbose output", action="store_true", default=False)
    parser.add_argument("-c", "--config-file", help="config file", type=str)
    parser.add_argument("-t", "--thread-count", help="no of threads", type=int, default=1)
    parser.add_argument("--session-start", help="session start time", type=str, default="13:30")
    parser.add_argument("--session-end", help="session end time", type=str, default="20:00")
    run(parser.parse_args())