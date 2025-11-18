import threading
import time

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from common.functions import *
from signaling.signaling import Signaling


def study_signal_thread(config, study):
    start = config['session']['start']
    end = config['session']['end']
    get_logger().info("Signaling started for study %s ...Start: %s End: %s", study, start, end)

    load_study(config, study)

    signaling = Signaling(config, study)
    get_logger().info("Running signaling Study: %s Start: %s End: %s with %s", study, start, end, signaling.init_msg)

    # initialize data, signal channels
    while not signaling.ready_for_streaming():
        signaling.init()
        time.sleep(1)

    # data stream is ready for processing
    signaling.stream()


def run_services(config: dict, start, end):
    threads = []

    logger = get_logger()

    logger.info("Starting signalling thread(s)")

    study = config['signal']['study']['name']
    thread = threading.Thread(target=study_signal_thread, name=f"signal-{study}", args=(config, study))
    thread.daemon = True
    thread.start()

    threads.append(thread)

    while True:
        time.sleep(1)
        for thread in threads:
            thread.join()


def main():
    """
    Start up the signaling function in a thread

    :return: Does not return. Will return when the session ends.
    """
    config = get_config("signaling")

    logger = get_logger()

    threads = []
    logger.info("Starting signalling thread(s)")

    study = config['signal']['study']['name']
    thread = threading.Thread(target=study_signal_thread, name=f"signal-{study}", args=(config, study))
    thread.daemon = True
    thread.start()

    threads.append(thread)

    while True:
        time.sleep(1)
        for thread in threads:
            thread.join()

    # process = mp.Process(target=run_services, args=(config, 0, 0))
    # process.daemon = True
    # process.start()
    #
    # logger.info(f"PID: {process.pid} started")
    # with open("status.txt", "w") as f:
    #     f.write(str(process.pid))
    #
    # while True:
    #     time.sleep(1)
    #     process.join()


if __name__ == "__main__":
    main()
