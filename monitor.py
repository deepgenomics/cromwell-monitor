#!/usr/bin/env python3

import logging
from signal import SIGTERM, signal
from time import time

import gcp_monitor

logging.getLogger().setLevel(logging.INFO)


# Detect container termination
def signal_handler(object, signum, frame):
    object.container_running = False


def main():
    """
    This function continuously measures runtime metrics every MEASUREMENT_TIME_SEC,
    and reports them to Stackdriver Monitoring API every REPORT_TIME_SEC.

    However, if it detects a container termination signal,
    it *should* report the final metric
    right after the current measurement, and then exit normally.
    """

    gcp_instance = gcp_monitor.gcp_monitor_variables()

    signal(SIGTERM, signal_handler(gcp_instance))

    gcp_monitor.reset(gcp_instance)
    while gcp_instance.container_running:
        gcp_monitor.measure(gcp_instance)
        if (
            not gcp_instance.container_running
            or (time() - gcp_instance.last_time) >= gcp_instance.REPORT_TIME_SEC
        ):
            gcp_monitor.report(gcp_instance)
            gcp_monitor.reset(gcp_instance)
    exit(0)


if __name__ == "__main__":
    main()
