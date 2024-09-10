#!/usr/bin/env python3

import logging
from signal import SIGTERM, signal
from time import time

import pynvml

import gcp_monitor

logging.getLogger().setLevel(logging.INFO)

# flag to keep track of container termination
container_running = True


# Detect container termination
def signal_handler(signum, frame):
    global container_running
    container_running = False


def main():
    """
    This function continuously measures runtime metrics every MEASUREMENT_TIME_SEC,
    and reports them to Stackdriver Monitoring API every REPORT_TIME_SEC.

    However, if it detects a container termination signal,
    it *should* report the final metric
    right after the current measurement, and then exit normally.
    """

    nvml_ok = True
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as e:
        # expected if the machine does not have an NVIDIA GPU
        logging.info(f"NVML initialization failed (probably no GPUs): {e}")
        nvml_ok = False

    gcp_instance, metrics_client = gcp_monitor.initialize_gcp_variables(nvml_ok)

    try:
        signal(SIGTERM, signal_handler)

        gcp_instance = gcp_monitor.reset(gcp_instance)
        while container_running:
            gcp_instance = gcp_monitor.measure(gcp_instance)
            if (
                not container_running
                or (time() - gcp_instance["last_time"])
                >= gcp_instance["REPORT_TIME_SEC"]
            ):
                gcp_instance = gcp_monitor.report(gcp_instance, metrics_client, nvml_ok)
                gcp_instance = gcp_monitor.reset(gcp_instance)
    finally:
        if nvml_ok:
            try:
                pynvml.nvmlShutdown()
                print("NVML shutdown successfully")
            except pynvml.NVMLError:
                print("Failed to shutdown NVML")


if __name__ == "__main__":
    main()
