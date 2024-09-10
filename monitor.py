#!/usr/bin/env python3

import logging
import os
from signal import SIGTERM, signal
from time import time
import requests
import gcp_monitor

logging.getLogger().setLevel(logging.INFO)

# flag to keep track of container termination
container_running = True


# Detect container termination
def signal_handler(signum, frame):
    global container_running
    container_running = False


def get_pricelist_dict() -> dict:
    """
    Query the cloudbilling api for current compute engine SKUs and prices,
    then collate the paginated json responses into a single json file

    Returns a dict containing the SKUs and prices of all compute engine services,
    or in the event of failure throws the appropriate
    requests.exceptions.RequestException
    """
    query_params = {
        "pageSize": 5000,
    }
    res = requests.get(
        "https://cloudbilling.googleapis.com/v1/services/6F81-5844-456A/skus",
        params=query_params)
    res.raise_for_status()
    services_json = res.json()
    next_page_token = services_json.get("nextPageToken", "")
    services_dict = services_json.get("skus", [])
    while next_page_token != "":
        query_params["pageToken"] = next_page_token
        res = requests.get(
            "https://cloudbilling.googleapis.com/v1/services/6F81-5844-456A/skus",
            params=query_params)
        services_json = res.json()
        next_page_token = services_json.get("nextPageToken", "")
        services_dict += services_json.get("skus", [])
    return services_dict


def main():
    """
    This function continuously measures runtime metrics every MEASUREMENT_TIME_SEC,
    and reports them to Stackdriver Monitoring API every REPORT_TIME_SEC.

    However, if it detects a container termination signal,
    it *should* report the final metric
    right after the current measurement, and then exit normally.
    """
    try:
        services_pricelist: dict = get_pricelist_dict()
        gcp_instance, metrics_client = gcp_monitor.initialize_gcp_variables(services_pricelist)
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to retrieve pricelist: {e}")
        logging.warning("Will attempt to continue monitoring without pricing data...")
        gcp_instance, metrics_client = gcp_monitor.initialize_gcp_variables()

    signal(SIGTERM, signal_handler)

    gcp_instance = gcp_monitor.reset(gcp_instance)
    while container_running:
        gcp_instance = gcp_monitor.measure(gcp_instance)
        if (
            not container_running
            or (time() - gcp_instance["last_time"]) >= gcp_instance["REPORT_TIME_SEC"]
        ):
            gcp_instance = gcp_monitor.report(gcp_instance, metrics_client)
            gcp_instance = gcp_monitor.reset(gcp_instance)


if __name__ == "__main__":
    main()
