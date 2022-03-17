#!/usr/bin/env python3

import json
import logging
from functools import reduce
from os import environ
from signal import SIGTERM, signal
from time import sleep, time

import psutil as ps
import requests
from google.api import label_pb2 as ga_label
from google.api import metric_pb2 as ga_metric
from google.cloud.monitoring_v3 import (
    MetricServiceClient,
    Point,
    TimeInterval,
    TimeSeries,
)
from googleapiclient.discovery import build as google_api

compute = google_api("compute", "v1")

logging.getLogger().setLevel(logging.INFO)


def get_machine_info():
    metadata = requests.get(
        "http://metadata.google.internal/computeMetadata/v1/instance/?recursive=true",
        headers={"Metadata-Flavor": "Google"},
    ).json()

    name = metadata["name"]
    _, project, _, zone = metadata["zone"].split("/")
    instance = (
        compute.instances().get(project=project, zone=zone, instance=name).execute()
    )

    disks = [get_disk(project, zone, disk) for disk in instance["disks"]]

    machine_info = {
        "project": project,
        "zone": zone,
        "region": zone[:-2],
        "name": name,
        "type": instance["machineType"].split("/")[-1],
        "preemptible": instance["scheduling"]["preemptible"],
        "disks": disks,
    }

    return machine_info


def get_disk(project, zone, disk):
    if disk["type"] == "PERSISTENT":
        name = disk["source"].split("/")[-1]
        resource = compute.disks().get(project=project, zone=zone, disk=name).execute()
        return {
            "type": resource["type"].split("/")[-1],
            "sizeGb": int(resource["sizeGb"]),
        }
    else:
        return {
            "type": "local-ssd",
            "sizeGb": 375,
        }


PRICELIST_JSON = (
    "pricelist.json"  # json that contains GCP pricing. used for cost metric.
)


def get_pricelist():
    with open(PRICELIST_JSON, "r") as f:
        return json.load(f)["gcp_price_list"]


def get_price_key(key, preemptible):
    return "CP-COMPUTEENGINE-" + key + ("-PREEMPTIBLE" if preemptible else "")


def get_machine_hour(machine, pricelist):
    if machine["type"].startswith("custom"):
        _, core, memory = machine["type"].split("-")
        core_key = get_price_key("CUSTOM-VM-CORE", machine["preemptible"])
        memory_key = get_price_key("CUSTOM-VM-RAM", machine["preemptible"])
        return (
            pricelist[core_key][machine["region"]] * int(core)
            + pricelist[memory_key][machine["region"]] * int(memory) / 2**10
        )
    else:
        price_key = get_price_key(
            "VMIMAGE-" + machine["type"].upper(), machine["preemptible"]
        )
        return pricelist[price_key][machine["region"]]


def get_disk_hour(machine, pricelist):
    total = 0
    for disk in machine["disks"]:
        price_key = "CP-COMPUTEENGINE-"
        if disk["type"] == "pd-standard":
            price_key += "STORAGE-PD-CAPACITY"
        elif disk["type"] == "pd-ssd":
            price_key += "STORAGE-PD-SSD"
        elif disk["type"] == "local-ssd":
            price_key += "LOCAL-SSD"
            if machine["preemptible"]:
                price_key += "-PREEMPTIBLE"
        price = pricelist[price_key][machine["region"]] * disk["sizeGb"]
        if disk["type"].startswith("pd"):
            price /= 730  # hours per month
        total += price
    return total


def reset():
    global memory_used, disk_used, disk_reads, disk_writes, last_time

    # Explicitly reset the CPU counter
    # because the first call of this method always reports 0
    ps.cpu_percent()

    memory_used = 0

    disk_used = 0
    disk_reads = disk_io("read_count")
    disk_writes = disk_io("write_count")

    last_time = time()


def measure():
    global memory_used, disk_used

    memory_used = max(memory_used, MEMORY_SIZE - mem_usage("available"))
    disk_used = max(disk_used, disk_usage("used"))
    logging.info("VM memory used: %s", memory_used)

    sleep(MEASUREMENT_TIME_SEC)


def mem_usage(param):
    return getattr(ps.virtual_memory(), param)


def disk_usage(param):
    return reduce(
        lambda usage, mount: usage + getattr(ps.disk_usage(mount), param),
        DISK_MOUNTS,
        0,
    )


def disk_io(param):
    return getattr(ps.disk_io_counters(), param)


def format_gb(value_bytes):
    return "%.1f" % round(value_bytes / 2**30, 1)


def get_metric(key, value_type, unit, description):
    return client.create_metric_descriptor(
        name=PROJECT_NAME,
        metric_descriptor=ga_metric.MetricDescriptor(
            type="/".join(["custom.googleapis.com", METRIC_ROOT, key]),
            description=description,
            metric_kind="GAUGE",
            value_type=value_type,
            unit=unit,
            labels=LABEL_DESCRIPTORS,
        ),
    )


def create_time_series(series):
    client.create_time_series(request={"name": PROJECT_NAME, "time_series": series})


def get_time_series(metric_descriptor, value):
    global last_time
    series = TimeSeries()

    series.metric.type = metric_descriptor.type
    labels = series.metric.labels
    labels["workflow_id"] = WORKFLOW_ID
    labels["task_call_name"] = TASK_CALL_NAME
    labels["task_call_index"] = TASK_CALL_INDEX
    labels["task_call_attempt"] = TASK_CALL_ATTEMPT
    labels["cpu_count"] = CPU_COUNT_LABEL
    labels["mem_size"] = MEMORY_SIZE_LABEL
    labels["disk_size"] = DISK_SIZE_LABEL
    labels["preemptible"] = PREEMPTIBLE_LABEL

    series.resource.type = "gce_instance"
    series.resource.labels["zone"] = MACHINE["zone"]
    series.resource.labels["instance_id"] = MACHINE["name"]

    end_time = int(max(time(), last_time + REPORT_TIME_SEC_MIN))
    interval = TimeInterval({"end_time": {"seconds": end_time}})
    point = Point({"interval": interval, "value": value})
    series.points = [point]

    return series


def report():
    global last_time
    time_delta = time() - last_time
    create_time_series(
        [
            get_time_series(CPU_UTILIZATION_METRIC, {"double_value": ps.cpu_percent()}),
            get_time_series(
                MEMORY_UTILIZATION_METRIC,
                {"double_value": memory_used / MEMORY_SIZE * 100},
            ),
            get_time_series(
                DISK_UTILIZATION_METRIC, {"double_value": disk_used / DISK_SIZE * 100}
            ),
            get_time_series(
                DISK_READS_METRIC,
                {"double_value": (disk_io("read_count") - disk_reads) / time_delta},
            ),
            get_time_series(
                DISK_WRITES_METRIC,
                {"double_value": (disk_io("write_count") - disk_writes) / time_delta},
            ),
            get_time_series(
                COST_ESTIMATE_METRIC,
                {"double_value": (time() - ps.boot_time()) * COST_PER_SEC},
            ),
        ]
    )
    logging.info("Successfully wrote time series to Cloud Monitoring API.")


# Define constants

# Cromwell variables passed to the container
# through environmental variables
WORKFLOW_ID = environ["WORKFLOW_ID"]
TASK_CALL_NAME = environ["TASK_CALL_NAME"]
TASK_CALL_INDEX = environ[
    "TASK_CALL_INDEX"
]  # this variable is shard number from scatter pattern
TASK_CALL_ATTEMPT = environ[
    "TASK_CALL_ATTEMPT"
]  # this variable is the number of retry ie. 0, 1, 2, etc
DISK_MOUNTS = environ["DISK_MOUNTS"].split()


# Get billing rates
MACHINE = get_machine_info()
PRICELIST = get_pricelist()
COST_PER_SEC = (
    get_machine_hour(MACHINE, PRICELIST) + get_disk_hour(MACHINE, PRICELIST)
) / 3600


client = MetricServiceClient()
PROJECT_NAME = client.common_project_path(MACHINE["project"])
logging.info("project name: %s", PROJECT_NAME)

METRIC_ROOT = "wdl_task"

MEASUREMENT_TIME_SEC = 60
REPORT_TIME_SEC_MIN = 60
REPORT_TIME_SEC = REPORT_TIME_SEC_MIN

LABEL_DESCRIPTORS = [
    ga_label.LabelDescriptor(
        key="workflow_id",
        description="Cromwell workflow ID",
    ),
    ga_label.LabelDescriptor(
        key="task_call_name",
        description="Cromwell task call name",
    ),
    ga_label.LabelDescriptor(
        key="task_call_index",
        description="Cromwell task call index",
    ),
    ga_label.LabelDescriptor(
        key="task_call_attempt",
        description="Cromwell task call attempt",
    ),
    ga_label.LabelDescriptor(
        key="cpu_count",
        description="Number of virtual cores",
    ),
    ga_label.LabelDescriptor(
        key="mem_size",
        description="Total memory size, GB",
    ),
    ga_label.LabelDescriptor(
        key="disk_size",
        description="Total disk size, GB",
    ),
    ga_label.LabelDescriptor(
        key="preemptible",
        description="Preemptible flag",
    ),
]

CPU_COUNT = ps.cpu_count()
CPU_COUNT_LABEL = str(CPU_COUNT)

MEMORY_SIZE = mem_usage("total")
MEMORY_SIZE_LABEL = format_gb(MEMORY_SIZE)

DISK_SIZE = disk_usage("total")
DISK_SIZE_LABEL = format_gb(DISK_SIZE)

PREEMPTIBLE_LABEL = str(MACHINE["preemptible"]).lower()

CPU_UTILIZATION_METRIC = get_metric(
    "cpu_utilization",
    "DOUBLE",
    "%",
    "% of CPU utilized in a Cromwell task call",
)

MEMORY_UTILIZATION_METRIC = get_metric(
    "mem_utilization",
    "DOUBLE",
    "%",
    "% of memory utilized in a Cromwell task call",
)

DISK_UTILIZATION_METRIC = get_metric(
    "disk_utilization",
    "DOUBLE",
    "%",
    "% of disk utilized in a Cromwell task call",
)

DISK_READS_METRIC = get_metric(
    "disk_reads",
    "DOUBLE",
    "{reads}/s",
    "Disk read IOPS in a Cromwell task call",
)

DISK_WRITES_METRIC = get_metric(
    "disk_writes",
    "DOUBLE",
    "{writes}/s",
    "Disk write IOPS in a Cromwell task call",
)

COST_ESTIMATE_METRIC = get_metric(
    "runtime_cost_estimate",
    "DOUBLE",
    "USD",
    "Cumulative runtime cost estimate for a Cromwell task call",
)

# Detect container termination


def signal_handler(signum, frame):
    global running
    running = False


running = True
signal(SIGTERM, signal_handler)

# Main loop
#
# It continuously measures runtime metrics every MEASUREMENT_TIME_SEC,
# and reports them to Stackdriver Monitoring API every REPORT_TIME_SEC.
#
# However, if it detects a container termination signal,
# it *should* report the final metric
# right after the current measurement, and then exit normally.

reset()
while running:
    measure()
    if not running or (time() - last_time) >= REPORT_TIME_SEC:
        report()
        reset()
exit(0)
