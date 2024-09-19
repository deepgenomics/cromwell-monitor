import copy
import json
import logging
from functools import reduce
from os import environ
from time import sleep, time

import psutil as ps
import pynvml
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


def initialize_gcp_variables(nvml_ok: bool):
    gcp_variables = {}

    # json that contains GCP pricing. used for cost metric.
    gcp_variables["PRICELIST_JSON"] = "pricelist.json"

    # initialize Google API client
    gcp_variables["compute"] = google_api("compute", "v1")

    # Define constants
    # Cromwell variables passed to the container
    # through environmental variables
    gcp_variables["WORKFLOW_ID"] = environ["WORKFLOW_ID"]
    gcp_variables["TASK_CALL_NAME"] = environ["TASK_CALL_NAME"]
    # TASK_CALL_INDEX is shard number from scatter pattern: ie. 0, 1, 2, etc
    gcp_variables["TASK_CALL_INDEX"] = environ["TASK_CALL_INDEX"]
    # TASK_CALL_ATTEMPT is the number of retry ie. 0, 1, 2, etc
    gcp_variables["TASK_CALL_ATTEMPT"] = environ["TASK_CALL_ATTEMPT"]
    gcp_variables["DISK_MOUNTS"] = environ["DISK_MOUNTS"].split()

    # METRIC_ROOT is the name we assigned to a custom gcloud monitoring metric
    gcp_variables["METRIC_ROOT"] = "wdl_task"

    gcp_variables["MEASUREMENT_TIME_SEC"] = 60
    gcp_variables["REPORT_TIME_SEC_MIN"] = 60
    gcp_variables["REPORT_TIME_SEC"] = gcp_variables["REPORT_TIME_SEC_MIN"]

    # Get billing rates
    gcp_variables["MACHINE"] = get_machine_info(gcp_variables["compute"])
    gcp_variables["PRICELIST"] = get_pricelist(gcp_variables["PRICELIST_JSON"])
    gcp_variables["COST_PER_SEC"] = (
        get_machine_hour(gcp_variables["MACHINE"], gcp_variables["PRICELIST"])
        + get_disk_hour(gcp_variables["MACHINE"], gcp_variables["PRICELIST"])
    ) / 3600

    gcp_variables["OWNER"] = (
        gcp_variables["MACHINE"]["owner"]
        if "owner" in gcp_variables["MACHINE"].keys()
        else ""
    )
    gcp_variables["ENTRANCE_WDL"] = (
        gcp_variables["MACHINE"]["entrance_wdl"]
        if "entrance_wdl" in gcp_variables["MACHINE"].keys()
        else ""
    )

    metrics_client = MetricServiceClient()
    gcp_variables["PROJECT_NAME"] = metrics_client.common_project_path(
        gcp_variables["MACHINE"]["project"]
    )
    logging.info("project name: %s", gcp_variables["PROJECT_NAME"])

    gcp_variables["LABEL_DESCRIPTORS"] = [
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
        ga_label.LabelDescriptor(
            key="owner",
            description="Owner Label defined by user in VectorHive2",
        ),
        ga_label.LabelDescriptor(
            key="entrance_wdl",
            description="Entrance WDL Label defined by VectorHive2",
        ),
    ]

    # psutil metrics
    gcp_variables["memory_used"] = 0
    gcp_variables["disk_used"] = 0
    gcp_variables["disk_reads"] = 0
    gcp_variables["disk_writes"] = 0
    gcp_variables["last_time"] = 0

    gcp_variables["CPU_COUNT"] = ps.cpu_count()
    gcp_variables["CPU_COUNT_LABEL"] = str(gcp_variables["CPU_COUNT"])

    gcp_variables["MEMORY_SIZE"] = mem_usage("total")
    gcp_variables["MEMORY_SIZE_LABEL"] = format_gb(gcp_variables["MEMORY_SIZE"])

    gcp_variables["DISK_SIZE"] = disk_usage(gcp_variables, "total")
    gcp_variables["DISK_SIZE_LABEL"] = format_gb(gcp_variables["DISK_SIZE"])

    gcp_variables["PREEMPTIBLE_LABEL"] = str(
        gcp_variables["MACHINE"]["preemptible"]
    ).lower()

    gcp_variables["CPU_UTILIZATION_METRIC"] = get_metric(
        gcp_variables,
        metrics_client,
        "cpu_utilization",
        "DOUBLE",
        "%",
        "% of CPU utilized in a Cromwell task call",
    )

    gcp_variables["MEMORY_UTILIZATION_METRIC"] = get_metric(
        gcp_variables,
        metrics_client,
        "mem_utilization",
        "DOUBLE",
        "%",
        "% of memory utilized in a Cromwell task call",
    )

    gcp_variables["DISK_UTILIZATION_METRIC"] = get_metric(
        gcp_variables,
        metrics_client,
        "disk_utilization",
        "DOUBLE",
        "%",
        "% of disk utilized in a Cromwell task call",
    )

    gcp_variables["DISK_READS_METRIC"] = get_metric(
        gcp_variables,
        metrics_client,
        "disk_reads",
        "DOUBLE",
        "{reads}/s",
        "Disk read IOPS in a Cromwell task call",
    )

    gcp_variables["DISK_WRITES_METRIC"] = get_metric(
        gcp_variables,
        metrics_client,
        "disk_writes",
        "DOUBLE",
        "{writes}/s",
        "Disk write IOPS in a Cromwell task call",
    )

    if nvml_ok:
        num_gpus = pynvml.nvmlDeviceGetCount()
        for i in range(num_gpus):
            gcp_variables[f"GPU{i}_UTILIZATION_METRIC"] = get_metric(
                gcp_variables,
                metrics_client,
                f"gpu{i}_utilization",
                "INT64",
                "%",
                f"GPU{i}: Percent of time over the past sample period during which one or more kernels was executing",
            )

            gcp_variables[f"GPU{i}_MEM_UTILIZATION_METRIC"] = get_metric(
                gcp_variables,
                metrics_client,
                f"gpu{i}_mem_time_utilization",
                "INT64",
                "%",
                f"GPU{i}: Percent of time over the past sample period during which global (device) memory was being read or written",
            )

            gcp_variables[f"GPU{i}_MEM_ALLOCATED_METRIC"] = get_metric(
                gcp_variables,
                metrics_client,
                f"gpu{i}_mem_allocated",
                "INT64",
                "%",
                f"GPU{i}: Percent of memory utilized (used / available)",
            )

    gcp_variables["COST_ESTIMATE_METRIC"] = get_metric(
        gcp_variables,
        metrics_client,
        "runtime_cost_estimate",
        "DOUBLE",
        "USD",
        "Cumulative runtime cost estimate for a Cromwell task call",
    )

    return gcp_variables, metrics_client


def get_machine_info(compute):
    metadata = requests.get(
        "http://metadata.google.internal/computeMetadata/v1/instance/?recursive=true",
        headers={"Metadata-Flavor": "Google"},
    ).json()

    name = metadata["name"]
    _, project, _, zone = metadata["zone"].split("/")
    instance = (
        compute.instances().get(project=project, zone=zone, instance=name).execute()
    )

    disks = [get_disk(compute, project, zone, disk) for disk in instance["disks"]]

    machine_info = {
        "project": project,
        "zone": zone,
        "region": zone[:-2],
        "name": name,
        "type": instance["machineType"].split("/")[-1],
        "preemptible": instance["scheduling"]["preemptible"],
        "disks": disks,
    }

    if "owner" in instance["labels"].keys():
        machine_info.update({"owner": instance["labels"]["owner"]})

    # GCP cloud monitoring API does not accept hyphen
    if "entrance-wdl" in instance["labels"].keys():
        machine_info.update({"entrance_wdl": instance["labels"]["entrance-wdl"]})

    return machine_info


def get_disk(compute, project, zone, disk):
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


def get_pricelist(pricelist_json):
    with open(pricelist_json, "r") as f:
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
    for disk in machine.get("disks"):
        price_key = "CP-COMPUTEENGINE-"
        if disk["type"] == "pd-standard" or disk["type"] == "pd-balanced":
            price_key += "STORAGE-PD-CAPACITY"
        elif disk["type"] == "pd-ssd":
            price_key += "STORAGE-PD-SSD"
        elif disk["type"] == "local-ssd":
            price_key += "LOCAL-SSD"
            if machine["preemptible"]:
                price_key += "-PREEMPTIBLE"
        assert price_key in pricelist.keys(), f"Unknown disk type: {disk['type']}"
        assert (
            machine["region"] in pricelist[price_key].keys()
        ), f"Unknown region: {machine['region']} for disk type: {disk['type']}"
        price = pricelist[price_key][machine["region"]] * disk["sizeGb"]
        if disk["type"].startswith("pd"):
            price /= 730  # hours per month
        total += price
    return total


def reset(gcp_variables):
    # Explicitly reset the CPU counter,
    # because the first call of this method always reports 0
    ps.cpu_percent()

    reset_variables = copy.deepcopy(gcp_variables)

    reset_variables["memory_used"] = 0
    reset_variables["disk_used"] = 0
    reset_variables["disk_reads"] = disk_io("read_count")
    reset_variables["disk_writes"] = disk_io("write_count")

    reset_variables["last_time"] = time()

    return reset_variables


def measure(gcp_variables):

    measure_variables = copy.deepcopy(gcp_variables)

    measure_variables["memory_used"] = max(
        measure_variables["memory_used"],
        measure_variables["MEMORY_SIZE"] - mem_usage("available"),
    )
    measure_variables["disk_used"] = max(
        measure_variables["disk_used"], disk_usage(measure_variables, "used")
    )
    logging.info("VM memory used: %s", measure_variables["memory_used"])

    sleep(measure_variables["MEASUREMENT_TIME_SEC"])

    return measure_variables


def mem_usage(param):
    return getattr(ps.virtual_memory(), param)


def disk_usage(gcp_variables, param):
    return reduce(
        lambda usage, mount: usage + getattr(ps.disk_usage(mount), param),
        gcp_variables["DISK_MOUNTS"],
        0,
    )


def disk_io(param):
    return getattr(ps.disk_io_counters(), param)


def format_gb(value_bytes):
    return "%.1f" % round(value_bytes / 2**30, 1)


def get_metric(gcp_variables, metrics_client, key, value_type, unit, description):
    return metrics_client.create_metric_descriptor(
        name=gcp_variables["PROJECT_NAME"],
        metric_descriptor=ga_metric.MetricDescriptor(
            type="/".join(["custom.googleapis.com", gcp_variables["METRIC_ROOT"], key]),
            description=description,
            metric_kind="GAUGE",
            value_type=value_type,
            unit=unit,
            labels=gcp_variables["LABEL_DESCRIPTORS"],
        ),
    )


def create_time_series(gcp_variables, metrics_client, series):
    metrics_client.create_time_series(
        request={"name": gcp_variables["PROJECT_NAME"], "time_series": series}
    )


def get_time_series(gcp_variables, metric_descriptor, value):
    series = TimeSeries()

    series.metric.type = metric_descriptor.type
    labels = series.metric.labels
    labels["workflow_id"] = gcp_variables["WORKFLOW_ID"]
    labels["task_call_name"] = gcp_variables["TASK_CALL_NAME"]
    labels["task_call_index"] = gcp_variables["TASK_CALL_INDEX"]
    labels["task_call_attempt"] = gcp_variables["TASK_CALL_ATTEMPT"]
    labels["cpu_count"] = gcp_variables["CPU_COUNT_LABEL"]
    labels["mem_size"] = gcp_variables["MEMORY_SIZE_LABEL"]
    labels["disk_size"] = gcp_variables["DISK_SIZE_LABEL"]
    labels["preemptible"] = gcp_variables["PREEMPTIBLE_LABEL"]
    if gcp_variables["OWNER"]:
        labels["owner"] = gcp_variables["OWNER"]
    if gcp_variables["ENTRANCE_WDL"]:
        labels["entrance_wdl"] = gcp_variables["ENTRANCE_WDL"]

    series.resource.type = "gce_instance"
    series.resource.labels["zone"] = gcp_variables["MACHINE"]["zone"]
    series.resource.labels["instance_id"] = gcp_variables["MACHINE"]["name"]

    end_time = int(
        max(time(), gcp_variables["last_time"] + gcp_variables["REPORT_TIME_SEC_MIN"])
    )
    interval = TimeInterval({"end_time": {"seconds": end_time}})
    point = Point({"interval": interval, "value": value})
    series.points = [point]

    return series


def report(gcp_variables, metrics_client, nvml_ok: bool = False):

    num_gpus = pynvml.nvmlDeviceGetCount() if nvml_ok else 0
    time_delta = time() - gcp_variables["last_time"]
    gpus = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(num_gpus)]
    # https://pypi.org/project/nvidia-ml-py is a thin wrapper over the C NVML library
    # See https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html
    # for type info and field names
    gpu_utilization_rates = [
        pynvml.nvmlDeviceGetUtilizationRates(gpus[i]) for i in range(num_gpus)
    ]
    gpu_mem_info = [pynvml.nvmlDeviceGetMemoryInfo(gpus[i]) for i in range(num_gpus)]
    gpu_metrics = [
        *[
            get_time_series(
                gcp_variables,
                gcp_variables[f"GPU{i}_UTILIZATION_METRIC"],
                {"int64_value": (gpu_utilization_rates[i].gpu)},
            )
            for i in range(num_gpus)
        ],
        *[
            get_time_series(
                gcp_variables,
                gcp_variables[f"GPU{i}_MEM_UTILIZATION_METRIC"],
                {"int64_value": (gpu_utilization_rates[i].memory)},
            )
            for i in range(num_gpus)
        ],
        *[
            get_time_series(
                gcp_variables,
                gcp_variables[f"GPU{i}_MEM_ALLOCATED_METRIC"],
                {"int64_value": 100 * (gpu_mem_info[i].used) / (gpu_mem_info[i].total)},
            )
            for i in range(num_gpus)
        ],
    ]
    create_time_series(
        gcp_variables,
        metrics_client,
        [
            get_time_series(
                gcp_variables,
                gcp_variables["CPU_UTILIZATION_METRIC"],
                {"double_value": ps.cpu_percent()},
            ),
            get_time_series(
                gcp_variables,
                gcp_variables["MEMORY_UTILIZATION_METRIC"],
                {
                    "double_value": gcp_variables["memory_used"]
                    / gcp_variables["MEMORY_SIZE"]
                    * 100
                },
            ),
            get_time_series(
                gcp_variables,
                gcp_variables["DISK_UTILIZATION_METRIC"],
                {
                    "double_value": gcp_variables["disk_used"]
                    / gcp_variables["DISK_SIZE"]
                    * 100
                },
            ),
            get_time_series(
                gcp_variables,
                gcp_variables["DISK_READS_METRIC"],
                {
                    "double_value": (
                        disk_io("read_count") - gcp_variables["disk_reads"]
                    )
                    / time_delta
                },
            ),
            get_time_series(
                gcp_variables,
                gcp_variables["DISK_WRITES_METRIC"],
                {
                    "double_value": (
                        disk_io("write_count") - gcp_variables["disk_writes"]
                    )
                    / time_delta
                },
            ),
            get_time_series(
                gcp_variables,
                gcp_variables["COST_ESTIMATE_METRIC"],
                {
                    "double_value": (time() - ps.boot_time())
                    * gcp_variables["COST_PER_SEC"]
                },
            ),
            *gpu_metrics,
        ],
    )
    logging.info("Successfully wrote time series to Cloud Monitoring API.")

    return gcp_variables
