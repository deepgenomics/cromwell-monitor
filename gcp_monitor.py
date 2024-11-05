import copy
import logging
import os
import re
from functools import reduce
from os import environ
from time import sleep, time
from types import MappingProxyType
from typing import Callable, List

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


def initialize_gcp_variables(
    nvml_ok: bool,
    services_pricelist: List[dict] = None,
    pricing_available: bool = False,
):
    gcp_variables = {}
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

    gcp_variables["MACHINE"] = get_machine_info(gcp_variables["compute"])
    # Get billing rates if pricing data is available
    if pricing_available:
        gcp_variables["COST_PER_SEC_NANODOLLARS"] = (
            get_machine_hour(gcp_variables["MACHINE"], services_pricelist)
            + get_disk_hour(gcp_variables["MACHINE"], services_pricelist)
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
    if pricing_available:
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
    # Getting VM GPU resources from the internal metadata here instead of from pynvml
    # because it is likely more robust
    gpu_data: List[dict] | None = instance.get("guestAccelerators", None)
    # Can't create machines with multiple GPU types, so just get the first element
    gpu_count = gpu_data[0].get("acceleratorCount", 0) if gpu_data else 0
    gpu_type = gpu_data[0].get("acceleratorType", None) if gpu_data else None
    # By default accelerator type is in the form of
    # projects/{project}/zones/{zone}/acceleratorTypes/{type}
    gpu_type = gpu_type.split("/")[-1] if gpu_type else None

    machine_info = {
        "project": project,
        "zone": zone,
        "region": zone[:-2],
        "name": name,
        "type": instance["machineType"].split("/")[-1],
        "preemptible": instance["scheduling"]["preemptible"],
        "disks": disks,
        "gpu_count": gpu_count,
        "gpu_type": gpu_type,
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


def get_price_key(key, preemptible):
    return "CP-COMPUTEENGINE-" + key + ("-PREEMPTIBLE" if preemptible else "")


def get_price_from_sku(sku: dict) -> tuple[int, float]:
    # Pricing api splits the price into whole dollars (units) and nano dollars (nanos).
    # We will return a combined nanodollars price. Conversion back to dollars is done
    # when the metric is submitted
    units_price = int(
        sku["pricingInfo"][0]["pricingExpression"]["tieredRates"][-1]["unitPrice"][
            "units"
        ]
    ) * (10**9)
    nanos_price = int(
        sku["pricingInfo"][0]["pricingExpression"]["tieredRates"][-1]["unitPrice"][
            "nanos"
        ]
    )
    return units_price + nanos_price


def get_machine_hour(machine, pricelist):
    machine_name_segments = machine["type"].split("-")
    machine_prefix = machine_name_segments[0].upper()
    # n1 custom machine api responses differ from other machine families
    machine_is_n1_custom = machine_prefix == "CUSTOM"
    # standard, custom, highmem, highcpu, etc.
    machine_is_custom = machine_name_segments[1] == "custom"
    machine_is_extended_memory = machine_name_segments[-1] == "ext"
    usage_type = "Preemptible" if machine["preemptible"] else "OnDemand"
    num_cpus: int | None = os.cpu_count()
    if num_cpus is None:
        raise ValueError("Could not determine number of CPUs")
    num_ram_gb = ps.virtual_memory().total / (1024**3)  # convert bytes to GiB
    num_gpus = machine.get("gpu_count", 0)
    gpu_type: str | None = machine.get("gpu_type", None)

    cpu_skus, memory_skus = get_cpu_and_mem_skus(
        machine,
        pricelist,
        machine_prefix,
        machine_is_n1_custom,
        machine_is_custom,
        machine_is_extended_memory,
        usage_type,
        num_gpus,
        gpu_type,
    )
    gpu_skus = get_gpu_skus(num_gpus, gpu_type, machine, pricelist, usage_type)

    # Check that only 1 sku is returned for each category
    if len(cpu_skus) != 1:
        raise ValueError(
            f"Expected 1 sku for CPU, got {len(cpu_skus)}, Skus: {cpu_skus}"
        )
    if len(memory_skus) != 1:
        raise ValueError(
            f"Expected 1 sku for RAM, got {len(memory_skus)}, Skus: {memory_skus}"
        )
    if num_gpus > 0 and len(gpu_skus) != 1:
        raise ValueError(
            f"Expected 1 sku for GPU, got {len(gpu_skus)}, Skus: {gpu_skus}"
        )

    cpu_nanodollars_price = get_price_from_sku(cpu_skus[0])
    cpu_price_per_hr = cpu_nanodollars_price * num_cpus
    ram_nanodollars_price = get_price_from_sku(memory_skus[0])
    ram_price_per_hr = ram_nanodollars_price * num_ram_gb

    if num_gpus > 0:
        gpu_nanodollars_price = get_price_from_sku(gpu_skus[0])
        gpu_price_per_hr = gpu_nanodollars_price * num_gpus
        return cpu_price_per_hr + ram_price_per_hr + gpu_price_per_hr
    else:
        return cpu_price_per_hr + ram_price_per_hr


def get_cpu_and_mem_skus(
    machine: dict,
    pricelist: List[dict],
    machine_prefix: str,
    machine_is_n1_custom: bool,
    machine_is_custom: bool,
    machine_is_extended_memory: bool,
    usage_type: str,
    num_gpus: int,
    gpu_type: str | None,
) -> List[dict]:
    cpu_filters: List[Callable[[dict], bool]] = []
    memory_filters: List[Callable[[dict], bool]] = []
    # Do a series of filters on the pricelist to get the correct sku
    cpu_filters.append(lambda sku: machine["region"] in sku["serviceRegions"])
    memory_filters.append(lambda sku: machine["region"] in sku["serviceRegions"])
    cpu_filters.append(lambda sku: usage_type in sku["category"]["usageType"])
    memory_filters.append(lambda sku: usage_type in sku["category"]["usageType"])
    # Need to concat usage type and custom because just "Custom" will return
    # skus for other machine families (eg. "E2 Custom" vs "Premptible Custom")
    if machine_is_n1_custom and usage_type == "Preemptible":
        cpu_filters.append(lambda sku: "Preemptible Custom" in sku["description"])
        memory_filters.append(lambda sku: "Preemptible Custom" in sku["description"])

    elif machine_is_n1_custom and usage_type == "OnDemand":
        # Cant use "Custom in description" because it will match other machine families
        # Need to ensure that Custom is the first word in the description to get
        # OnDemand N1 Custom machines
        cpu_filters.append(lambda sku: bool(re.search(r"^Custom ", sku["description"])))
        memory_filters.append(
            lambda sku: bool(re.search(r"^Custom ", sku["description"]))
        )
    else:
        cpu_filters.append(lambda sku: machine_prefix in sku["description"])
        memory_filters.append(lambda sku: machine_prefix in sku["description"])

    if machine_prefix == "N1":  # N1 Standard machines need different filters
        cpu_filters.append(lambda sku: "Core" in sku["description"])
        memory_filters.append(lambda sku: "Ram" in sku["description"])
    else:
        cpu_filters.append(lambda sku: "CPU" in sku["category"]["resourceGroup"])
        memory_filters.append(lambda sku: "RAM" in sku["category"]["resourceGroup"])

    if machine_is_custom or machine_is_n1_custom:
        # Filter out non-custom machines from core and memory skus
        cpu_filters.append(lambda sku: "Custom" in sku["description"])
        memory_filters.append(lambda sku: "Custom" in sku["description"])
    else:
        # Filter out custom machines from core and memory skus
        cpu_filters.append(lambda sku: "Custom" not in sku["description"])
        memory_filters.append(lambda sku: "Custom" not in sku["description"])

    if machine_is_extended_memory:
        memory_filters.append(lambda sku: "Extended" in sku["description"])
    else:
        memory_filters.append(lambda sku: "Extended" not in sku["description"])

    # Edge-case where h100 mega gpu machines have
    # different sku names for CPU and RAM
    if num_gpus > 0:
        if gpu_type not in _GPU_NAME_FROM_TYPE:
            raise ValueError(f"Unknown GPU type: {gpu_type}")
        if _GPU_NAME_FROM_TYPE.get(gpu_type) == "H100 80GB Plus":
            cpu_filters.append(lambda sku: "A3Plus" in sku["description"])
            memory_filters.append(lambda sku: "A3Plus" in sku["description"])
    cpu_skus = list(reduce(lambda result, f: filter(f, result), cpu_filters, pricelist))
    memory_skus = list(
        reduce(lambda result, f: filter(f, result), memory_filters, pricelist)
    )
    return cpu_skus, memory_skus


_GPU_NAME_FROM_TYPE = MappingProxyType(
    {
        "nvidia-tesla-t4": "T4",
        "nvidia-tesla-v100": "V100",
        "nvidia-tesla-p100": "P100",
        "nvidia-tesla-p4": "P4",
        "nvidia-l4": "L4",
        "nvidia-tesla-a100": "A100 40GB",
        "nvidia-a100-80gb": "A100 80GB",
        "nvidia-h100-80gb": "H100 80GB GPU",
        "nvidia-h100-mega-80gb": "H100 80GB Plus",
    }
)


def get_gpu_skus(num_gpus, gpu_type, machine, pricelist, usage_type):
    if num_gpus == 0:
        return []
    if gpu_type not in _GPU_NAME_FROM_TYPE:
        raise ValueError(f"Unknown GPU type: {gpu_type}")
    gpu_filters: List[Callable[[dict], bool]] = []
    gpu_filters.append(lambda sku: "GPU" in sku["category"]["resourceGroup"])
    gpu_filters.append(lambda sku: machine["region"] in sku["serviceRegions"])
    gpu_filters.append(
        lambda sku: _GPU_NAME_FROM_TYPE.get(gpu_type) in sku["description"]
    )
    gpu_filters.append(lambda sku: usage_type in sku["category"]["usageType"])
    return list(reduce(lambda result, f: filter(f, result), gpu_filters, pricelist))


_DISK_NAME_FROM_TYPE = MappingProxyType(
    {
        "pd-standard": "Storage PD Capacity",
        "pd-balanced": "Balanced PD Capacity",
        "pd-ssd": "SSD backed PD Capacity",
        "local-ssd": "SSD backed Local Storage",
        "pd-extreme": "Extreme PD Capacity",
        "hyperdisk-throughput": "Hyperdisk Throughput Capacity",
        "hyperdisk-ml": "Hyperdisk ML Capacity",
        "hyperdisk-extreme": "Hyperdisk Extreme Capacity",
        "hyperdisk-balanced": "Hyperdisk Balanced Capacity",
    }
)


def get_disk_hour(machine, pricelist):
    # This function will ignore Hyperdisk Throughput Storage Pools cost since it is
    # billed monthly based on resources allocated to the pool, not the VM
    total = 0
    for disk in machine.get("disks"):
        if disk["type"] not in _DISK_NAME_FROM_TYPE:
            raise ValueError(f"Unknown disk type: {disk['type']}")
        search_term = _DISK_NAME_FROM_TYPE[disk["type"]]
        disk_skus = get_disk_skus(machine, pricelist, search_term)
        if len(disk_skus) != 1:
            raise ValueError(
                f"Expected 1 sku for disk, got {len(disk_skus)}, Skus: {disk_skus}"
            )

        # Disk prices are per month, need to convert to hourly, 730 hours in a month
        disk_price_gb_nanodollars = get_price_from_sku(disk_skus[0]) / 730
        price = disk_price_gb_nanodollars * disk["sizeGb"]
        total += price
    # Casting to int is a reasonable compromise to avoid weird floating point errors
    return int(total)


def get_disk_skus(machine: dict, pricelist: List[dict], search_term: str) -> List[dict]:
    disk_filters: List[Callable[[dict], bool]] = []
    # Filter skus to be in the same region as the VM
    disk_filters.append(lambda sku: machine["region"] in sku["serviceRegions"])
    # All disks only have OnDemand billing
    disk_filters.append(lambda sku: "OnDemand" in sku["category"]["usageType"])
    disk_filters.append(lambda sku: "Storage" in sku["category"]["resourceFamily"])
    # Assume all disks are not regional, HA, confidential, etc.
    # Use the following set of negative filters to remove those
    disk_filters.append(lambda sku: "Pools" not in sku["description"])
    disk_filters.append(lambda sku: "Confidential" not in sku["description"])
    disk_filters.append(lambda sku: "Regional" not in sku["description"])
    disk_filters.append(lambda sku: "High Availability" not in sku["description"])
    # Filter by disk type
    disk_filters.append(lambda sku: search_term in sku["description"])

    return list(reduce(lambda result, f: filter(f, result), disk_filters, pricelist))


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


def report(
    gcp_variables,
    metrics_client,
    nvml_ok: bool = False,
    pricing_available: bool = False,
):
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
    cost_metric = (
        [
            get_time_series(
                gcp_variables,
                gcp_variables["COST_ESTIMATE_METRIC"],
                {
                    "double_value": (time() - ps.boot_time())
                    * (gcp_variables["COST_PER_SEC_NANODOLLARS"] / 10**9)
                },
            )
        ]
        if pricing_available
        else []
    )
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
            *gpu_metrics,
            *cost_metric,
        ],
    )
    logging.info("Successfully wrote time series to Cloud Monitoring API.")
    return gcp_variables
