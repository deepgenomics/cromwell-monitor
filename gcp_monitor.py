import copy
import logging
from functools import reduce
from os import environ
import os
from time import sleep, time
from typing import List

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


def initialize_gcp_variables(services_pricelist: dict = None):
    gcp_variables = {}

    gcp_variables["PRICELIST"] = services_pricelist
    # Var to track if should attempt to add pricing metrics
    gcp_variables["PRICING_AVAILABLE"] = True if services_pricelist else False

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
    if gcp_variables["PRICING_AVAILABLE"]:
        try:
            gcp_variables["COST_PER_SEC"] = (
                get_machine_hour(gcp_variables["MACHINE"], gcp_variables["PRICELIST"])
                + get_disk_hour(gcp_variables["MACHINE"], gcp_variables["PRICELIST"])
            ) / 3600
        except ValueError as e:
            logging.error(f"Failed to get pricing data: {e}")
            logging.warning(
                "Will attempt to continue monitoring without pricing data..."
            )
            gcp_variables["PRICING_AVAILABLE"] = False

    # Get VectorHive2 related labels
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
    if gcp_variables["PRICING_AVAILABLE"]:
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


def get_price_key(key, preemptible):
    return "CP-COMPUTEENGINE-" + key + ("-PREEMPTIBLE" if preemptible else "")


def get_machine_hour(machine, pricelist):
    machine_prefix = machine["type"].split("-")[0].upper()
    # n1 custom machine api responses differ from other machine families
    machine_is_n1_custom = machine_prefix == "custom"
    # standard, custom, highmem, highcpu, etc.
    machine_is_custom = True if machine["type"].split("-")[1] == "custom" else False
    machine_is_extended_memory: bool = machine["type"].split("-")[-1] == "ext"
    usage_type = "Preemptible" if machine["preemptible"] else "OnDemand"
    num_cpus: int | None = os.cpu_count()
    if num_cpus is None:
        raise ValueError("Could not determine number of CPUs")
    num_ram_gb = ps.virtual_memory().total / (1024**3)  # convert bytes to GiB
    num_gpus = machine["guestAccelerators"].get("acceleratorCount", 0)
    gpu_type: str | None = machine["guestAccelerators"].get("acceleratorType", None)
    # By default accelerator type is in the form of
    # projects/{project}/zones/{zone}/acceleratorTypes/{type}
    gpu_type = gpu_type.split("/")[-1] if gpu_type else None

    # Initial sku filtering results will go in these vars
    core_skus: List[dict] | None = None
    memory_skus: List[dict] | None = None
    gpu_skus: List[dict] | None = None
    # Do a series of filters on the pricelist to get the correct sku
    if machine_is_n1_custom:  # N1 and N1 Custom machines need different filters
        # Need to concat usage type and custom because just "Custom" will return
        # skus for other machine families (eg. "E2 Custom" vs "Premptible Custom")
        machine_type_skus = [
            sku
            for sku in pricelist
            if str(usage_type) + " Custom" in sku["description"]
        ]
        regional_skus = [
            sku
            for sku in machine_type_skus
            if machine["region"] in sku["serviceRegions"]
        ]
        usage_type_skus = [
            sku for sku in regional_skus if usage_type in sku["category"]["usageType"]
        ]
        core_skus = [
            sku for sku in usage_type_skus if "CPU" in sku["category"]["resourceGroup"]
        ]
        memory_skus = [
            sku for sku in usage_type_skus if "RAM" in sku["category"]["resourceGroup"]
        ]
    elif machine_prefix == "N1":  # N1 and N1 Custom machines need different filters
        machine_type_skus = [
            sku
            for sku in pricelist
            if machine_prefix in sku["category"]["resourceGroup"]
        ]
        regional_skus = [
            sku
            for sku in machine_type_skus
            if machine["region"] in sku["serviceRegions"]
        ]
        usage_type_skus = [
            sku for sku in regional_skus if usage_type in sku["category"]["usageType"]
        ]
        core_skus = [sku for sku in usage_type_skus if "Core" in sku["description"]]
        memory_skus = [sku for sku in usage_type_skus if "Ram" in sku["description"]]
    else:
        machine_type_skus = [
            sku for sku in pricelist if machine_prefix in sku["description"]
        ]
        regional_skus = [
            sku
            for sku in machine_type_skus
            if machine["region"] in sku["serviceRegions"]
        ]
        usage_type_skus = [
            sku for sku in regional_skus if usage_type in sku["category"]["usageType"]
        ]
        core_skus = [
            sku for sku in usage_type_skus if "CPU" in sku["category"]["resourceGroup"]
        ]
        memory_skus = [
            sku for sku in usage_type_skus if "RAM" in sku["category"]["resourceGroup"]
        ]

    if machine_is_custom or machine_is_n1_custom:
        # Filter out non-custom machines from core and memory skus
        core_skus = [sku for sku in core_skus if "Custom" in sku["description"]]
        memory_skus = [sku for sku in memory_skus if "Custom" in sku["description"]]
    else:
        # Filter out custom machines from core and memory skus
        core_skus = [sku for sku in core_skus if "Custom" not in sku["description"]]
        memory_skus = [sku for sku in memory_skus if "Custom" not in sku["description"]]

    if machine_is_extended_memory:
        memory_skus = [sku for sku in memory_skus if "Extended" in sku["description"]]
    else:
        memory_skus = [
            sku for sku in memory_skus if "Extended" not in sku["description"]
        ]

    if num_gpus > 0:
        # Convert GPU type to name used in pricing API
        match gpu_type:
            case "nvidia-tesla-t4":
                gpu_type = "T4"
            case "nvidia-tesla-v100":
                gpu_type = "V100"
            case "nvidia-tesla-p100":
                gpu_type = "P100"
            case "nvidia-tesla-p4":
                gpu_type = "P4"
            case "nvidia-l4":
                gpu_type = "L4"
            case "nvidia-tesla-a100":
                gpu_type = "A100 40GB"
            case "nvidia-a100-80gb":
                gpu_type = "A100 80GB"
            case "nvidia-h100-80gb":
                # add 'GPU' so this case isnt a substring of H100 Mega
                gpu_type = "H100 80GB GPU"
            case "nvidia-h100-mega-80gb":
                gpu_type = "H100 80GB Plus"
            case _:
                raise ValueError(f"Unknown GPU type: {gpu_type}")

        gpu_resource_skus = [
            sku for sku in pricelist if "GPU" in sku["category"]["resourceGroup"]
        ]
        gpu_region_skus = [
            sku
            for sku in gpu_resource_skus
            if machine["region"] in sku["serviceRegions"]
        ]
        gpu_type_skus = [
            sku for sku in gpu_region_skus if gpu_type in sku["description"]
        ]
        gpu_skus = [
            sku for sku in gpu_type_skus if usage_type in sku["category"]["usageType"]
        ]
        # Edge-case where h100 mega gpu machines have
        # different sku names for CPU and RAM
        if gpu_type == "H100 80GB Plus":
            core_skus = [sku for sku in core_skus if "A3Plus" in sku["description"]]
            memory_skus = [sku for sku in memory_skus if "A3Plus" in sku["description"]]

    # Check that only 1 sku is returned for each category
    if len(core_skus) != 1:
        logging.error(f"Expected 1 sku for CPU, got {len(core_skus)}")
        logging.error(f"Skus: {core_skus}")
        raise ValueError(f"Expected 1 sku for CPU, got {len(core_skus)}")
    if len(memory_skus) != 1:
        logging.error(f"Expected 1 sku for RAM, got {len(memory_skus)}")
        logging.error(f"Skus: {memory_skus}")
        raise ValueError(f"Expected 1 sku for RAM, got {len(memory_skus)}")
    if num_gpus > 0 and len(gpu_skus) != 1:
        logging.error(f"Expected 1 sku for GPU, got {len(gpu_skus)}")
        logging.error(f"Skus: {gpu_skus}")
        raise ValueError(f"Expected 1 sku for GPU, got {len(gpu_skus)}")

    # Pricing api splits the price into units and nanos.
    # Units are whole dollars, nanos are cents but represented as nanos of a dollar
    cpu_dollars_price = core_skus[0]["pricingInfo"][0]["pricingExpression"][
        "tieredRates"
    ][-1]["unitPrice"]["units"]
    cpu_cents_price = core_skus[0]["pricingInfo"][0]["pricingExpression"][
        "tieredRates"
    ][-1]["unitPrice"]["nanos"] / (10**9)
    cpu_price_per_hr = (cpu_dollars_price + cpu_cents_price) * num_cpus
    ram_dollars_price = memory_skus[0]["pricingInfo"][0]["pricingExpression"][
        "tieredRates"
    ][-1]["unitPrice"]["units"]
    ram_cents_price = memory_skus[0]["pricingInfo"][0]["pricingExpression"][
        "tieredRates"
    ][-1]["unitPrice"]["nanos"] / (10**9)
    ram_price_per_hr = (ram_dollars_price + ram_cents_price) * num_ram_gb
    if num_gpus > 0:
        gpu_dollars_price = gpu_skus[0]["pricingInfo"][0]["pricingExpression"][
            "tieredRates"
        ][-1]["unitPrice"]["units"]
        gpu_cents_price = gpu_skus[0]["pricingInfo"][0]["pricingExpression"][
            "tieredRates"
        ][-1]["unitPrice"]["nanos"] / (10**9)
        gpu_price_per_hr = (gpu_dollars_price + gpu_cents_price) * num_gpus
        return cpu_price_per_hr + ram_price_per_hr + gpu_price_per_hr
    else:
        return cpu_price_per_hr + ram_price_per_hr


def get_disk_hour(machine, pricelist):
    # This function will ignore Hyperdisk Throughput Storage Pools cost since it is
    # billed monthly based on resources allocated to the pool, not the VM
    total = 0
    for disk in machine.get("disks"):
        search_term: str | None = None
        match disk["type"]:
            case "pd-standard":
                search_term = "Storage PD Capacity"
            case "pd-balanced":
                search_term = "Balanced PD Capacity"
            case "pd-ssd":
                search_term = "SSD backed PD Capacity"
            case "local-ssd":
                search_term = "SSD backed Local Storage"
            case "pd-extreme":
                search_term = "Extreme PD Capacity"
            case "hyperdisk-throughput":
                search_term = "Hyperdisk Throughput Capacity"
            case "hyperdisk-ml":
                search_term = "Hyperdisk ML Capacity"
            case "hyperdisk-extreme":
                search_term = "Hyperdisk Extreme Capacity"
            case "hyperdisk-balanced":
                search_term = "Hyperdisk Balanced Capacity"
            case _:
                raise ValueError(f"Unknown disk type: {disk['type']}")
        # Filter skus to be in the same region as the VM
        regional_price_skus = [
            sku for sku in pricelist if machine["region"] in sku["serviceRegions"]
        ]
        # All disks only have OnDemand billing
        filtered_price_skus = [
            sku
            for sku in regional_price_skus
            if "OnDemand" in sku["category"]["usageType"]
        ]
        filtered_price_skus = [
            sku
            for sku in filtered_price_skus
            if "Storage" in sku["category"]["resourceFamily"]
        ]
        # Assume all disks are not regional, HA, confidential, etc.
        # Use the following set of negative filters to remove those
        filtered_price_skus = [
            sku for sku in filtered_price_skus if "Pools" not in sku["description"]
        ]
        filtered_price_skus = [
            sku
            for sku in filtered_price_skus
            if "Confidential" not in sku["description"]
        ]
        filtered_price_skus = [
            sku for sku in filtered_price_skus if "Regional" not in sku["description"]
        ]
        filtered_price_skus = [
            sku
            for sku in filtered_price_skus
            if "High Availability" not in sku["description"]
        ]
        # Filter by disk type
        disk_sku = [
            sku for sku in filtered_price_skus if search_term in sku["description"]
        ]
        if len(disk_sku) != 1:
            logging.error(f"Expected 1 sku for disk, got {len(disk_sku)}")
            logging.error(f"Skus: {disk_sku}")
            raise ValueError(f"Expected 1 sku for disk, got {len(disk_sku)}")

        disk_price_gb_dollars = disk_sku[0]["pricingInfo"][0]["pricingExpression"][
            "tieredRates"
        ][-1]["unitPrice"]["units"]
        disk_price_gb_cents = disk_sku[0]["pricingInfo"][0]["pricingExpression"][
            "tieredRates"
        ][-1]["unitPrice"]["nanos"] / (10**9)
        # Disk prices are per month, need to convert to hourly, 730 hours in a month
        disk_price_gb = (disk_price_gb_dollars + disk_price_gb_cents) / 730

        price = disk_price_gb * disk["sizeGb"]
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


def report(gcp_variables, metrics_client):
    time_delta = time() - gcp_variables["last_time"]
    series = [
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
                "double_value": (disk_io("read_count") - gcp_variables["disk_reads"])
                / time_delta
            },
        ),
        get_time_series(
            gcp_variables,
            gcp_variables["DISK_WRITES_METRIC"],
            {
                "double_value": (disk_io("write_count") - gcp_variables["disk_writes"])
                / time_delta
            },
        ),
    ]
    if gcp_variables["PRICING_AVAILABLE"]:
        series.append(
            get_time_series(
                gcp_variables,
                gcp_variables["COST_ESTIMATE_METRIC"],
                {
                    "double_value": (time() - ps.boot_time())
                    * gcp_variables["COST_PER_SEC"]
                },
            ),
        )

    create_time_series(
        gcp_variables,
        metrics_client,
    )
    logging.info("Successfully wrote time series to Cloud Monitoring API.")

    return gcp_variables
