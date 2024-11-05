import json
from typing import List
from unittest.mock import Mock, patch

import pytest
import requests_mock

from gcp_monitor import (
    get_disk_hour,
    get_machine_hour,
    get_machine_info,
    initialize_gcp_variables,
)

test_metadata_payload = {
    "attributes": {"ssh-keys": "some-ssh-key-value"},
    "cpuPlatform": "Intel Broadwell",
    "description": "",
    "disks": [
        {
            "deviceName": "persistent-disk-0",
            "index": 0,
            "interface": "SCSI",
            "mode": "READ_WRITE",
            "type": "PERSISTENT",
        }
    ],
    "guestAttributes": {},
    "hostname": "cromwell-monitor-test.c.dg-devel.internal",
    "id": 3459161384849251192,
    "image": "projects/ubuntu-os-cloud/global/images/ubuntu-2004-focal-v20220204",
    "legacyEndpointAccess": {"0.1": 0, "v1beta1": 0},
    "licenses": [{"id": "1234567890"}],
    "machineType": "projects/642504272574/machineTypes/n1-standard-2",
    "maintenanceEvent": "NONE",
    "name": "cromwell-monitor-test",
    "networkInterfaces": [
        {
            "accessConfigs": [
                {"externalIp": "34.130.192.185", "type": "ONE_TO_ONE_NAT"}
            ],
            "dnsServers": ["169.254.169.254"],
            "forwardedIps": [],
            "gateway": "10.188.0.1",
            "ip": "10.188.0.7",
            "ipAliases": [],
            "mac": "42:01:0a:bc:00:07",
            "mtu": 1460,
            "network": "projects/642504272574/networks/default",
            "subnetmask": "255.255.240.0",
            "targetInstanceIps": [],
        }
    ],
    "preempted": "FALSE",
    "remainingCpuTime": -1,
    "scheduling": {
        "automaticRestart": "FALSE",
        "onHostMaintenance": "TERMINATE",
        "preemptible": "TRUE",
    },
    "serviceAccounts": {
        "642504272574-compute@developer.gserviceaccount.com": {
            "aliases": ["default"],
            "email": "642504272574-compute@developer.gserviceaccount.com",
            "scopes": ["https://www.googleapis.com/auth/cloud-platform"],
        },
        "default": {
            "aliases": ["default"],
            "email": "642504272574-compute@developer.gserviceaccount.com",
            "scopes": ["https://www.googleapis.com/auth/cloud-platform"],
        },
    },
    "tags": [],
    "virtualClock": {"driftToken": "0"},
    "zone": "projects/642504272574/zones/northamerica-northeast2-a",
}

test_metric_response = """
name:
  "projects/dg-devel/metricDescriptors/custom.googleapis.com/wdl_task/cpu_utilization"
labels {
  key: "workflow_id"
  description: "Cromwell workflow ID"
}
labels {
  key: "task_call_name"
  description: "Cromwell task call name"
}
labels {
  key: "task_call_index"
  description: "Cromwell task call index"
}
labels {
  key: "task_call_attempt"
  description: "Cromwell task call attempt"
}
labels {
  key: "cpu_count"
  description: "Number of virtual cores"
}
labels {
  key: "mem_size"
  description: "Total memory size, GB"
}
labels {
  key: "disk_size"
  description: "Total disk size, GB"
}
labels {
  key: "preemptible"
  description: "Preemptible flag"
}
labels {
  key: "owner"
  description: "Owner Label defined by user in VectorHive2"
}
labels {
  key: "entrance_wdl"
  description: "Entrance WDL Label defined by VectorHive2"
}
metric_kind: GAUGE
value_type: DOUBLE
unit: "%"
description: "% of CPU utilized in a Cromwell task call"
type: "custom.googleapis.com/wdl_task/cpu_utilization"
"""

test_instance_payload = {
    "kind": "compute#instance",
    "id": "3459161384849251192",
    "creationTimestamp": "2022-02-28T20:48:23.968-08:00",
    "name": "cromwell-monitor-test",
    "tags": {"fingerprint": "42WmSpB8rSM="},
    "machineType": "https://www.googleapis.com/"
    + "compute/v1/projects/dg-devel/"
    + "zones/northamerica-northeast2-a/machineTypes/n1-standard-2",
    "status": "RUNNING",
    "zone": "https://www.googleapis.com/"
    + "compute/v1/projects/dg-devel/zones/northamerica-northeast2-a",
    "canIpForward": False,
    "networkInterfaces": [
        {
            "kind": "compute#networkInterface",
            "network": "https://www.googleapis.com/"
            + "compute/v1/projects/dg-devel/global/networks/default",
            "subnetwork": "https://www.googleapis.com/"
            + "compute/v1/projects/dg-devel/"
            + "regions/northamerica-northeast2/subnetworks/default",
            "networkIP": "10.188.0.7",
            "name": "nic0",
            "accessConfigs": [
                {
                    "kind": "compute#accessConfig",
                    "type": "ONE_TO_ONE_NAT",
                    "name": "external-nat",
                    "natIP": "34.130.192.185",
                    "networkTier": "PREMIUM",
                }
            ],
            "fingerprint": "-6kaaHLQ_wM=",
            "stackType": "IPV4_ONLY",
        }
    ],
    "disks": [
        {
            "kind": "compute#attachedDisk",
            "type": "PERSISTENT",
            "mode": "READ_WRITE",
            "source": "https://www.googleapis.com/"
            + "compute/v1/projects/dg-devel/"
            + "zones/northamerica-northeast2-a/disks/cromwell-monitor-test",
            "deviceName": "persistent-disk-0",
            "index": 0,
            "boot": True,
            "autoDelete": True,
            "licenses": [
                "https://www.googleapis.com/"
                + "compute/v1/"
                + "projects/ubuntu-os-cloud/global/licenses/ubuntu-2004-lts"
            ],
            "interface": "SCSI",
            "guestOsFeatures": [
                {"type": "VIRTIO_SCSI_MULTIQUEUE"},
                {"type": "SEV_CAPABLE"},
                {"type": "UEFI_COMPATIBLE"},
                {"type": "GVNIC"},
            ],
            "diskSizeGb": "20",
            "shieldedInstanceInitialState": {
                "dbxs": [
                    {
                        "content": "some_content",
                        "fileType": "BIN",
                    }
                ]
            },
        }
    ],
    "metadata": {
        "kind": "compute#metadata",
        "fingerprint": "sfUKjUNBk-8=",
        "items": [
            {
                "key": "ssh-keys",
                "value": "some_value",
            }
        ],
    },
    "serviceAccounts": [
        {
            "email": "642504272574-compute@developer.gserviceaccount.com",
            "scopes": ["https://www.googleapis.com/auth/cloud-platform"],
        }
    ],
    "selfLink": "https://www.googleapis.com/"
    + "compute/v1/projects/dg-devel/"
    + "zones/northamerica-northeast2-a/instances/cromwell-monitor-test",
    "scheduling": {
        "onHostMaintenance": "TERMINATE",
        "automaticRestart": False,
        "preemptible": True,
    },
    "cpuPlatform": "Intel Broadwell",
    "labels": {
        "owner": "test_owner",
        "test": "cromwell-monitoring",
        "created_by": "peter",
        "entrance-wdl": "label1",
    },
    "labelFingerprint": "P-yFK-2-QSM=",
    "startRestricted": False,
    "deletionProtection": False,
    "shieldedInstanceConfig": {
        "enableSecureBoot": False,
        "enableVtpm": True,
        "enableIntegrityMonitoring": True,
    },
    "shieldedInstanceIntegrityPolicy": {"updateAutoLearnPolicy": True},
    "fingerprint": "KsLNMAaXiXQ=",
    "advancedMachineFeatures": {"enableNestedVirtualization": True},
    "lastStartTimestamp": "2022-03-16T15:25:06.813-07:00",
    "lastStopTimestamp": "2022-03-16T15:06:40.137-07:00",
}


@requests_mock.Mocker(kw="requests")
def test_initialize_gcp_variables(**kwargs):
    kwargs["requests"].get(
        "http://metadata.google.internal/"
        + "computeMetadata/v1/instance/?recursive=true",
        json=test_metadata_payload,
    )

    test_environ_variables = {
        "WORKFLOW_ID": "17399163265929080700",
        "TASK_CALL_NAME": "unit_test",
        "TASK_CALL_INDEX": "0",
        "TASK_CALL_ATTEMPT": "0",
        "DISK_MOUNTS": "/",
        "OWNER": "test_owner",
        "ENTRACNCE_WDL": "label1",
    }

    test_machine_info_output = {
        "project": "642504272574",
        "zone": "northamerica-northeast2-a",
        "region": "northamerica-northeast2",
        "name": "cromwell-monitor-test",
        "type": "n1-standard-2",
        "preemptible": True,
        "disks": [{"type": "pd-standard", "sizeGb": 20}],
        "owner": "test_owner",
        "entrance_wdl": "label1",
    }

    with patch("gcp_monitor.get_metric", return_value=test_metric_response), patch.dict(
        "os.environ", test_environ_variables
    ), patch(
        "gcp_monitor.get_machine_info", return_value=test_machine_info_output
    ), patch(
        "pynvml.nvmlDeviceGetCount", return_value=2
    ):
        actual_instance, _ = initialize_gcp_variables(nvml_ok=True)

    assert actual_instance["WORKFLOW_ID"] == "17399163265929080700"
    assert actual_instance["TASK_CALL_NAME"] == "unit_test"
    assert actual_instance["OWNER"] == "test_owner"
    assert (
        actual_instance["ENTRANCE_WDL"]
        == test_instance_payload["labels"]["entrance-wdl"]
    )
    for i in [0, 1]:
        for metric in [
            "UTILIZATION_METRIC",
            "MEM_UTILIZATION_METRIC",
            "MEM_ALLOCATED_METRIC",
        ]:
            assert f"GPU{i}_{metric}" in actual_instance


@requests_mock.Mocker(kw="requests")
def test_get_machine_info(**kwargs):
    kwargs["requests"].get(
        "http://metadata.google.internal/"
        + "computeMetadata/v1/instance/?recursive=true",
        json=test_metadata_payload,
    )
    expected_machine_info_output = {
        "project": "642504272574",
        "zone": "northamerica-northeast2-a",
        "region": "northamerica-northeast2",
        "name": "cromwell-monitor-test",
        "type": "n1-standard-2",
        "preemptible": True,
        "disks": [{"type": "pd-standard", "sizeGb": 20}],
        "owner": "test_owner",
        "entrance_wdl": "label1",
        "gpu_count": 0,
        "gpu_type": None,
    }

    test_environ_variables = {
        "WORKFLOW_ID": "17399163265929080700",
        "TASK_CALL_NAME": "unit_test",
        "TASK_CALL_INDEX": "0",
        "TASK_CALL_ATTEMPT": "0",
        "DISK_MOUNTS": "/",
    }
    # to mock chained call such as compute.instances().get().execute()
    compute = Mock()
    instances = compute.instances.return_value
    get = instances.get.return_value
    get.execute.return_value = test_instance_payload
    with patch(
        "gcp_monitor.get_disk", return_value={"type": "pd-standard", "sizeGb": 20}
    ), patch.dict("os.environ", test_environ_variables):
        actual_machine_info_output = get_machine_info(compute)

    assert actual_machine_info_output == expected_machine_info_output


def get_services_pricelist(filepath: str) -> List[dict]:
    with open(filepath) as f:
        return json.load(f)


def get_nanodollar_price(sku_id: str, pricelist: List[dict]) -> tuple[int, int]:
    sku = next(sku for sku in pricelist if sku_id in sku["skuId"])
    units = (
        int(
            sku["pricingInfo"][0]["pricingExpression"]["tieredRates"][-1]["unitPrice"][
                "units"
            ]
        )
        * 10**9
    )
    nanos = sku["pricingInfo"][0]["pricingExpression"]["tieredRates"][-1]["unitPrice"][
        "nanos"
    ]
    return units + nanos


def test_get_machine_hour_n1_standard():
    n1_machine = {
        "project": "642504272574",
        "zone": "us-central1-a",
        "region": "us-central1",
        "name": "cromwell-monitor-test",
        "type": "n1-standard-2",
        "preemptible": True,
        "disks": [{"type": "pd-standard", "sizeGb": 20}],
        "owner": "test_owner",
        "entrance_wdl": "label1",
        "gpu_count": 0,
        "gpu_type": None,
    }
    pricelist = get_services_pricelist(
        "tests/data/test_get_machine_hour_n1_standard_pricelist.json"
    )
    ram_sku_id = "5451-0A15-0123"
    ram_nanodollar_price = get_nanodollar_price(ram_sku_id, pricelist)
    cpu_sku_id = "D498-1ECA-87C1"
    cpu_nanodollar_price = get_nanodollar_price(cpu_sku_id, pricelist)
    num_ram_gb = 7.5
    ram_cost_hr = ram_nanodollar_price * num_ram_gb
    num_cpus = 2
    cpu_cost_hr = cpu_nanodollar_price * num_cpus
    with patch("os.cpu_count", return_value=num_cpus), patch(
        "psutil.virtual_memory"
    ) as mock_virtual_memory:
        mock_virtual_memory.return_value.total = num_ram_gb * (
            1024**3
        )  # convert Gib to bytes
        actual = get_machine_hour(n1_machine, pricelist)
    assert actual == cpu_cost_hr + ram_cost_hr


def test_get_machine_hour_ondemand_n1_custom_ext():
    n1_custom_extended_machine = {
        "project": "642504272574",
        "zone": "us-central1-a",
        "region": "us-central1",
        "name": "cromwell-monitor-test",
        "type": "custom-2-4096-ext",
        "preemptible": False,
        "disks": [{"type": "pd-standard", "sizeGb": 20}],
        "owner": "test_owner",
        "entrance_wdl": "label1",
        "gpu_count": 0,
        "gpu_type": None,
    }
    pricelist = get_services_pricelist(
        "tests/data/test_get_machine_hour_ondemand_n1_custom_ext_pricelist.json"
    )
    ram_sku = "972B-1B48-9D16"
    ram_nanodollar_price = get_nanodollar_price(ram_sku, pricelist)
    cpu_sku = "ACBC-6999-A1C4"
    cpu_nanodollar_price = get_nanodollar_price(cpu_sku, pricelist)
    num_ram_gb = 4
    ram_cost_hr = ram_nanodollar_price * num_ram_gb
    num_cpus = 2
    cpu_cost_hr = cpu_nanodollar_price * num_cpus
    with patch("os.cpu_count", return_value=num_cpus), patch(
        "psutil.virtual_memory"
    ) as mock_virtual_memory:
        mock_virtual_memory.return_value.total = num_ram_gb * (
            1024**3
        )  # convert Gib to bytes
        actual = get_machine_hour(n1_custom_extended_machine, pricelist)
    assert actual == cpu_cost_hr + ram_cost_hr


def test_get_machine_hour_preemptible_n1_custom_ext():
    n1_custom_extended_machine = {
        "project": "642504272574",
        "zone": "us-central1-a",
        "region": "us-central1",
        "name": "cromwell-monitor-test",
        "type": "custom-2-4096-ext",
        "preemptible": True,
        "disks": [{"type": "pd-standard", "sizeGb": 20}],
        "owner": "test_owner",
        "entrance_wdl": "label1",
        "gpu_count": 0,
        "gpu_type": None,
    }
    pricelist = get_services_pricelist(
        "tests/data/test_get_machine_hour_preemptible_n1_custom_ext_pricelist.json"
    )
    ram_sku_id = "C1E6-3CA5-CE59"
    ram_nanodollar_price = get_nanodollar_price(ram_sku_id, pricelist)
    cpu_sku_id = "4A30-9DBE-ECEA"
    cpu_nanodollar_price = get_nanodollar_price(cpu_sku_id, pricelist)
    num_ram_gb = 4
    ram_cost_hr = ram_nanodollar_price * num_ram_gb
    num_cpus = 2
    cpu_cost_hr = cpu_nanodollar_price * num_cpus
    with patch("os.cpu_count", return_value=num_cpus), patch(
        "psutil.virtual_memory"
    ) as mock_virtual_memory:
        mock_virtual_memory.return_value.total = num_ram_gb * (
            1024**3
        )  # convert Gib to bytes
        actual = get_machine_hour(n1_custom_extended_machine, pricelist)
    assert actual == cpu_cost_hr + ram_cost_hr


def test_get_machine_hour_h100_mega():
    h100_mega_machine = {
        "project": "642504272574",
        "zone": "us-central1-a",
        "region": "us-central1",
        "name": "cromwell-monitor-test",
        "type": "a3-highgpu-8g",
        "preemptible": True,
        "disks": [{"type": "pd-standard", "sizeGb": 20}],
        "owner": "test_owner",
        "entrance_wdl": "label1",
        "gpu_count": 8,
        "gpu_type": "nvidia-h100-mega-80gb",
    }
    pricelist = get_services_pricelist(
        "tests/data/test_get_machine_hour_h100_mega_pricelist.json"
    )
    ram_sku_id = "9A1D-C6C8-D7B9"
    ram_nanodollar_price = get_nanodollar_price(ram_sku_id, pricelist)
    cpu_sku_id = "AEAF-12C5-E41B"
    cpu_nanodollar_price = get_nanodollar_price(cpu_sku_id, pricelist)
    num_ram_gb = 1872
    ram_cost_hr = ram_nanodollar_price * num_ram_gb
    num_cpus = 208
    cpu_cost_hr = cpu_nanodollar_price * num_cpus
    gpu_sku_id = "8609-4BAD-F240"
    gpu_nanodollar_price = get_nanodollar_price(gpu_sku_id, pricelist)
    num_gpus = 8
    gpu_cost_hr = gpu_nanodollar_price * num_gpus
    with patch("os.cpu_count", return_value=num_cpus), patch(
        "psutil.virtual_memory"
    ) as mock_virtual_memory:
        mock_virtual_memory.return_value.total = num_ram_gb * (
            1024**3
        )  # convert Gib to bytes
        actual = get_machine_hour(h100_mega_machine, pricelist)
    assert actual == cpu_cost_hr + ram_cost_hr + gpu_cost_hr


def test_get_machine_hour_no_skus():
    machine = {
        "project": "642504272574",
        "zone": "us-central1-a",
        "region": "us-central1",
        "name": "cromwell-monitor-test",
        "type": "n1-standard-2",
        "preemptible": True,
        "disks": [{"type": "pd-standard", "sizeGb": 20}],
        "owner": "test_owner",
        "entrance_wdl": "label1",
        "gpu_count": 0,
        "gpu_type": None,
    }
    pricelist = [
        {
            "serviceRegions": ["us-central1"],
            "description": "Random machine type",
            "category": {
                "usageType": "OnDemand",
                "resourceFamily": "Compute",
                "resourceGroup": "Filler",
            },
        }
    ]
    with pytest.raises(ValueError):
        get_machine_hour(machine, pricelist=pricelist)


def test_get_machine_hour_too_many_skus():
    h100_mega_machine = {
        "project": "642504272574",
        "zone": "us-central1-a",
        "region": "us-central1",
        "name": "cromwell-monitor-test",
        "type": "a3-highgpu-8g",
        "preemptible": True,
        "disks": [{"type": "pd-standard", "sizeGb": 20}],
        "owner": "test_owner",
        "entrance_wdl": "label1",
        "gpu_count": 8,
        "gpu_type": "nvidia-h100-mega-80gb",
    }
    extra_sku = {
        "serviceRegions": ["us-central1"],
        "description": "A3Plus machine type with H100 80GB Plus GPU",
        "category": {
            "usageType": "Preemptible",
            "resourceFamily": "Compute",
            "resourceGroup": "GPU",
        },
    }
    pricelist = get_services_pricelist(
        "tests/data/test_get_machine_hour_h100_mega_pricelist.json"
    ) + [extra_sku]
    with pytest.raises(ValueError):
        get_machine_hour(h100_mega_machine, pricelist=pricelist)


def test_get_machine_hour_unknown_gpu():
    fake_gpu_machine = {
        "project": "642504272574",
        "zone": "us-central1-a",
        "region": "us-central1",
        "name": "cromwell-monitor-test",
        "type": "a3-highgpu-8g",
        "preemptible": True,
        "disks": [{"type": "pd-standard", "sizeGb": 20}],
        "owner": "test_owner",
        "entrance_wdl": "label1",
        "gpu_count": 8,
        "gpu_type": "fake-gpu",
    }
    with pytest.raises(ValueError):
        get_machine_hour(
            fake_gpu_machine,
            get_services_pricelist(
                "tests/data/test_get_machine_hour_h100_mega_pricelist.json"
            ),
        )


def test_get_disk_hour():
    pd_standard_machine = {
        "project": "642504272574",
        "zone": "us-central1-a",
        "region": "us-central1",
        "name": "cromwell-monitor-test",
        "type": "n1-standard-2",
        "preemptible": True,
        "disks": [
            {"type": "pd-standard", "sizeGb": 20},
            {"type": "pd-ssd", "sizeGb": 30},
        ],
        "owner": "test_owner",
        "entrance_wdl": "label1",
        "gpu_count": 0,
        "gpu_type": None,
    }
    pricelist = get_services_pricelist("tests/data/test_get_disk_hour_pricelist.json")
    actual = get_disk_hour(pd_standard_machine, pricelist)
    disk_standard_sku_id = "D973-5D65-BAB2"
    disk_standard_nandollars = get_nanodollar_price(disk_standard_sku_id, pricelist)
    num_disk_standard_gb = 20
    # disk price is per month, 730 hrs in a month
    disk_standard_cost_hr = int((disk_standard_nandollars / 730) * num_disk_standard_gb)
    disk_ssd_sku_id = "B188-61DD-52E4"
    disk_ssd_nanodollars = get_nanodollar_price(disk_ssd_sku_id, pricelist)
    num_disk_ssd_gb = 30
    disk_ssd_cost_hr = int((disk_ssd_nanodollars / 730) * num_disk_ssd_gb)
    assert actual == disk_standard_cost_hr + disk_ssd_cost_hr


def test_get_disk_hour_no_skus():
    machine = {
        "project": "642504272574",
        "zone": "us-central1-a",
        "region": "us-central1",
        "name": "cromwell-monitor-test",
        "type": "n1-standard-2",
        "preemptible": True,
        "disks": [{"type": "pd-standard", "sizeGb": 20}],
        "owner": "test_owner",
        "entrance_wdl": "label1",
        "gpu_count": 0,
        "gpu_type": None,
    }
    pricelist = [
        {
            "serviceRegions": ["us-central1"],
            "description": "Random machine type",
            "category": {
                "usageType": "OnDemand",
                "resourceFamily": "Compute",
                "resourceGroup": "Filler",
            },
        }
    ]
    with pytest.raises(ValueError):
        get_disk_hour(machine, pricelist=pricelist)


def test_get_disk_hour_too_many_skus():
    pd_standard_machine = {
        "project": "642504272574",
        "zone": "us-central1-a",
        "region": "us-central1",
        "name": "cromwell-monitor-test",
        "type": "n1-standard-2",
        "preemptible": True,
        "disks": [{"type": "pd-standard", "sizeGb": 20}],
        "owner": "test_owner",
        "entrance_wdl": "label1",
        "gpu_count": 0,
        "gpu_type": None,
    }
    extra_sku = {
        "serviceRegions": ["us-central1"],
        "description": "Storage PD Capacity",
        "category": {
            "usageType": "OnDemand",
            "resourceFamily": "Storage",
            "resourceGroup": "Disk",
        },
    }
    pricelist = get_services_pricelist(
        "tests/data/test_get_disk_hour_pricelist.json"
    ) + [extra_sku]
    with pytest.raises(ValueError):
        get_disk_hour(pd_standard_machine, pricelist=pricelist)


def test_get_disk_hour_unknown_disk():
    fake_disk_machine = {
        "project": "642504272574",
        "zone": "us-central1-a",
        "region": "us-central1",
        "name": "cromwell-monitor-test",
        "type": "n1-standard-2",
        "preemptible": True,
        "disks": [{"type": "fake-disk", "sizeGb": 20}],
        "owner": "test_owner",
        "entrance_wdl": "label1",
        "gpu_count": 0,
        "gpu_type": None,
    }
    with pytest.raises(ValueError):
        get_disk_hour(
            fake_disk_machine,
            get_services_pricelist("tests/data/test_get_disk_hour_pricelist.json"),
        )
