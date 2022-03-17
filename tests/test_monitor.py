import json
from unittest.mock import patch

import requests_mock
from googleapiclient.discovery import build as google_api

from gcp_monitor import gcp_monitor_variables, get_machine_info

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
  key: "owner_label"
  description: "Owner Label defined by user in VectorHive2"
}
labels {
  key: "entrance_wdl_label"
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
        "owner": "fake_payload",
        "test": "cromwell-monitoring",
        "created_by": "peter",
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


@requests_mock.Mocker(kw="mock")
def test_gcp_instance_workflow_id(**kwargs):
    kwargs["mock"].get(
        "http://metadata.google.internal/"
        + "computeMetadata/v1/instance/?recursive=true",
        json=test_metadata_payload,
    )
    with open("tests/data/pricelist.json") as reader:
        test_pricelist = json.load(reader)["gcp_price_list"]
        with patch("gcp_monitor.get_pricelist", return_value=test_pricelist), patch(
            "gcp_monitor.get_metric", return_value=test_metric_response
        ):
            expected_instance = gcp_monitor_variables()
    actual_instance_workflow_id = "17399163265929080700"

    assert expected_instance.WORKFLOW_ID == actual_instance_workflow_id


@requests_mock.Mocker(kw="mock")
def test_gcp_instance_task_call_name(**kwargs):
    kwargs["mock"].get(
        "http://metadata.google.internal/"
        + "computeMetadata/v1/instance/?recursive=true",
        json=test_metadata_payload,
    )
    with open("tests/data/pricelist.json") as reader:
        test_pricelist = json.load(reader)["gcp_price_list"]
        with patch("gcp_monitor.get_pricelist", return_value=test_pricelist), patch(
            "gcp_monitor.get_metric", return_value=test_metric_response
        ):
            expected_instance = gcp_monitor_variables()
    actual_instance_task_call_name = "unit_test"

    assert expected_instance.TASK_CALL_NAME == actual_instance_task_call_name


@requests_mock.Mocker(kw="mock")
def test_gcp_instance_owner_label(**kwargs):
    kwargs["mock"].get(
        "http://metadata.google.internal/"
        + "computeMetadata/v1/instance/?recursive=true",
        json=test_metadata_payload,
    )
    with open("tests/data/pricelist.json") as reader:
        test_pricelist = json.load(reader)["gcp_price_list"]
        with patch("gcp_monitor.get_pricelist", return_value=test_pricelist), patch(
            "gcp_monitor.get_metric", return_value=test_metric_response
        ):
            expected_instance = gcp_monitor_variables()
    actual_instance_owner_label = "test_owner"

    assert expected_instance.OWNER_LABEL == actual_instance_owner_label


@requests_mock.Mocker(kw="mock")
def test_gcp_instance_entrance_wdl_label(**kwargs):
    kwargs["mock"].get(
        "http://metadata.google.internal/"
        + "computeMetadata/v1/instance/?recursive=true",
        json=test_metadata_payload,
    )
    with open("tests/data/pricelist.json") as reader:
        test_pricelist = json.load(reader)["gcp_price_list"]
        with patch("gcp_monitor.get_pricelist", return_value=test_pricelist), patch(
            "gcp_monitor.get_metric", return_value=test_metric_response
        ):
            expected_instance = gcp_monitor_variables()
    actual_instance_entrance_wdl_label = ""

    assert expected_instance.ENTRANCE_WDL_LABEL == actual_instance_entrance_wdl_label


@requests_mock.Mocker(kw="mock")
def test_get_machine_info(**kwargs):
    kwargs["mock"].get(
        "http://metadata.google.internal/"
        + "computeMetadata/v1/instance/?recursive=true",
        json=test_metadata_payload,
    )
    actual_machine_info_output = {
        "project": "642504272574",
        "zone": "northamerica-northeast2-a",
        "region": "northamerica-northeast2",
        "name": "cromwell-monitor-test",
        "type": "n1-standard-2",
        "preemptible": True,
        "disks": [{"type": "pd-standard", "sizeGb": 20}],
        "owner_label": "test_owner",
    }
    compute = google_api("compute", "v1")
    with patch.object(
        compute, "instances.get.execute", return_value=test_instance_payload
    ):
        expected_machine_info_output = get_machine_info(compute)
    assert expected_machine_info_output == actual_machine_info_output
