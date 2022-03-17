# Cromwell task monitor

This repo is forked from [the Broad Institute](https://github.com/broadinstitute/cromwell-monitor) and contains code for monitoring resource utilization in
[Cromwell](https://github.com/broadinstitute/cromwell)
tasks running on
[Google Cloud Life Sciences API v2beta](https://cloud.google.com/life-sciences/docs/reference/rest/v2beta/projects.locations.pipelines/run).


### How to use Cromwell task monitor in VectorHive2

[Link to Demo & Tutorial](https://docs.google.com/presentation/d/1RnAq6LKC5zpLIXyNoVmNRNREfX0sZMDvzoyEwJtx49w/edit?usp=sharing)


### Setup a development environment

After cloning this repo, from within the root directory:
```bash
$ mamba env create --file requirements.dev
$ conda activate cromwell_monitor
$ pre-commit install # install git hook scripts
```


### How to test

Run unit tests with:

```bash
$ make test
```

### How does Cromwell task monitor work

The [monitoring script](monitor.py)
is intended to be used through a Docker image (as part of an associated "monitoring action"), currently built as
[gcr.io/dg-platform/vh2-cromwell-monitor](https://console.cloud.google.com/gcr/images/dg-platform/global/vh2-cromwell-monitor?project=dg-platform).

It uses [psutil](https://psutil.readthedocs.io) to
continuously measure CPU, memory and disk utilization
and disk IOPS, and periodically report them
as as [custom metrics to Cloud Monitoring API](https://cloud.google.com/monitoring/custom-metrics).

The labels for each time point contain the following metadata:
- Cromwell-specific values, such as workflow ID, task call name, index and attempt.
- GCP instance values such as instance name, zone, number of CPU cores, total memory and disk size.

This approach enables:

1)  Users to easily plot real-time resource usage statistics across all tasks in
    a workflow, or for a single task call across many workflow runs,
    etc.

    This monitoring tool can be very powerful to quickly determine the outlier tasks
    that could use optimization, without the need for any configuration
    or code.

2)  Scripts to easily get aggregate statistics
    on resource utilization and to produce suggestions
    based on those.

[TestMonitoring.wdl](TestMonitoring.wdl) can be used to
verify that the monitoring action/container is
working as intended.
