# Memory and CPU limits

## Memory
Every container must declare a memory limit. The minimum is 16 MB and the maximum is 1024 MB per container.

## CPU
CPU limits are expressed in cores and must be greater than 0 and at most 2.0. Default recommendation for web workloads is 0.5 cores.

## Requests above the limits
Requests above these limits are refused automatically. Ask the platform team to raise the quota instead of splitting a workload across many containers.
