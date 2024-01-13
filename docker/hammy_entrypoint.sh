#!/bin/bash
cd /root
unified_agent &
source /root/.bashrc
yc config set instance-service-account true
source log_yc.sh
YC_INSTANCE_ID="$(curl -H Metadata-Flavor:Google 169.254.169.254/computeMetadata/v1/instance/id)"
log_yc bash main.sh "$@"
if [ -z "$ND" ]; then
    yc compute instance delete --id "$YC_INSTANCE_ID" --async
fi
if [ -z "$NS" ]; then
    echo 1 > /proc/sys/kernel/sysrq
    echo o > /proc/sysrq-trigger
fi