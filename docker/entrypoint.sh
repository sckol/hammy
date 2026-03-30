#!/bin/bash
set -eo pipefail

# Instance ID for self-destruct
yc config set instance-service-account true
YC_INSTANCE_ID="$(curl -sH Metadata-Flavor:Google 169.254.169.254/computeMetadata/v1/instance/id)"

# Run experiment — S3 keys come from container environment
# HAMMY_EXPERIMENT selects the experiment module (default: experiments.01_walk)
EXPERIMENT="${HAMMY_EXPERIMENT:-experiments.01_walk}"
echo "Running: python3 -m $EXPERIMENT $@"
python3 -m "$EXPERIMENT" "$@" 2>&1 | tee /root/experiment.log
EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo "Success — self-destructing"
    yc compute instance delete --id "$YC_INSTANCE_ID" --async
    echo 1 > /proc/sys/kernel/sysrq && echo o > /proc/sysrq-trigger
else
    echo "FAILED (exit $EXIT_CODE) — VM alive for 5 min for debugging"
    (sleep 300 && yc compute instance delete --id "$YC_INSTANCE_ID" --async && echo 1 > /proc/sys/kernel/sysrq && echo o > /proc/sysrq-trigger) &
    wait
fi
