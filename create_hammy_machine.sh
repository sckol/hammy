#!/bin/bash
set -e

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <version> <experiment-args> [--cpu] [--experiment MODULE]"
    echo "  $0 5.1 \"--level 2 --no-calculations\"                           # GPU, experiment 1"
    echo "  $0 5.1 \"--level 4\" --experiment experiments.02_lattice_types    # GPU, experiment 2"
    echo "  $0 5.1 \"--level 2 --no-calculations\" --cpu                     # CPU only"
    exit 0
fi

VERSION=$1; ARGS=$2; shift 2
MODE="gpu"; EXPERIMENT="experiments.01_walk"
while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu) MODE="cpu"; shift ;;
        --experiment) EXPERIMENT="$2"; shift 2 ;;
        *) shift ;;
    esac
done

# S3 credentials from secrets.env
eval $(~/bin/secrets-extract HAMMY_S3_ACCESS_KEY HAMMY_S3_SECRET_KEY)

if [ "$MODE" = "--cpu" ]; then
    COMPOSE=docker/hammy_machine_cpu.compose
    PLATFORM_ARGS="--cores 2 --memory 4G --platform standard-v3"
else
    COMPOSE=docker/hammy_machine.compose
    PLATFORM_ARGS="--cores 8 --gpus 1 --memory 48G --platform gpu-standard-v2"
fi

sed -e "s/XXX/$VERSION/" -e "s|YYY|$ARGS|" \
    -e "s/AAA/$HAMMY_S3_ACCESS_KEY/" -e "s|SSS|$HAMMY_S3_SECRET_KEY|" \
    -e "s|EEE|$EXPERIMENT|" \
    "$COMPOSE" > hammy_machine.compose.tmp

~/yandex-cloud/bin/yc compute instance create-with-container \
    $PLATFORM_ARGS \
    --zone ru-central1-a \
    --preemptible \
    --ssh-key ~/.ssh/id_ed25519.pub \
    --service-account-name compute \
    --docker-compose-file hammy_machine.compose.tmp \
    --async \
    --public-ip

rm -f hammy_machine.compose.tmp
