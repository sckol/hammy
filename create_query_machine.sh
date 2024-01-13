#!/bin/bash
if [[ $# -lt 3 ]] ; then
    echo 'Provide at least 3 arguments: path to the query, number of cores, memory in Gb (optionally true for public-ip)'
    exit 0
fi
if [ -n "$4" ]; then
  public_ip="--public-ip"
fi
query=$(<$1)
yc compute instance create-with-container \
  --cores "$2" \
  --memory "$3"G \
  --zone ru-central1-a \
  --preemptible \
  --platform standard-v3 \
  --container-arg "$query" \
  --container-image cr.yandex/crpse3p7sm03fmuqh8ft/chquery:latest \
  --container-privileged \
  --ssh-key ~/.ssh/id_rsa.pub \
  --service-account-name compute \
  --async "$public_ip"