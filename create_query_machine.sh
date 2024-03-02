#!/bin/bash
if [[ $# -lt 3 ]] ; then
    echo 'Provide 3 arguments: path to the query, number of cores, memory in Gb'
    exit 0
fi
query=$(<$1)
yc compute instance create-with-container \
  --cores "$2" \
  --memory "$3"G \
  --zone ru-central1-d \
  --preemptible \
  --platform standard-v3 \
  --container-arg "$query" \
  --container-image cr.yandex/crpse3p7sm03fmuqh8ft/chquery:latest \
  --container-privileged \
  --ssh-key ~/.ssh/id_rsa.pub \
  --service-account-name compute \
  --async \
  --public-ip