#!/bin/bash
set -e
if [[ $# -lt 2 ]] ; then
    echo 'Provide 2 arguments: version, number of minutes'
    exit 0
fi
sed  -e "s/XXX/$1/" -e "s/YYY/$2/" hammy_machine.compose > hammy_machine.compose.tmp
yc compute instance create-with-container \
  --cores 8 \
  --gpus 1 \
  --memory 48G \
  --zone ru-central1-a \
  --preemptible \
  --platform gpu-standard-v2 \
  --ssh-key ~/.ssh/id_rsa.pub \
  --service-account-name compute \
  --docker-compose-file hammy_machine.compose.tmp \
  --async \
  --public-ip