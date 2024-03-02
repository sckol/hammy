#!/bin/bash
if [[ $# -lt 3 ]] ; then
    echo 'Provide 3 arguments: version, number of iterations per core, number of cores'
    exit 0
fi
yc compute instance create-with-container \
  --cores "$3" \
  --memory "$3"G \
  --zone ru-central1-d \
  --preemptible \
  --platform standard-v3 \
  --container-arg "$2" --container-arg "$3" \
  --container-image "cr.yandex/crpse3p7sm03fmuqh8ft/hammy:$1" \
  --container-privileged \
  --ssh-key ~/.ssh/id_rsa.pub \
  --service-account-name compute \
  --async \
  --public-ip