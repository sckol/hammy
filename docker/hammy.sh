#!/bin/bash
if [ $# -ne 1 ]; then
  echo "Specify the number of minutes as argument. There were $# arguments ($@)"
  exit
else
  echo "Number of minutes is $1"
fi
sed  -i -e "s/XXX/$(yc kms symmetric-crypto decrypt --name hammy --ciphertext-file s3_access_key_id.cipher)/" -e "s/YYY/$(yc kms symmetric-crypto decrypt --name hammy --ciphertext-file s3_secret_access_key.cipher)/" .s3cfg
source run_nice.sh
run_nice python3 ./hammy.py "$@"