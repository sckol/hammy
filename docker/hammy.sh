#!/bin/bash
if [ $# -ne 2 ]; then
  echo "Specify the number of iterations per core and number of cores as arguments. There were $# arguments ($@)"
  exit
else
  echo "Number of iterations per core is $1, number of cores if $2"
fi
sed  -i -e "s/XXX/$(yc kms symmetric-crypto decrypt --name hammy --ciphertext-file s3_access_key_id.cipher)/" -e "s/YYY/$(yc kms symmetric-crypto decrypt --name hammy --ciphertext-file s3_secret_access_key.cipher)/" .s3cfg
source run_nice.sh
run_nice ./hammy "$@"
cd out
ls
fn=`ls *.parquet | head -n1`
if [ -z "$fn" ]; then
  echo "Parquet files not found"
  exit
fi
prefix=`echo $fn | sed -n "s/\([^_]\+\)_\([[:digit:]]\+\)_\([[:digit:]]\+\).*/\1\/\2\/\3/p"`
s3cmd put *.parquet "s3://hammy/$prefix/raw/"
s3cmd get "s3://hammy/stats.csv" stats-old.csv
if [ -f stats-old.csv ]; then
  tail -n +2 stats.csv >> stats-old.csv
  mv stats-old.csv stats.csv
fi
s3cmd put stats.csv "s3://hammy/stats.csv"
  