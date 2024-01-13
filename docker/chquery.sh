cd /root
source run_nice.sh
sed  -i -e "s/XXX/$(yc kms symmetric-crypto decrypt --name hammy --ciphertext-file s3_access_key_id.cipher)/" -e "s/YYY/$(yc kms symmetric-crypto decrypt --name hammy --ciphertext-file s3_secret_access_key.cipher)/" clickhouse-local.xml
if [ $# -ne 1 ]; then
    query="SELECT 'Specify the query as the argument'"
else
    query="$1"
fi
echo "Executing the query $query"
run_nice ./clickhouse --config-file /root/clickhouse-local.xml --query "$query"
yc compute instance delete --id "$YC_INSTANCE_ID" --async
echo "Query done"