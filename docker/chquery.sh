cd /root
source run_nice.sh
sed  -i -e "s/XXX/$(yc kms symmetric-crypto decrypt --name hammy --ciphertext-file s3_access_key_id.cipher)/" -e "s/YYY/$(yc kms symmetric-crypto decrypt --name hammy --ciphertext-file s3_secret_access_key.cipher)/" clickhouse-local.xml
if [ $# -ne 1 ]; then
    query="SELECT 'Specify the query as the argument'"
else
    query="$1"
fi
echo "$query" > main_query.sql
echo "Executing the query beginning with " \"$(head -n2 main_query.sql)\" and ending with \"$(tail -n2 main_query.sql)\"
run_nice ./clickhouse --config-file clickhouse-local.xml --queries-file main_query.sql
echo "Query done"
