FROM cr.yandex/crpse3p7sm03fmuqh8ft/hammy-base:latest
WORKDIR /root
RUN curl https://clickhouse.com/ | sh
COPY chquery.sh /root/main.sh
COPY s3_access_key_id.cipher s3_secret_access_key.cipher clickhouse-local.xml /root
