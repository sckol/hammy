FROM cr.yandex/crpse3p7sm03fmuqh8ft/hammy-base:latest
WORKDIR /root
RUN apt update && apt install -y pip && pip install s3cmd python-magic
RUN apt install -y libbrotli-dev libbz2-dev libcurl4-openssl-dev liblz4-dev \ 
  libc-ares-dev libre2-dev libsnappy-dev libssl-dev libutf8proc-dev libzstd-dev \
  nlohmann-json3-dev protobuf-compiler-grpc zlib1g-dev libthrift-dev
COPY docker/hammy.sh /root/main.sh
COPY build/hammy \
  docker/.s3cfg \
  docker/s3_access_key_id.cipher \
  docker/s3_secret_access_key.cipher /root  