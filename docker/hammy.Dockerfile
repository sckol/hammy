FROM sckol/hammy-base:4.0fix
WORKDIR /root
RUN pip install cffi boto3 psutil pickle-blosc
RUN apt-get install wget unzip
RUN wget -q https://www.pcg-random.org/downloads/pcg-c-basic-0.9.zip && \
  unzip -q -p pcg-c-basic-0.9.zip pcg-c-basic-0.9/pcg_basic.c > pcg_basic.c && \
  unzip -q -p pcg-c-basic-0.9.zip pcg-c-basic-0.9/pcg_basic.h > pcg_basic.h
COPY docker/hammy.sh /root/main.sh
COPY simulator/hamlitonian/3/1/hamiltonian_3_1.py hammy.py
COPY docker/.s3cfg \
  docker/s3_access_key_id.cipher \
  docker/s3_secret_access_key.cipher /root  