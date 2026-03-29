FROM cupy/cupy:v13.1.0
WORKDIR /root

RUN apt-get update && apt-get install -y curl wget unzip

# Yandex Cloud CLI (for self-destruct only)
RUN curl -sSL https://storage.yandexcloud.net/yandexcloud-yc/install.sh | bash -s -- -n -i /usr/local

# PCG random (C kernel dependency)
RUN wget -q https://www.pcg-random.org/downloads/pcg-c-basic-0.9.zip && \
    unzip -q -p pcg-c-basic-0.9.zip pcg-c-basic-0.9/pcg_basic.c > pcg_basic.c && \
    unzip -q -p pcg-c-basic-0.9.zip pcg-c-basic-0.9/pcg_basic.h > pcg_basic.h && \
    rm pcg-c-basic-0.9.zip

# Python deps (offline wheels to avoid PyPI connectivity issues)
COPY docker/wheels/ /tmp/wheels/
RUN pip install --no-index --find-links=/tmp/wheels/ xarray cffi psutil boto3 matplotlib h5netcdf h5py scs \
    && rm -rf /tmp/wheels/

# Hammy code
COPY hammy_lib/ /root/hammy/hammy_lib/
COPY experiments/ /root/hammy/experiments/
ENV PYTHONPATH=/root/hammy
ENV PYTHONUNBUFFERED=1

# Entrypoint
COPY docker/entrypoint.sh /root/entrypoint.sh
RUN chmod +x /root/entrypoint.sh
ENTRYPOINT ["/root/entrypoint.sh"]
