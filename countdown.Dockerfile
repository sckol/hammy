# Use a lightweight base image
FROM ubuntu:22.04
USER root

RUN apt update
RUN apt install -y curl
RUN curl -sSL https://storage.yandexcloud.net/yandexcloud-yc/install.sh > yc_install.sh \
  && bash yc_install.sh -n -i /usr/local \
  && yc config profile create my-profile

COPY countdown.sh /countdown.sh
COPY log_yc.sh /log_yc.sh

CMD ["/bin/bash", "-c", "-l", "source /log_yc.sh && log_yc bash /countdown.sh"]
