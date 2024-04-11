FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04
USER root

RUN apt update
RUN apt install -y curl
RUN curl -sSL https://storage.yandexcloud.net/yandexcloud-yc/install.sh > yc_install.sh \
  && bash yc_install.sh -n -i /usr/local \
  && yc config profile create my-profile
COPY ymonitoring.yml /etc/yandex/unified_agent/config.yml
RUN apt install -y iproute2 \    
  && ua_version=$(curl -s https://storage.yandexcloud.net/yc-unified-agent/latest-version) \
  bash -c 'curl -s https://storage.yandexcloud.net/yc-unified-agent/releases/$ua_version/unified_agent > /usr/bin/unified_agent' \
  && chmod +x /usr/bin/unified_agent 

COPY log_yc.sh hammy_entrypoint.sh run_nice.sh /root
COPY countdown.sh /root/main.sh
RUN chmod +x /root/hammy_entrypoint.sh

ENTRYPOINT ["/root/hammy_entrypoint.sh"]