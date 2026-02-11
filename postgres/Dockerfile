FROM BASE_IMAGE
ARG VERSION=VERSION
ARG TARGETARCH
ARG STATELESS=false

LABEL name="pg-deeplake" \
  version="${VERSION}" \
  vendor="Activeloop" \
  license="/LICENSE" \
  description="PostgreSQL with DeepLake Extension installed." \
  contact="support@activeloop.ai"

ENV DEEPLAKE_LICENSE_FILE=/LICENSE
RUN apt-get update && \
  apt-get install --no-install-recommends -y \
  barman \
  barman-cli \
  barman-cli-cloud \
  libecpg6 \
  python3-pip \
  ca-certificates \
  libecpg-compat3 && \
  rm -rf /var/lib/apt/lists/* && \
  pip3 install --no-cache-dir deeplake --break-system-packages && \
  mkdir -p /etc/pki/tls/certs/ && \
  ln -s /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt

COPY ./debs/ /tmp/debs/
COPY --chmod=444 ./LICENSE /LICENSE
COPY ./postgres/docker-entrypoint.d/ /docker-entrypoint-initdb.d/
RUN apt-get install --no-install-recommends -y /tmp/debs/pg-deeplake-${VERSION}_${TARGETARCH}.deb && rm -rf /tmp/debs/
COPY ./serverless/scripts/init-deeplake-stateless.sh /tmp/init-deeplake-stateless.sh
COPY ./serverless/config/postgresql-overrides.conf /tmp/postgresql-overrides.conf
COPY ./serverless/scripts/health-check.sh /tmp/health-check.sh
RUN if [ "$STATELESS" = "true" ]; then \
      mv /tmp/init-deeplake-stateless.sh /docker-entrypoint-initdb.d/3-stateless-init.sh && \
      chmod 755 /docker-entrypoint-initdb.d/3-stateless-init.sh && \
      mv /tmp/postgresql-overrides.conf /etc/postgresql-overrides.conf && \
      chmod 644 /etc/postgresql-overrides.conf && \
      mv /tmp/health-check.sh /usr/local/bin/health-check.sh && \
      chmod 755 /usr/local/bin/health-check.sh && \
      mkdir -p /deeplake-data; \
    else \
      rm -f /tmp/init-deeplake-stateless.sh /tmp/postgresql-overrides.conf /tmp/health-check.sh; \
    fi
USER 999
