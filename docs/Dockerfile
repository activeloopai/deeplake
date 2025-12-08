FROM squidfunk/mkdocs-material

ARG GIT_CONFIG=/etc/gitconfig

WORKDIR /indra/docs

RUN apk add --no-cache gcc python3-dev musl-dev linux-headers nodejs npm

RUN pip install --no-cache-dir \
        mistune==3.0.2 \
        black \
        mkdocs-awesome-pages-plugin \
        mkdocs-jupyter \
        mkdocs-redirects \
        mkdocs-glightbox \
        emoji \
        mike

ARG GITHUB_TOKEN

RUN if [ -z "${GITHUB_TOKEN}" ];then \
        >&2 echo "No GITHUB_TOKEN set, using regular library versions";\
        pip install --no-cache-dir griffe mkdocstrings-python; \
    else \
        >&2 echo "GITHUB_TOKEN set, using insiders library versions";\
        pip install git+https://${GITHUB_TOKEN}@github.com/pawamoy-insiders/griffe.git \
        git+https://${GITHUB_TOKEN}@github.com/pawamoy-insiders/mkdocstrings-python.git; \
    fi

RUN npm install -g typedoc typescript typedoc-plugin-markdown

COPY plugins /plugins
RUN pip install -e /plugins

ENV GIT_CONFIG=${GIT_CONFIG}
RUN git config --file "${GIT_CONFIG}" user.name "activeloop-bot" \
    && git config --file "${GIT_CONFIG}" user.email "bot@activeloop.ai"

ENTRYPOINT ["mkdocs"]
CMD ["serve", "--dev-addr=0.0.0.0:8000"]
