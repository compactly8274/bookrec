FROM python:3.12-slim

ARG UID=99
ARG GID=1000

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -g ${GID} -r bookrec && useradd -u ${UID} -r -g bookrec -d /app -s /sbin/nologin bookrec

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

ENV PYTHONUNBUFFERED=1
ENV MODEL_NAME=all-MiniLM-L6-v2

EXPOSE 8000

# /config must be writable by the unprivileged user; /calibre is read-only
RUN mkdir -p /config && chown -R bookrec:bookrec /config /app

WORKDIR /app/src
USER bookrec

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
