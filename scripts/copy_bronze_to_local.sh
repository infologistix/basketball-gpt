#!/usr/bin/env bash
set -euo pipefail

SOURCE_DATABASE_URL="${SOURCE_DATABASE_URL:-}"
DATABASE_URL="${DATABASE_URL:-postgresql://basketballgpt:basketballgpt@localhost:55432/basketballgpt}"
SOURCE_DB_SCHEMA="${SOURCE_DB_SCHEMA:-bronze}"

if [[ -z "$SOURCE_DATABASE_URL" ]]; then
    echo "Set SOURCE_DATABASE_URL to the port-forwarded cluster Postgres URL." >&2
    exit 1
fi

if [[ ! "$SOURCE_DB_SCHEMA" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
    echo "Schema must be a simple PostgreSQL identifier." >&2
    exit 1
fi

convert_localhost_for_docker() {
    local url="$1"
    url="${url//@localhost:/@host.docker.internal:}"
    url="${url//@127.0.0.1:/@host.docker.internal:}"
    printf '%s' "$url"
}

SOURCE_IN_DOCKER="${SOURCE_DATABASE_URL_CONTAINER:-$(convert_localhost_for_docker "$SOURCE_DATABASE_URL")}"
TARGET_IN_DOCKER="${TARGET_DATABASE_URL_CONTAINER:-$(convert_localhost_for_docker "$DATABASE_URL")}"

docker compose up -d db

docker run --rm \
    postgres:18 \
    psql --dbname "$TARGET_IN_DOCKER" -v ON_ERROR_STOP=1 -c "DROP SCHEMA IF EXISTS $SOURCE_DB_SCHEMA CASCADE"

docker run --rm \
    -e "SRC=$SOURCE_IN_DOCKER" \
    -e "DST=$TARGET_IN_DOCKER" \
    -e "SCHEMA=$SOURCE_DB_SCHEMA" \
    postgres:18 \
    sh -lc 'pg_dump --dbname "$SRC" --schema "$SCHEMA" --no-owner --no-privileges | sed "/transaction_timeout/d" | psql --dbname "$DST" -v ON_ERROR_STOP=1'
