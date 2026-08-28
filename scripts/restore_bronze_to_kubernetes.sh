#!/usr/bin/env bash
set -euo pipefail

LOCAL_DATABASE_URL="${LOCAL_DATABASE_URL:-postgresql://basketballgpt:basketballgpt@db:5432/basketballgpt}"
LOCAL_DOCKER_NETWORK="${LOCAL_DOCKER_NETWORK:-basketballcrawler_default}"
K8S_DATABASE_URL="${K8S_DATABASE_URL:-}"
K8S_PGPASSWORD="${K8S_PGPASSWORD:-}"

if [[ -z "$K8S_DATABASE_URL" ]]; then
    echo "Set K8S_DATABASE_URL to the port-forwarded Kubernetes Postgres URL." >&2
    exit 1
fi

raw_dump="$(mktemp)"
filtered_dump="$(mktemp)"
cleanup() {
    rm -f "$raw_dump" "$filtered_dump"
}
trap cleanup EXIT

docker run --rm \
    --network "$LOCAL_DOCKER_NETWORK" \
    postgres:18 \
    pg_dump "$LOCAL_DATABASE_URL" \
        --schema bronze \
        --no-owner \
        --no-privileges \
    > "$raw_dump"

# The Kubernetes role can create objects inside the existing bronze schema but
# is not the owner of the schema itself, so skip schema creation.
grep -v '^CREATE SCHEMA bronze;$' "$raw_dump" > "$filtered_dump"

docker run --rm -i \
    -e "PGPASSWORD=$K8S_PGPASSWORD" \
    postgres:18 \
    psql "$K8S_DATABASE_URL" \
        --set ON_ERROR_STOP=1 \
        --quiet \
    < "$filtered_dump"
