param(
    [string]$SourceDatabaseUrl = $env:SOURCE_DATABASE_URL,
    [string]$TargetDatabaseUrl = $env:DATABASE_URL,
    [string]$SourceSchema = $(if ($env:SOURCE_DB_SCHEMA) { $env:SOURCE_DB_SCHEMA } else { "bronze" })
)

$ErrorActionPreference = "Stop"

if (-not $SourceDatabaseUrl) {
    throw "Set SOURCE_DATABASE_URL to the port-forwarded cluster Postgres URL."
}

if (-not $TargetDatabaseUrl) {
    $TargetDatabaseUrl = "postgresql://basketballgpt:basketballgpt@localhost:55432/basketballgpt"
}

if ($SourceSchema -notmatch '^[A-Za-z_][A-Za-z0-9_]*$') {
    throw "Schema must be a simple PostgreSQL identifier."
}

function Convert-LocalhostForDocker([string]$Url) {
    return $Url -replace "@localhost:", "@host.docker.internal:" -replace "@127\.0\.0\.1:", "@host.docker.internal:"
}

$sourceInDocker = if ($env:SOURCE_DATABASE_URL_CONTAINER) {
    $env:SOURCE_DATABASE_URL_CONTAINER
} else {
    Convert-LocalhostForDocker $SourceDatabaseUrl
}

$targetInDocker = if ($env:TARGET_DATABASE_URL_CONTAINER) {
    $env:TARGET_DATABASE_URL_CONTAINER
} else {
    Convert-LocalhostForDocker $TargetDatabaseUrl
}

docker compose up -d db

docker run --rm `
    -e "DST=$targetInDocker" `
    -e "SCHEMA=$SourceSchema" `
    postgres:18 `
    psql --dbname "$targetInDocker" -v ON_ERROR_STOP=1 -c "DROP SCHEMA IF EXISTS $SourceSchema CASCADE"

docker run --rm `
    -e "SRC=$sourceInDocker" `
    -e "DST=$targetInDocker" `
    -e "SCHEMA=$SourceSchema" `
    postgres:18 `
    sh -lc 'pg_dump --dbname "$SRC" --schema "$SCHEMA" --no-owner --no-privileges | sed "/transaction_timeout/d" | psql --dbname "$DST" -v ON_ERROR_STOP=1'
