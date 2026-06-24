#!/usr/bin/env bash
# Load environment variables from repo-root .env and print their values.
#
# Source from other scripts so exports persist in the current shell:
#   source scripts/internal_evals/load_env.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "No .env file found at $ENV_FILE" >&2
  return 1 2>/dev/null || exit 1
fi

mapfile -t LOADED_ENV_VARS < <(
  sed -nE 's/^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=.*/\2/p' "$ENV_FILE"
)

set -a
# shellcheck source=/dev/null
source "$ENV_FILE"
set +a

echo "Loaded environment variables from $ENV_FILE:"
for var in "${LOADED_ENV_VARS[@]}"; do
  printf '  %s=%s\n' "$var" "${!var}"
done
