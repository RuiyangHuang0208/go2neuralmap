#!/usr/bin/env bash
set -u

MAX_PARALLEL=2
DURATION="30m"
LOG_LEVEL="INFO"
REPEATS=5

EXPERIMENTS=(
  "demo_plane_9"
  "demo_plane_10"
  "rl_nav_4"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}" || exit 1

run_experiment() {
  local exp_name="$1"
  local run_idx="$2"
  echo "[eval] starting ${exp_name} repeat ${run_idx}/${REPEATS} for ${DURATION}"
  timeout -s INT "${DURATION}" \
    python scripts/experiments/starter.py \
      --log_level="${LOG_LEVEL}" \
      --my_exp_name "${exp_name}" \
      --headless
  local status=$?
  if [[ ${status} -eq 124 ]]; then
    echo "[eval] ${exp_name} repeat ${run_idx}/${REPEATS} reached ${DURATION}; timeout sent SIGINT"
    return 0
  fi
  echo "[eval] ${exp_name} repeat ${run_idx}/${REPEATS} exited with status ${status}"
  return "${status}"
}

pids=()
names=()
statuses=()

wait_for_one() {
  local pid
  local name
  local status

  pid="${pids[0]}"
  name="${names[0]}"
  wait "${pid}"
  status=$?
  statuses+=("${name}:${status}")
  pids=("${pids[@]:1}")
  names=("${names[@]:1}")
}

for exp_name in "${EXPERIMENTS[@]}"; do
  for run_idx in $(seq 1 "${REPEATS}"); do
    run_experiment "${exp_name}" "${run_idx}" &
    pids+=("$!")
    names+=("${exp_name}#${run_idx}")

    if [[ ${#pids[@]} -ge ${MAX_PARALLEL} ]]; then
      wait_for_one
    fi
  done
done

while [[ ${#pids[@]} -gt 0 ]]; do
  wait_for_one
done

failed=0
for item in "${statuses[@]}"; do
  name="${item%%:*}"
  status="${item##*:}"
  echo "[eval] result ${name}: ${status}"
  if [[ "${status}" != "0" ]]; then
    failed=1
  fi
done

exit "${failed}"
