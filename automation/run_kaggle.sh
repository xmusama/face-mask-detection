#!/usr/bin/env bash
set -euo pipefail

KERNEL_ID="tristianyosa/face-mask-from-scratch-cv"
ACCELERATOR="${KAGGLE_ACCELERATOR:-NvidiaTeslaT4}"
TIMEOUT_SECONDS="${KAGGLE_TIMEOUT_SECONDS:-43200}"
POLL_SECONDS="${KAGGLE_POLL_SECONDS:-120}"
RESULTS_DIR="${KAGGLE_RESULTS_DIR:-results}"

echo "Pushing Kaggle kernel: ${KERNEL_ID}"
echo "Accelerator: ${ACCELERATOR}"
echo "Timeout: ${TIMEOUT_SECONDS}s"
kaggle kernels push -p . --accelerator "${ACCELERATOR}" --timeout "${TIMEOUT_SECONDS}"

echo
echo "Checking kernel status every ${POLL_SECONDS}s..."
while true; do
  STATUS_OUTPUT="$(kaggle kernels status "${KERNEL_ID}")"
  echo
  date -u '+%Y-%m-%d %H:%M:%S UTC'
  echo "${STATUS_OUTPUT}"

  if echo "${STATUS_OUTPUT}" | grep -qi 'complete'; then
    echo
    echo "Kernel complete. Downloading outputs to ${RESULTS_DIR}/ ..."
    mkdir -p "${RESULTS_DIR}/figures" "${RESULTS_DIR}/metrics" "${RESULTS_DIR}/models"
    kaggle kernels output "${KERNEL_ID}" -p "${RESULTS_DIR}"

    find "${RESULTS_DIR}" -maxdepth 1 -type f -name 'metrics.json' -exec mv -f {} "${RESULTS_DIR}/metrics/" \;
    find "${RESULTS_DIR}" -maxdepth 1 -type f -name '*.png' -exec mv -f {} "${RESULTS_DIR}/figures/" \;
    find "${RESULTS_DIR}" -maxdepth 1 -type f \( -name '*.keras' -o -name '*.h5' -o -name '*.tflite' \) -exec mv -f {} "${RESULTS_DIR}/models/" \;
    find "${RESULTS_DIR}" -maxdepth 1 -type f -name '*.log' -delete
    echo "Done."
    break
  fi

  if echo "${STATUS_OUTPUT}" | grep -Eiq 'error|failed|canceled|cancelled'; then
    echo
    echo "Kernel did not complete successfully. Check Kaggle logs for details."
    exit 1
  fi

  sleep "${POLL_SECONDS}"
done

echo
echo "Final status:"
kaggle kernels status "${KERNEL_ID}"
