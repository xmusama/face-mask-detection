#!/usr/bin/env bash
set -euo pipefail

KERNEL_ID="xmusama/face-mask-from-scratch-cv-earlystop"
ACCELERATOR="${KAGGLE_ACCELERATOR:-NvidiaTeslaT4}"
TIMEOUT_SECONDS="${KAGGLE_TIMEOUT_SECONDS:-21600}"
POLL_SECONDS="${KAGGLE_POLL_SECONDS:-120}"
WORK_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "${WORK_DIR}"
}
trap cleanup EXIT

cp face_mask_from_scratch_cv_earlystop.ipynb "${WORK_DIR}/"
cp kernel-metadata-earlystop.json "${WORK_DIR}/kernel-metadata.json"
cp requirements.txt "${WORK_DIR}/"

echo "Pushing Kaggle kernel: ${KERNEL_ID}"
echo "Accelerator: ${ACCELERATOR}"
echo "Timeout: ${TIMEOUT_SECONDS}s"
kaggle kernels push -p "${WORK_DIR}" --accelerator "${ACCELERATOR}" --timeout "${TIMEOUT_SECONDS}"

echo
echo "Checking kernel status every ${POLL_SECONDS}s..."
while true; do
  STATUS_OUTPUT="$(kaggle kernels status "${KERNEL_ID}")"
  echo
  date -u '+%Y-%m-%d %H:%M:%S UTC'
  echo "${STATUS_OUTPUT}"

  if echo "${STATUS_OUTPUT}" | grep -qi 'complete'; then
    echo
    echo "Kernel complete. Downloading outputs to outputs_earlystop/ ..."
    mkdir -p outputs_earlystop
    kaggle kernels output "${KERNEL_ID}" -p outputs_earlystop
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
