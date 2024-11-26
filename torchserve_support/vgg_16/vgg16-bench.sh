#! /bin/bash

# Generate curl requests of varying batch sizes to be sent to the inference server
# get images from a common vgg16 cat image store.
IMAGE_DIR="../image_store/cat_dog_vgg16/cats"
SERVER="http://10.129.28.180:8080"
MODEL="vgg16"

BATCH_SIZE=$1

# Check CLA
if [ -z "$BATCH_SIZE" ]; then
  echo "Usage: $0 <batch_size>"
  exit 1
fi

IMAGES=($(ls $IMAGE_DIR | head -n $BATCH_SIZE))

send_req_batch() {
  BATCH_DATA=""
  for IMAGE in "${IMAGES[@]}"; do
    BATCH_DATA+=" -F data=@${IMAGE_DIR}/${IMAGE}"
  done

  START_TIME=$(date +%s%N)
  eval "curl -X POST \"$SERVER/predictions/$MODEL\" $BATCH_DATA >/dev/null 2>&1"
  END_TIME=$(date +%s%N)

  ELAPSED_TIME=$((($END_TIME - $START_TIME) / 1000000))  # Convert ns to ms
  echo "$ELAPSED_TIME"
}

ITERATIONS=100

TOTAL_TIME=0
for ((i=1; i<=ITERATIONS; i++)); do
  ITER_TIME=$(send_req_batch)
  TOTAL_TIME=$((TOTAL_TIME + ITER_TIME))
done

AVERAGE_TIME=$((TOTAL_TIME / ITERATIONS))
echo "Average execution time over $ITERATIONS iterations for batch size $BATCH_SIZE: $AVERAGE_TIME ms"
