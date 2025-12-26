MODEL_NAME=omniASR_LLM_300M_v2
BASE_TAG=cu126-pt280
# Remove "omniASR_" prefix, lowercase and replace '_' with '-'
TAG_SUFFIX=$(echo $MODEL_NAME | sed 's/^omniASR_//' | tr 'A-Z_' 'a-z-')

docker buildx build \
    --platform linux/amd64 \
    --build-arg MODEL_NAME=$MODEL_NAME \
    -t omniasr-server:$BASE_TAG-$TAG_SUFFIX .
