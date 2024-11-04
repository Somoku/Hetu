conda activate hetu
source ../../hetu.exp

export HETU_SWITCH_ALGORITHM=NEW_GREEDY
export HETU_SWITCH_PROFILE=TIME
export HETU_INTERNAL_LOG_LEVEL=INFO
export HETU_STRAGGLER=ANALYSIS

export HETU_MEMORY_PROFILE=WARN
export HETU_MAX_SPLIT_SIZE_MB=200
export HETU_MAX_INTERNAL_FRAGMENT_SIZE_MB=20

export HETU_PARALLEL_ATTN_SPLIT_PATTERN=NORMAL

export NCCL_DEBUG=VERSION

echo "env done"