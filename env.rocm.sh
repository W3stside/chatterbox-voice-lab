# ROCm / gfx1100 (RX 7900 XTX) performance env for Chatterbox.
# Source before cli.py or the server:   source env.rocm.sh
# Safe on non-ROCm hosts (vars just sit unused). From the perf research workflow.

export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"

# MIOpen: persist tuned conv solvers so the GemmFwdRest workspace=0 naive path isn't
# re-searched every restart. Run ONCE with MIOPEN_FIND_MODE=NORMAL to populate the DB,
# then the default FAST(2) reuses it. (May be partly ignored on torch+rocm — verify by
# grepping 'GemmFwdRest' after a synth; want 0.)
export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-2}"
export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-$HOME/.config/miopen}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_CUSTOM_CACHE_DIR:-$HOME/.config/miopen}"
mkdir -p "$MIOPEN_USER_DB_PATH"

# AOTriton flash-attention SDPA on RDNA3 (engages with the t3 sdpa patch; harmless before).
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export FLASH_ATTENTION_TRITON_AMD_AUTOTUNE=FALSE

# Allocator: smooth fragmentation/realloc hitches (helps first-audio jitter).
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"

# GEMM: rocBLAS is the safe default on torch 2.6 for gfx1100 (hipBLASLt path is flaky).
export TORCH_BLAS_PREFER_HIPBLASLT=0

# --- optional / deploy-time only (leave commented for the bench) ---
# TunableOp: tune ONCE offline (TUNING=1), then ship read-only (TUNING=0). ~+20% on T3 GEMMs.
# export PYTORCH_TUNABLEOP_ENABLED=1
# export PYTORCH_TUNABLEOP_TUNING=0
# export PYTORCH_TUNABLEOP_FILENAME="$HOME/.cache/supra_tunableop_%d.csv"
# Quiet MIOpen warning spam ONLY after confirming the slow conv path is actually gone:
# export MIOPEN_LOG_LEVEL=3

# Do NOT set HSA_XNACK (Strix-Halo APU only). Do NOT upgrade torch past 2.6 for speed
# (2.7+ has a gfx1100 hipBLASLt mm/bmm regression — pytorch#150155).
