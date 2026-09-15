# syntax=docker/dockerfile:1
# Build context: Git archives and verified runtime supplements produced by build.py.
# The pinned base supplies Python, CUDA, PyTorch, native libraries, and package metadata.
# Source-specific cache paths isolate kernels compiled from different Git revisions.
ARG BASE_IMAGE=voipmonitor/vllm@sha256:a298fe1cd207eaf97bd2ff2686716ed25b7009c09b36650eba732a4a7dc51512
FROM ${BASE_IMAGE}

ARG BASE_IMAGE
ARG BASE_IMAGE_DIGEST
ARG SOURCE_HASH
ARG MATERIALIZATION_MANIFEST_SHA256
ARG IMAGE_TAG
ARG B12X_COMMIT
ARG B12X_TREE
ARG B12X_UPSTREAM_BASE
ARG VLLM_COMMIT
ARG VLLM_TREE
ARG VLLM_UPSTREAM_BASE
ARG LAUNCHER_SHA256

RUN test -n "${SOURCE_HASH}" && test -n "${B12X_COMMIT}" && test -n "${VLLM_COMMIT}" \
    && rm -rf /opt/glm53-flash/vllm /opt/glm53-flash/b12x
COPY b12x /opt/glm53-flash/b12x
COPY vllm /opt/glm53-flash/vllm
COPY materialization-manifest.json /opt/glm53-flash/qualification/materialization-manifest.json
COPY runtime-files.sha256 /opt/glm53-flash/qualification/runtime-files.sha256

ENV LOCAL_INFERENCE_CACHE_FINGERPRINT=${SOURCE_HASH} \
    XDG_CACHE_HOME=/cache/jit/${SOURCE_HASH} \
    VLLM_CACHE_ROOT=/cache/jit/${SOURCE_HASH}/vllm \
    VLLM_CACHE_DIR=/cache/jit/${SOURCE_HASH}/vllm \
    TRITON_CACHE_DIR=/cache/jit/${SOURCE_HASH}/triton \
    TORCH_EXTENSIONS_DIR=/cache/jit/${SOURCE_HASH}/torch-extensions \
    TORCHINDUCTOR_CACHE_DIR=/cache/jit/${SOURCE_HASH}/torchinductor \
    VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/cache/jit/${SOURCE_HASH}/flashinfer-autotune \
    FLASHINFER_WORKSPACE_BASE=/cache/jit/${SOURCE_HASH}/flashinfer \
    TVM_FFI_CACHE_DIR=/cache/jit/${SOURCE_HASH}/tvm-ffi \
    TVM_CACHE_DIR=/cache/jit/${SOURCE_HASH}/tvm \
    TILELANG_CACHE_DIR=/cache/jit/${SOURCE_HASH}/tilelang \
    CUTE_DSL_CACHE_DIR=/cache/jit/${SOURCE_HASH}/cute-dsl \
    B12X_CUTE_COMPILE_CACHE_DIR=/cache/jit/${SOURCE_HASH}/b12x/cute \
    B12X_COMPILE_CACHE_DIR=/cache/jit/${SOURCE_HASH}/b12x/compile \
    SPARKINFER_COMPILE_CACHE_DIR=/cache/jit/${SOURCE_HASH}/b12x/compile \
    DG_JIT_CACHE_DIR=/cache/jit/${SOURCE_HASH}/deep-gemm \
    MM_SPARSE_ATTN_AOT_CACHE=/cache/jit/${SOURCE_HASH}/minfer/mm-sparse-attn \
    MINFER_FMHA_CACHE_DIR=/cache/jit/${SOURCE_HASH}/minfer/fmha-sm120 \
    NUMBA_CACHE_DIR=/cache/jit/${SOURCE_HASH}/numba \
    CUDA_CACHE_PATH=/cache/jit/${SOURCE_HASH}/cuda \
    CUPY_CACHE_DIR=/cache/jit/${SOURCE_HASH}/cupy \
    TMPDIR=/container-tmp

# Override source and qualification labels inherited from the runtime foundation.
LABEL io.joninco.native-libraries.source-image="sha256:a298fe1cd207eaf97bd2ff2686716ed25b7009c09b36650eba732a4a7dc51512" \
      io.joninco.native-libraries.status="Runtime supplements verified against sources.lock.json; native input changes rejected" \
      io.local-inference-lab.port.b12x-commit="${B12X_COMMIT}" \
      io.local-inference-lab.port.b12x-pull-request="" \
      io.local-inference-lab.port.materialization-manifest="/opt/glm53-flash/qualification/materialization-manifest.json" \
      io.local-inference-lab.port.materialization-manifest-sha256="${MATERIALIZATION_MANIFEST_SHA256}" \
      io.local-inference-lab.port.source-key="${SOURCE_HASH}" \
      io.local-inference-lab.port.vllm-commit="${VLLM_COMMIT}" \
      io.local-inference-lab.port.vllm-pull-request="" \
      local-inference.b12x.base-revision="${B12X_UPSTREAM_BASE}" \
      local-inference.b12x.branch="master" \
      local-inference.b12x.commit="${B12X_COMMIT}" \
      local-inference.b12x.composition="Committed joninco/b12x master source tree" \
      local-inference.b12x.foundation.commit="${B12X_UPSTREAM_BASE}" \
      local-inference.b12x.integration-prs="none: committed joninco fork source" \
      local-inference.b12x.integration.lock_sha256="none" \
      local-inference.b12x.integration.prs="none: committed joninco fork source" \
      local-inference.b12x.integration.tree="${B12X_TREE}" \
      local-inference.b12x.merge-heads="${B12X_COMMIT}" \
      local-inference.b12x.package-tree="${B12X_TREE}" \
      local-inference.b12x.patch_sha256="none" \
      local-inference.b12x.repo="https://github.com/joninco/b12x.git" \
      local-inference.b12x.source-revision="${B12X_COMMIT}" \
      local-inference.b12x.tree="${B12X_TREE}" \
      local-inference.b12x.upstream-base="${B12X_UPSTREAM_BASE}" \
      local-inference.b12x.upstream-revision="${B12X_UPSTREAM_BASE}" \
      local-inference.cache.fingerprint="${SOURCE_HASH}" \
      local-inference.qualification.hardware="research-only; no serving qualification embedded" \
      local-inference.qualification.parallelism="research-only; no serving qualification embedded" \
      local-inference.qualification.prefill="research-only; no serving qualification embedded" \
      local-inference.qualification.status="research-only" \
      local-inference.runtime.cache-fingerprint="${SOURCE_HASH}" \
      local-inference.runtime.launcher.sha256="${LAUNCHER_SHA256}" \
      local-inference.runtime.source-lock.path="/opt/glm53-flash/qualification/materialization-manifest.json" \
      local-inference.runtime.source-lock.sha256="${MATERIALIZATION_MANIFEST_SHA256}" \
      local-inference.runtime.source-mode="committed-fork-trees-with-verified-r27-native-supplements" \
      local-inference.runtime.vllm.composition.commit="${VLLM_COMMIT}" \
      local-inference.runtime.vllm.port.commit="${VLLM_COMMIT}" \
      local-inference.status="research-only; source verification is not serving qualification" \
      local-inference.vllm.base-revision="${VLLM_UPSTREAM_BASE}" \
      local-inference.vllm.branch="dev/jovian-judgement" \
      local-inference.vllm.commit="${VLLM_COMMIT}" \
      local-inference.vllm.composition="Committed joninco/vllm dev/jovian-judgement source tree" \
      local-inference.vllm.foundation.commit="${VLLM_UPSTREAM_BASE}" \
      local-inference.vllm.integration-prs="none: committed joninco fork source" \
      local-inference.vllm.integration.lock_sha256="none" \
      local-inference.vllm.integration.prs="none: committed joninco fork source" \
      local-inference.vllm.integration.tree="${VLLM_TREE}" \
      local-inference.vllm.merge-heads="${VLLM_COMMIT}" \
      local-inference.vllm.package-tree="${VLLM_TREE}" \
      local-inference.vllm.package-version="committed branch tree ${VLLM_TREE}" \
      local-inference.vllm.patch_sha256="none" \
      local-inference.vllm.repo="https://github.com/joninco/vllm.git" \
      local-inference.vllm.source-revision="${VLLM_COMMIT}" \
      local-inference.vllm.tree="${VLLM_TREE}" \
      local-inference.vllm.upstream-base="${VLLM_UPSTREAM_BASE}" \
      local-inference.vllm.upstream-revision="${VLLM_UPSTREAM_BASE}" \
      org.opencontainers.image.base.digest="${BASE_IMAGE_DIGEST}" \
      org.opencontainers.image.base.name="${BASE_IMAGE}" \
      org.opencontainers.image.description="Committed joninco sources with 64-row DCP and expert output-tile support; native libraries and dependencies preserved from the r27 runtime." \
      org.opencontainers.image.revision="${VLLM_COMMIT}" \
      org.opencontainers.image.source="https://github.com/joninco/blackwell-llm-docker/tree/main/glm53" \
      org.opencontainers.image.title="GLM-5.3-NVFP4 joninco fork sources on the r27 runtime" \
      org.opencontainers.image.version="${IMAGE_TAG}"

ENTRYPOINT ["/opt/glm53-flash/vllm/serve-glm53.sh"]
