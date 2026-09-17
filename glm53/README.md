# GLM-5.3-NVFP4 from the joninco forks

Status: **implemented**. The published image passed DCP=1/MTP=3 startup and
streamed-generation smoke checks on eight RTX PRO 6000 Blackwell Max-Q GPUs.
Performance, numerical correctness, and DCP=4 acceptance remain **research-only**.

This recipe builds committed source from
[joninco/b12x, branch `master`](https://github.com/joninco/b12x/tree/master) and
[joninco/vllm, branch `dev/jovian-judgement`](https://github.com/joninco/vllm/tree/dev/jovian-judgement).
It preserves the CUDA 13.3 runtime and compiled libraries from an immutable
Jovian Judgement r27 foundation. Model weights are external to the image.

## Pull the published image

```bash
docker pull joninco/vllm:glm53-b12x686450b7-vllm5b28d30b59-r27
```

The registry digest identifies the exact published artifact:

```bash
docker pull joninco/vllm@sha256:763f2bddf5d628d4f14267b44d646fb84d9251065b552132d7c38865ebccd037
```

| Source | Commit | Git tree |
| --- | --- | --- |
| b12x `master` | `686450b72a5665ebeeb571ed5c1729b190a829da` | `e0a9f13abd34fcf0ce560ce21c637b02a368d068` |
| vLLM `dev/jovian-judgement` | `5b28d30b59b0973743ab0a171eee10fce72ddf3f` | `3b6790fd16c6a4949b6924be820cd0cedaae0065` |

The vLLM source declares the draft model's checkpoint name prefixes, so the
loader opens only the shard that holds the multi-token-prediction layer and
reads the checkpoint once at startup. The smoke-checked launch logged
`Safetensors index filter selected 1/85 checkpoint shards` and a 1.1 s
draft-weight pass after the 23.7 s target-weight pass.

## Build the pinned sources

Requirements: Linux x86-64, Git, Docker with BuildKit and the `runc` runtime,
and `uv` with Python 3.12. Building and checking source identity require no GPU
or model checkpoint. The build downloads repository objects and, when absent
locally, the pinned runtime image.

```bash
git clone https://github.com/joninco/blackwell-llm-docker.git
cd blackwell-llm-docker
uv run --no-project --python 3.12 glm53/build.py
```

The default tag is
`joninco/vllm:glm53-b12x686450b7-vllm5b28d30b59-r27-local`.
The default output directory, `glm53/build/`, must not already exist. Use
`--output` with another absent directory for a separate build; use `--tag`
to choose a local image name. The script builds and verifies locally. Registry
publication is a separate `docker push` operation.

[sources.lock.json](sources.lock.json) pins both source revisions, the runtime
digest, the native supplement inventory, and the published runtime-file digest.
The builder:

1. Fetches the specified Git commits into isolated checkouts.
2. Rejects native or dependency input changes against the compatible source
   revisions recorded in the lock file.
3. Extracts 112 compiled/generated files from the pinned runtime, checking
   every file against [native-vllm.sha256](native-vllm.sha256).
4. Exports committed Git trees into a Docker context. Native supplements fill
   only paths absent from the vLLM source; committed files take precedence.
5. Builds the source layer without installing or upgrading dependencies.
6. Verifies every source/native file in the image and checks CPU imports.

The output directory contains `resolved-sources.json`, `context/`,
`build-args.json`, `build.log`, `verification.json`, and `BUILD.json`.
The image embeds its source manifest and runtime inventory under
`/opt/glm53-flash/qualification/`.

The pinned rebuild must match all 8,096 source/native files in the published
image, including ten shared libraries. Its source fingerprint is
`a7a133d8ada9a211b476f213251cf7077e931324a4fcf4b5376624cd536c3877`;
its complete runtime inventory digest is
`adca04e511333f232d6cf99ad644921f30ec1f1a4252ff31e8cbecbe47445042`.
Image digests can differ because recipe metadata, build provenance, and layer
packaging differ. Pull by registry digest to obtain the exact published image.

## Build the branch heads

```bash
uv run --no-project --python 3.12 glm53/build.py \
  --branch-heads \
  --output ../glm53-branch-build
```

This resolves `joninco/b12x:master` and
`joninco/vllm:dev/jovian-judgement` to commit IDs before exporting their source.
It records both IDs and derives a local tag from them. No upstream merge,
cherry-pick, or source patch is performed by the recipe.

Native reuse is deliberately limited to revisions whose changes from the
compatible commits are Python files under the package, `tests`, `benchmarks`,
or `examples` directories. Changes to C++/CUDA/Rust, build scripts, requirements,
or other files cause the build to stop. A different native runtime requires
a separately built and validated foundation and an updated lock/inventory.
Passing this input check does not establish compatibility of arbitrary Python
changes with native interfaces; branch-head images still need serving checks.

The compatible source references are vLLM
`8d66b502b7b4f58e505c6de2a5ea9857b4b7715c` and b12x
`77f9a59814551693447b19f9b12b4ec28d5e3c1b`. They identify source that served
successfully with these native libraries. They are not claims that the
foundation's binaries were compiled from those exact repository trees.

## Runtime foundation

The [Dockerfile](Dockerfile) uses:

```text
voipmonitor/vllm@sha256:a298fe1cd207eaf97bd2ff2686716ed25b7009c09b36650eba732a4a7dc51512
```

The foundation supplies PyTorch 2.13.0, CUDA 13.3,
Triton 3.7.1+gitf797708c.nv26.7, CUTLASS DSL 4.6.2, FlashInfer 0.6.18+cu133,
and vLLM's compiled native extensions. Changed CuTe kernels compile at serving
startup or runtime. Compilation cache paths include the source fingerprint.

This is a source-layer build on a precompiled foundation. It does not compile
the entire CUDA/PyTorch/vLLM native stack from scratch. Package distribution
metadata and generated version files retain foundation versions; commit
labels, the manifest, and file hashes establish source identity.

## Serve with DCP=1 and MTP=3

[compose.json](compose.json) preserves the smoke-tested serving settings and
parameterizes the host model directory, image, and port. It requires eight
96 GB Blackwell GPUs, a compatible NVIDIA driver and Container Toolkit, and
a complete local GLM-5.3-NVFP4 checkpoint with `config.json`.

From the repository root:

```bash
export GLM53_MODEL_DIR=/absolute/path/to/GLM-5.3-NVFP4
docker compose -f glm53/compose.json up -d
curl -fsS http://127.0.0.1:8000/health
```

The default image is the published digest. To use a local rebuild, set
`GLM53_IMAGE` to its tag before running Compose. Set `GLM53_PORT` to change
the host-network listening port. Compose creates separate compilation and
temporary-data volumes. The configuration uses privileged containers, host
networking, and host IPC for the PCIe serving setup.

The launch uses TP=8, DCP=1, three speculative tokens, FP8 KV storage,
64 maximum sequences, an 8,192-token scheduling budget, a 524,288-token model
limit, and `FULL_AND_PIECEWISE` CUDA graphs with a 256-row capture ceiling.
Its fused all-reduce ceiling is 192 KiB, corresponding to 16 BF16 rows at
hidden size 6,144. The source contains 64-row support, but this launch retains
the 16-row fused-reduction ceiling; DCP transport is inactive at DCP=1.

## Verification scope

[VALIDATION.json](VALIDATION.json) records image identities, recipe source
verification, and the published image's functional smoke-check summary.
The published image completed 30 streamed requests and 15,360 generated tokens:
short prompts at concurrency 1, 4, 8, and 16, plus one 30,989-token prompt.
Every request generated 512 tokens, and each batch recorded decode activity
and accepted speculative tokens. No preemptions, container restarts, or
matching server fault lines were recorded during those checks.

The configured 524,288-token context limit was not exercised. These results
establish startup and streamed decoding for the stated configuration; they
do not establish throughput gains, numerical equivalence, or DCP=4 adoption.
