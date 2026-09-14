# JustRL2 training image.
#
# Built on the community Miles image, which already contains everything the training
# stack needs: PyTorch + CUDA, TransformerEngine, apex, Megatron-LM
# (radixark/Megatron-LM @ miles-main, checked out at /root/Megatron-LM), SGLang
# (sglang-miles branch at /sgl-workspace/sglang), Megatron-Bridge (used by the
# HF -> torch_dist converter) and Ray. See https://github.com/radixark/miles/tree/main/docker
# for how that image is produced and which tags exist.
#
#   docker build -t justrl2 .
#   docker run --gpus all --ipc=host --network=host -it \
#       -v /path/to/models:/workspace/JustRL2/models \
#       -v /path/to/datasets:/workspace/JustRL2/datasets \
#       -v /path/to/runs:/workspace/JustRL2/runs justrl2
#
# DSpark speculative decoding is NOT available in this image (it needs an SGLang build
# that carries the DSpark scheduler); leave DSPARK_DRAFT_MODEL_PATH empty.
#
# Which base tag: pick by the *host driver*, since the tag decides the CUDA runtime.
#
#   radixark/miles:dev        CUDA 13.0.3  -> needs driver >= 580   (also has arm64)
#   radixark/miles:dev-cu12   CUDA 12.9.2  -> works on driver >= 525 (amd64 only)
#
# Both carry the same Megatron-LM (radixark/Megatron-LM @ miles-main) and sglang-miles,
# and both have flash-attn (FA2 + FA3), TransformerEngine 2.17 and apex prebuilt, so the
# recipe and its configs are identical either way. `dev-cu12` is Miles' own
# `--variant cu12-x86` build on lmsysorg/sglang:v0.5.19-cu129 — it is what to use when
# the host driver cannot go to 580, and it removes the need for the hand-rolled
# bare-metal stack in justrl2/setup/bare_metal_cu129.sh.
#
# Both `dev` and `dev-cu12` are rebuilt daily and MEGATRON_COMMIT is empty upstream (=
# branch HEAD at build time), so pin the dated tag for a run you intend to resume:
#   docker build --build-arg MILES_IMAGE=radixark/miles:dev-cu12-202609130149 -t justrl2 .

ARG MILES_IMAGE=radixark/miles:dev
FROM ${MILES_IMAGE}

WORKDIR /workspace/JustRL2
COPY . .

# train.sh expects the two frameworks next to the repo root.
RUN ln -sfn /root/Megatron-LM Megatron-LM && \
    ln -sfn /sgl-workspace/sglang sglang

# The image already satisfies requirements.txt; install only what the recipe adds.
RUN pip install --no-cache-dir -e . --no-deps && \
    pip install --no-cache-dir "math-verify==0.9.0" "antlr4-python3-runtime"

# Dependencies are baked in; setup.sh only applies the Megatron patch (idempotent).
ENV SKIP_PIP_INSTALL=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

CMD ["/bin/bash"]
