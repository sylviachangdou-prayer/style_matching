# Hugging Face Space（Docker SDK）镜像：同时提供 StyleMatch API 和静态页面。
# HF 要求 Dockerfile 位于 Space 仓库根目录，scripts/push-space.sh 会把它放过去。
FROM python:3.12-slim

RUN useradd -m -u 1000 user
WORKDIR /app

# 只装 CPU 版 torch，体积小很多。
COPY web/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

USER user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    STYLEMATCH_INDEX_DIR=/home/user/index \
    STYLEMATCH_DEVICE=cpu \
    PYTHONUNBUFFERED=1

# 构建时把两个编码器下进镜像层。
#
# 免费 Space 的磁盘是临时的：容器重启后镜像层还在，家目录的写入会丢。
# 所以权重必须在构建时进镜像，否则每次休眠唤醒都要重下约 3.3 GB，
# 第一个访客要等几分钟；进了镜像之后唤醒只剩把模型读进内存，约 30–60 秒。
#
# 用 SentenceTransformer 而不是 snapshot_download，是为了只下运行时真正会用到的
# 文件（e5-base 仓库里还有 onnx / openvino 变体，全下会多几个 GB）。
ARG STYLEMATCH_MODEL_REPO=sylviachangdou-prayer/stylematch-authorship
ARG STYLEMATCH_MODEL_REVISION=v2
ARG STYLEMATCH_TOPIC_MODEL=intfloat/multilingual-e5-base
RUN STYLE_REPO="$STYLEMATCH_MODEL_REPO" STYLE_REVISION="$STYLEMATCH_MODEL_REVISION" TOPIC_REPO="$STYLEMATCH_TOPIC_MODEL" \
    python -c "import os; from sentence_transformers import SentenceTransformer; \
SentenceTransformer(os.environ['STYLE_REPO'], revision=os.environ['STYLE_REVISION']); \
SentenceTransformer(os.environ['TOPIC_REPO'])"

# 代码放在权重之后，改代码不会让上面那个大层失效。
COPY --chown=user:user scripts ./scripts
COPY --chown=user:user web ./web

# 索引只有 12 MB，留在启动时下载，这样换索引不用重新构建镜像。
# 需要在 Space 设置里给 STYLEMATCH_HUB_REPO。
EXPOSE 7860
CMD ["uvicorn", "web.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
