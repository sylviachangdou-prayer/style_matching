# Hugging Face Space (Docker SDK) image for the StyleMatch API + static page.
# HF requires the Dockerfile at the Space repo root: copy this file there when
# pushing (see web/README.md). Build context must contain scripts/ and web/.
FROM python:3.12-slim

RUN useradd -m -u 1000 user
WORKDIR /app

COPY web/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY scripts ./scripts
COPY web ./web

USER user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    STYLEMATCH_INDEX_DIR=/home/user/index \
    STYLEMATCH_DEVICE=cpu

# Set STYLEMATCH_HUB_REPO (Space settings -> Variables) to the HF dataset repo
# holding artifacts/multilingual_style_index_v1; it is downloaded at startup.
EXPOSE 7860
CMD ["uvicorn", "web.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
