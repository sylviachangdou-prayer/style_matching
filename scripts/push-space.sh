#!/usr/bin/env bash
# 把 StyleMatch 推到 Hugging Face Space（Docker SDK）。
#
# 只打包运行时需要的东西：Dockerfile、README.md（带 HF 配置头）、web/、scripts/。
# 排除 web/pic（前端没引用，只是素材）、tests 和 __pycache__。
# web/static 里的图片走 Git LFS——HF 拒绝不走 Xet/LFS 的二进制文件。
#
# 用法：bash scripts/push-space.sh
# 首次会要账号密码：用户名填 HF 用户名，密码填 Write 类型的 Access Token
# （https://huggingface.co/settings/tokens）。
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_root"

space_url="$(git remote get-url space 2>/dev/null || true)"
if [ -z "$space_url" ]; then
  echo "还没配置 space 远端。先运行：" >&2
  echo "  git remote add space https://huggingface.co/spaces/你的用户名/stylematch" >&2
  exit 1
fi

if ! command -v git-lfs >/dev/null 2>&1; then
  echo "需要 git-lfs：brew install git-lfs" >&2
  exit 1
fi

staging="$(mktemp -d)"
trap 'rm -rf "$staging"' EXIT

echo "==> 打包"
mkdir -p "$staging"
cp web/Dockerfile "$staging/Dockerfile"          # HF 要求 Dockerfile 在根目录
rsync -a --exclude='__pycache__' --exclude='.DS_Store' scripts "$staging/"
rsync -a --exclude='__pycache__' --exclude='.DS_Store' \
      --exclude='pic' --exclude='tests' web "$staging/"

# Space 的配置写在根 README.md 的头部，HF 从这里读 SDK 和端口。
cat > "$staging/README.md" <<'MD'
---
title: StyleMatch
emoji: 🖋️
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Open-set author style retrieval with style and topic scores
---

# StyleMatch

给一段文字，返回风格最接近的作者档案，风格分与主题分分开给出，并允许「没有强匹配」。

源码与研究记录：<https://github.com/sylviachangdou-prayer>
MD

cd "$staging"
git init -q -b main
git lfs install --local >/dev/null
git lfs track "*.jpg" "*.jpeg" "*.png" "*.webp" "*.gif" "*.svg" "*.woff" "*.woff2" >/dev/null
git add .gitattributes
git add -A
git -c user.name="${GIT_AUTHOR_NAME:-deploy}" \
    -c user.email="${GIT_AUTHOR_EMAIL:-deploy@local}" \
    commit -q -m "Deploy StyleMatch $(git -C "$project_root" rev-parse --short HEAD)"

echo "==> 推送到 $space_url"
git push --force "$space_url" main

echo "==> 完成。去 Space 页面看 Building 进度（第一次要下 3.3 GB 权重，约 10–20 分钟）。"
