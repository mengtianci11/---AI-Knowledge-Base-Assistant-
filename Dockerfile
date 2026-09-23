# 企业知识库 AI 助手 - 生产镜像
# 构建：docker build -t kb-assistant .
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 先装依赖，利用 Docker 层缓存
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# 数据目录（挂载卷时覆盖）
VOLUME ["/app/data"]

CMD ["python", "企业知识库 AI 助手 - 综合毕业项目.py"]