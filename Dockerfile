# 轻量镜像：只需 numpy / pandas / Pillow，浏览器在宿主机上开
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY chromterm/ ./chromterm/
COPY index.html .
COPY assets/ ./assets/
COPY models/ ./models/
COPY examples/ ./examples/
COPY app.py ./

# 容器里必须监听 0.0.0.0，宿主机映射端口后才能访问
ENV PYTHONUNBUFFERED=1
EXPOSE 8770
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "8770"]