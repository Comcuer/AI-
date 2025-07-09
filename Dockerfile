# 使用一个官方的、精简的Python 3.11镜像作为基础
FROM python:3.11-slim

# 设置容器内的工作目录
WORKDIR /app

# ★★★ 在这里，我们可以自由地安装系统工具 ★★★
# 先更新软件包列表，然后安装ffmpeg
RUN apt-get update && apt-get install -y ffmpeg

# 将我们的依赖清单文件复制到容器里
COPY requirements.txt .

# 在容器内，安装所有的Python库
RUN pip install --no-cache-dir -r requirements.txt

# 将我们项目的所有其他文件复制到容器里
COPY . .

# CMD指令会被Render的启动命令覆盖，但写上是一个好习惯
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]