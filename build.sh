#!/usr/bin/env bash
# exit on error
set -o errexit

# 在安装Python依赖前，先用apt-get命令安装ffmpeg
apt-get update && apt-get install -y ffmpeg