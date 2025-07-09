# AI 会议纪要系统

这是一个基于Python FastAPI、Whisper和Gemini API的智能会议纪要生成工具。

## 主要功能
- 支持上传任意时长的音频文件（mp3, m4a, wav等）。
- 自动将音频转写为文字稿。
- 自动为文字稿生成包含关键决策、行动项和摘要的智能纪要。
- 永久化存储历史记录，并支持下载和删除。

## 技术栈
- **后端:** FastAPI, Uvicorn
- **数据库:** SQLAlchemy, SQLite
- **AI模型:** OpenAI Whisper, Google Gemini
- **音频处理:** Pydub

## 本地运行指南

1.  **克隆仓库**
    ```bash
    git clone <仓库的URL>
    cd AI_Meeting_Minutes
    ```

2.  **创建并激活虚拟环境**
    ```bash
    # 创建
    python -m venv venv
    # 激活 (Windows)
    .\venv\Scripts\activate
    # 激活 (Mac/Linux)
    source venv/bin/activate
    ```

3.  **安装所有依赖库**
    ```bash
    pip install -r requirements.txt
    ```

4.  **配置API密钥**
    * 打开 `app.py` 文件。
    * 在文件顶部的配置区，填入您自己的`GEMINI_API_KEY`和`OPENAI_API_KEY`。

5.  **安装FFmpeg**
    * 本项目处理长音频依赖FFmpeg。请确保您的操作系统已安装FFmpeg，并已将其添加到系统环境变量(PATH)中。

6.  **启动服务器**
    ```bash
    uvicorn app:app --reload
    ```

7.  在浏览器中打开 `http://127.0.0.1:8000` 访问应用。