# app.py - 100%完整最终版 (包含分块上传和后台任务)

# ======================= 1. 导入所有需要的工具 =======================
import os
import shutil
import uuid
import time
from contextlib import asynccontextmanager
from math import ceil
from pydub import AudioSegment

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Request, Form, status
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, PlainTextResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.background import BackgroundTasks

from sqlalchemy import create_engine, Column, Integer, String, DateTime, select
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import func

import google.generativeai as genai
from openai import OpenAI

# ======================= 2. 所有配置项 =======================
DATABASE_URL = "sqlite:///./meetings_archive.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "在服务器环境变量中设置")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "在服务器环境变量中设置")
# 从服务器环境变量读取代理地址，如果不存在，则此变量为None
PROXY_URL = os.getenv("PROXY_URL")
# ★★★ 新增：创建并配置我们的“门卫” ★★★
# get_remote_address 会获取访问者的IP地址
# default_limits 是我们的默认规则，但我们稍后会为具体接口单独设置
limiter = Limiter(key_func=get_remote_address)

if PROXY_URL:
    os.environ['HTTPS_PROXY'] = PROXY_URL
    os.environ['HTTP_PROXY'] = PROXY_URL

# ======================= 3. 数据库模型区 =======================
class Recording(Base):
    __tablename__ = "recordings"
    id = Column(Integer, primary_key=True, index=True)
    original_filename = Column(String, index=True)
    storage_path = Column(String, unique=True)
    transcript = Column(String, default="处理中...")
    summary = Column(String, default="处理中...")
    upload_time = Column(DateTime(timezone=True), server_default=func.now())
    status = Column(String, default="上传中...")

# ======================= 4. FastAPI 应用主体 =======================


# ★★★ 新增：我们自己的、格式100%正确的错误处理器 ★★★
async def rate_limit_exception_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": f"您访问太频繁了，请稍后重试。限制：{exc.detail}"}
    )
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("应用启动...")
    Base.metadata.create_all(bind=engine)
    if not os.path.exists('uploads'): os.makedirs('uploads')
    if not os.path.exists('temp_chunks'): os.makedirs('temp_chunks')
    print("数据库表和文件夹检查完成。")
    yield

app = FastAPI(title="AI 会议纪要系统 (终极版)", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
# ★★★ 新增：将“门卫”和它的“规则手册”注册到我们的应用中 ★★★
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exception_handler)# type: ignore
templates = Jinja2Templates(directory="templates")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 后台AI处理任务
def long_running_ai_task(db_path: str, recording_id: int):
    engine = create_engine(db_path, connect_args={"check_same_thread": False})
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = Session()
    try:
        print(f"后台任务开始：处理记录ID {recording_id}")
        recording = db.get(Recording, recording_id) # type: ignore
        if not recording or not recording.storage_path: return # type: ignore

        client = OpenAI(api_key=OPENAI_API_KEY, timeout=600.0)
        transcript_text = transcribe_large_audio(client, recording.storage_path) # type: ignore
        
        genai.configure(api_key=GEMINI_API_KEY) # type: ignore
        prompt = f"请为以下会议文字稿生成一份包含关键决策、行动项和智能摘要的会议纪要：\n\"\"\"\n{transcript_text}\n\"\"\""
        model = genai.GenerativeModel('gemini-1.5-flash-latest') # type: ignore
        response = model.generate_content(prompt)
        summary_text = response.text
        
        recording.transcript = transcript_text # type: ignore
        recording.summary = summary_text # type: ignore
        recording.status = "已完成" # type: ignore
        db.commit()
        print(f"后台任务完成：记录ID {recording_id} 已更新。")
    except Exception as e:
        recording = db.get(Recording, recording_id)
        if recording:
            recording.status = f"处理失败: {str(e)[:200]}" # type: ignore
            db.commit()
        print(f"后台任务出错：处理记录ID {recording_id} 时发生错误: {e}")
    finally:
        db.close()

# 音频切分函数
def transcribe_large_audio(client: OpenAI, file_path: str) -> str:
    audio = AudioSegment.from_file(file_path)
    max_size = 23 * 1024 * 1024
    if os.path.getsize(file_path) < max_size:
        with open(file_path, "rb") as f:
            return client.audio.transcriptions.create(model="whisper-1", file=f, response_format="text").strip()
    chunk_length_ms = 10 * 60 * 1000
    full_transcript = []
    total_chunks = ceil(len(audio) / chunk_length_ms)
    for i, start_ms in enumerate(range(0, len(audio), chunk_length_ms)):
        chunk = audio[start_ms : start_ms + chunk_length_ms]
        chunk_file_path = f"./temp_chunks/chunk_{i}.mp3"
        print(f"正在处理分块 {i+1}/{total_chunks}...")
        chunk.export(chunk_file_path, format="mp3")
        with open(chunk_file_path, "rb") as f:
            transcript_part = client.audio.transcriptions.create(model="whisper-1", file=f, response_format="text").strip()
            full_transcript.append(transcript_part)
        os.remove(chunk_file_path)
    if os.path.exists('temp_chunks'): shutil.rmtree('temp_chunks'); os.makedirs('temp_chunks')
    return " ".join(full_transcript)

# ======================= 5. API路由/应用逻辑区 =======================
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request, db = Depends(get_db)):
    recordings = db.execute(select(Recording).order_by(Recording.upload_time.desc())).scalars().all()
    return templates.TemplateResponse("index.html", {"request": request, "recordings": recordings})

# 新的上传分块接口
@app.post("/upload-chunk/")
@limiter.limit("100/minute") # 为分块上传设置一个宽松的限制
def upload_chunk(request: Request, chunk: UploadFile = File(...), identifier: str = Form(...), chunk_index: str = Form(...)):
    temp_dir = f"./temp_chunks/{identifier}"
    os.makedirs(temp_dir, exist_ok=True)
    chunk_path = os.path.join(temp_dir, chunk_index)
    with open(chunk_path, "wb") as buffer:
        shutil.copyfileobj(chunk.file, buffer)
    return JSONResponse(content={"detail": f"Chunk {chunk_index} for {identifier} received."})

# 新的合并与处理接口
@app.post("/process-file/")
@limiter.limit("5/hour") # 核心：为这个最耗钱的接口，设置严格的速率限制
def process_file(
    request: Request, # 新增request参数
    background_tasks: BackgroundTasks,
    identifier: str = Form(...),
    original_filename: str = Form(...),
    db = Depends(get_db)
):
    temp_dir = f"./temp_chunks/{identifier}"
    unique_id = str(uuid.uuid4())
    file_extension = os.path.splitext(original_filename)[1]
    final_filename = f"{unique_id}{file_extension}"
    final_path = f"./uploads/{final_filename}"
    try:
        chunk_files = sorted(os.listdir(temp_dir), key=int)
        with open(final_path, "wb") as final_file:
            for chunk_filename in chunk_files:
                with open(os.path.join(temp_dir, chunk_filename), "rb") as chunk_file:
                    final_file.write(chunk_file.read())
        shutil.rmtree(temp_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件分块合并失败: {e}")
    new_recording = Recording(original_filename=original_filename, storage_path=final_path, status="处理中...")
    db.add(new_recording)
    db.commit()
    db.refresh(new_recording)
    background_tasks.add_task(long_running_ai_task, DATABASE_URL, new_recording.id) # type: ignore
    return JSONResponse(content={"detail": "文件上传成功，已提交至后台处理。", "id": new_recording.id})

# ... 下载和删除功能的路由保持不变 ...
@app.get("/download/audio/{recording_id}")
def download_audio(recording_id: int, db = Depends(get_db)):
    recording = db.get(Recording, recording_id)
    if not recording: raise HTTPException(status_code=404, detail="记录未找到")
    return FileResponse(path=recording.storage_path, filename=recording.original_filename)

@app.get("/download/transcript/{recording_id}")
def download_transcript(recording_id: int, db = Depends(get_db)):
    recording = db.get(Recording, recording_id)
    if not recording: raise HTTPException(status_code=404, detail="记录未找到")
    return PlainTextResponse(content=recording.transcript, headers={"Content-Disposition": f"attachment; filename=transcript_{recording_id}.txt"})

@app.get("/download/summary/{recording_id}")
def download_summary(recording_id: int, db = Depends(get_db)):
    recording = db.get(Recording, recording_id)
    if not recording: raise HTTPException(status_code=404, detail="记录未找到")
    return PlainTextResponse(content=recording.summary, headers={"Content-Disposition": f"attachment; filename=summary_{recording_id}.txt"})

@app.post("/delete/{recording_id}")
def delete_recording(recording_id: int, db = Depends(get_db)):
    recording = db.get(Recording, recording_id)
    if not recording: raise HTTPException(status_code=404, detail="记录未找到")
    if recording.storage_path and os.path.exists(recording.storage_path):
        os.remove(recording.storage_path)
    db.delete(recording)
    db.commit()
    return RedirectResponse(url="/", status_code=303)