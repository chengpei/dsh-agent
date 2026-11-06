"""持久的 Web 对话服务：FastAPI + SSE，全局单会话，可手动「新开对话」。

启动：
    python server.py            # -> http://localhost:8000

复用 main.py 的约定：自行 load_dotenv 注入 DEEPSEEK_API_KEY 等环境变量，
供底层 dsh runtime 子进程继承（SDK 不会自动读 .env）。
"""

import json
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from deepseek_harness import DeepSeekHarness

# ---------------------------------------------------------------- 配置
load_dotenv(Path(__file__).resolve().parent / ".env")

WORKSPACE = str(Path(Path(__file__).resolve().parent / "workspace").resolve())
DSH_HOME = str(Path(Path(__file__).resolve().parent / "dsh-home").resolve())
PROVIDER = "deepseek-official"
MODEL = "deepseek-v4-flash"
MAX_TOKENS = 49_152
PROFILE = "sdk"

# ---------------------------------------------------------------- 全局资源
_lock = threading.Lock()          # 保护 harness/session 的创建与切换
_run_lock = threading.Lock()      # 同一时刻只允许一个会话跑 turn（全局单会话串行）
_harness: DeepSeekHarness | None = None
_session = None                   # 当前全局 Session；首次使用 / reset 时重建


def get_session(reset: bool):
    """返回当前全局 Session。reset=True 或首次调用时创建新的（随机 uuid）会话。"""
    global _harness, _session
    with _lock:
        if _harness is None:
            _harness = DeepSeekHarness(
                provider=PROVIDER,
                model=MODEL,
                max_tokens=MAX_TOKENS,
                cwd=WORKSPACE,
                dsh_home=DSH_HOME,
                profile=PROFILE,
            )
            _harness.start()
        if reset or _session is None:
            # 每次都用随机 uuid，避免与磁盘上已持久化的同名会话冲突
            _session = _harness.start_session(session_id=uuid.uuid4().hex)
        return _session


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    # 关闭时回收 dsh runtime 子进程
    if _harness is not None:
        try:
            _harness.close()
        except Exception:
            pass


app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    reset: bool = False


# ---------------------------------------------------------------- SSE
def sse_frame(event: str, data: object) -> str:
    """把事件编码为一条 SSE 帧。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def split_message_stream(ev: dict):
    """把一条 assistant/message 按其 stream 的 text-chunks 拆成细粒度 SSE 帧，
    按模型真实生成时间轴(dt)逐 token 转发 —— 真正的服务端流式输出。
    前端收到一帧 append 一帧，不本地重放、不人为延迟（服务端快就快）。
    reasoning 思维链不暴露。
    帧序列：msg_start(text 兜底) -> text(token)*n -> msg_end。"""
    d = ev.get("data") or {}
    msg = d.get("message") or {}
    mid = msg.get("id") or "assistant"
    full = "".join(
        b.get("text", "")
        for b in (msg.get("content") or [])
        if isinstance(b, dict) and b.get("type") == "text"
    )
    yield sse_frame("msg_start", {"id": mid, "text": full})
    stream = d.get("stream")
    if isinstance(stream, list):
        for it in stream:
            if not isinstance(it, dict) or it.get("type") != "text-chunks":
                continue
            texts = it.get("texts") or []
            dt = it.get("dt") or []
            for i, tok in enumerate(texts):
                # dt[i-1] 是 tok[i] 相对 tok[i-1] 的耗时(ms)；复现模型生成节奏
                if i > 0 and i - 1 < len(dt) and dt[i - 1]:
                    time.sleep(dt[i - 1] / 1000.0)
                yield sse_frame("text", {"id": mid, "token": tok})
    yield sse_frame("msg_end", {"id": mid})


def event_generator(message: str, reset: bool):
    q: "queue.Queue[tuple[str, object]]" = queue.Queue()
    S = get_session(reset)
    yield sse_frame("meta", {"session_id": S.id})

    def run_turn():
        # session.run 是同步阻塞的：在独立线程里跑，把事件塞进 queue，
        # 让 SSE 生成器逐条取出并转发（不能在生成器里直接 return/yield）。
        # _run_lock 放在 worker 内部：锁的生命周期跟随 turn 本身，即使客户端
        # 断开导致生成器被关闭，锁也不会提前释放，从而避免同一会话被并发 run。
        with _run_lock:
            try:
                result = S.run(message, on_notification=lambda n: q.put(("notif", n)))
                q.put(("finish", result))
            except Exception as exc:  # noqa: BLE001 —— 统一转成 error 帧
                q.put(("error", {"message": str(exc)}))

    thread = threading.Thread(target=run_turn, daemon=True)
    thread.start()
    try:
        while True:
            kind, payload = q.get()
            if kind == "finish":
                yield sse_frame(
                    "status",
                    {
                        "finish_reason": payload.finish_reason,
                        "final_response": payload.final_response,
                    },
                )
                break
            if kind == "error":
                yield sse_frame("error", payload)
                break
            # kind == "notif"：assistant/message 拆成流式帧，其余透传 agent 事件
            n = payload
            if n.method == "session.event":
                ev = n.payload.get("event")
                if isinstance(ev, dict):
                    if ev.get("type") == "assistant/message":
                        yield from split_message_stream(ev)
                    else:
                        yield sse_frame("agent", ev)
    finally:
        # 客户端断开 / 生成器被关闭时，等当前 turn 自然结束，再允许下一个请求进场
        thread.join()


# ---------------------------------------------------------------- 路由
@app.get("/")
def index():
    return FileResponse(Path(__file__).resolve().parent / "web" / "index.html")


@app.get("/api/session")
def session_info():
    return {
        "session_id": _session.id if _session else None,
        "started": _session is not None,
    }


@app.post("/api/chat", response_class=StreamingResponse)
def chat(req: ChatRequest):
    """POST /api/chat  body: {"message": "...", "reset": false}
    返回 text/event-stream。"""
    return StreamingResponse(
        event_generator(req.message, req.reset),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
