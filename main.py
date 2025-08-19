import json
import os
import time
import uuid
import threading
from typing import Any, Dict, List, Optional, TypedDict, Union

import requests
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field


# CodeGeeX Token Management
class CodeGeeXToken(TypedDict):
    token: str
    is_valid: bool
    last_used: float
    error_count: int


# Global variables
VALID_CLIENT_KEYS: set = set("sk-gweidwei008")
CODEGEEX_TOKENS: List[CodeGeeXToken] = ["fc793b58-4e57-4a81-9eea-24a69f09a52e"]
CODEGEEX_MODELS: List[str] = ["claude-3-7-sonnet", "claude-sonnet-4"]
token_rotation_lock = threading.Lock()
MAX_ERROR_COUNT = 3
ERROR_COOLDOWN = 300  # 5 minutes cooldown for tokens with errors
DEBUG_MODE = os.environ.get("DEBUG_MODE", "false").lower() == "true"


# Pydantic Models
class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]
    reasoning_content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = True
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


class ChatCompletionChoice(BaseModel):
    message: ChatMessage
    index: int = 0
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int] = Field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    )


class StreamChoice(BaseModel):
    delta: Dict[str, Any] = Field(default_factory=dict)
    index: int = 0
    finish_reason: Optional[str] = None


class StreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[StreamChoice]


# FastAPI App
app = FastAPI(title="CodeGeeX OpenAI API Adapter")
security = HTTPBearer(auto_error=False)


def log_debug(message: str):
    """Debug日志函数"""
    if DEBUG_MODE:
        print(f"[DEBUG] {message}")


def load_client_api_keys():
    """Load client API keys from client_api_keys.json"""
    global VALID_CLIENT_KEYS
    try:
        with open("client_api_keys.json", "r", encoding="utf-8") as f:
            keys = json.load(f)
            VALID_CLIENT_KEYS = set(keys) if isinstance(keys, list) else set()
            print(f"Successfully loaded {len(VALID_CLIENT_KEYS)} client API keys.")
    except FileNotFoundError:
        print("Error: client_api_keys.json not found. Client authentication will fail.")
        VALID_CLIENT_KEYS = set()
    except Exception as e:
        print(f"Error loading client_api_keys.json: {e}")
        VALID_CLIENT_KEYS = set()


def load_codegeex_tokens():
    """Load CodeGeeX tokens from codegeex.txt"""
    global CODEGEEX_TOKENS
    CODEGEEX_TOKENS = []
    try:
        with open("codegeex.txt", "r", encoding="utf-8") as f:
            for line in f:
                token = line.strip()
                if token:
                    CODEGEEX_TOKENS.append({
                        "token": token,
                        "is_valid": True,
                        "last_used": 0,
                        "error_count": 0
                    })
            print(f"Successfully loaded {len(CODEGEEX_TOKENS)} CodeGeeX tokens.")
    except FileNotFoundError:
        print("Error: codegeex.txt not found. API calls will fail.")
    except Exception as e:
        print(f"Error loading codegeex.txt: {e}")


def get_best_codegeex_token() -> Optional[CodeGeeXToken]:
    """Get the best available CodeGeeX token using a smart selection algorithm."""
    with token_rotation_lock:
        now = time.time()
        valid_tokens = [
            token for token in CODEGEEX_TOKENS 
            if token["is_valid"] and (
                token["error_count"] < MAX_ERROR_COUNT or 
                now - token["last_used"] > ERROR_COOLDOWN
            )
        ]
        
        if not valid_tokens:
            return None
            
        # Reset error count for tokens that have been in cooldown
        for token in valid_tokens:
            if token["error_count"] >= MAX_ERROR_COUNT and now - token["last_used"] > ERROR_COOLDOWN:
                token["error_count"] = 0
                
        # Sort by last used (oldest first) and error count (lowest first)
        valid_tokens.sort(key=lambda x: (x["last_used"], x["error_count"]))
        token = valid_tokens[0]
        token["last_used"] = now
        return token


def _convert_messages_to_codegeex_format(messages: List[ChatMessage]):
    """Convert OpenAI messages format to CodeGeeX prompt and history format."""
    if not messages:
        return "", []
    
    # Extract the last user message as prompt
    last_user_msg = None
    for msg in reversed(messages):
        if msg.role == "user":
            last_user_msg = msg
            break
    
    if not last_user_msg:
        raise HTTPException(status_code=400, detail="No user message found in the conversation.")
    
    prompt = last_user_msg.content if isinstance(last_user_msg.content, str) else ""
    
    # Build history from previous messages (excluding the last user message)
    history = []
    user_content = ""
    assistant_content = ""
    
    for i, msg in enumerate(messages[:-1] if messages[-1].role == "user" else messages):
        if msg == last_user_msg:
            continue
            
        if msg.role == "user":
            # If we have a complete pair, add it to history
            if user_content and assistant_content:
                history.append({
                    "query": user_content,
                    "answer": assistant_content,
                    "id": f"{uuid.uuid4()}"
                })
                user_content = ""
                assistant_content = ""
            
            # Start a new pair with this user message
            content = msg.content if isinstance(msg.content, str) else ""
            user_content = content
        
        elif msg.role == "assistant":
            content = msg.content if isinstance(msg.content, str) else ""
            assistant_content = content
            
            # If we have a complete pair, add it to history
            if user_content:
                history.append({
                    "query": user_content,
                    "answer": assistant_content,
                    "id": f"{uuid.uuid4()}"
                })
                user_content = ""
                assistant_content = ""
    
    # Handle any remaining unpaired messages
    if user_content and not assistant_content:
        # Unpaired user message - treat as part of the prompt
        prompt = user_content + "\n" + prompt
    
    return prompt, history


async def authenticate_client(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """Authenticate client based on API key in Authorization header"""
    if not VALID_CLIENT_KEYS:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: Client API keys not configured on server.",
        )

    if not auth or not auth.credentials:
        raise HTTPException(
            status_code=401,
            detail="API key required in Authorization header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if auth.credentials not in VALID_CLIENT_KEYS:
        raise HTTPException(status_code=403, detail="Invalid client API key.")


@app.on_event("startup")
async def startup():
    """应用启动时初始化配置"""
    print("Starting CodeGeeX OpenAI API Adapter server...")
    load_client_api_keys()
    load_codegeex_tokens()
    print("Server initialization completed.")


def get_models_list_response() -> ModelList:
    """Helper to construct ModelList response from cached models."""
    model_infos = [
        ModelInfo(
            id=model,
            created=int(time.time()),
            owned_by="anthropic"
        )
        for model in CODEGEEX_MODELS
    ]
    return ModelList(data=model_infos)


@app.get("/v1/models", response_model=ModelList)
async def list_v1_models(_: None = Depends(authenticate_client)):
    """List available models - authenticated"""
    return get_models_list_response()


@app.get("/models", response_model=ModelList)
async def list_models_no_auth():
    """List available models without authentication - for client compatibility"""
    return get_models_list_response()


@app.get("/debug")
async def toggle_debug(enable: bool = Query(None)):
    """切换调试模式"""
    global DEBUG_MODE
    if enable is not None:
        DEBUG_MODE = enable
    return {"debug_mode": DEBUG_MODE}


def _codegeex_stream_generator(response, model: str):
    """Real-time streaming with format conversion - CodeGeeX to OpenAI"""
    stream_id = f"chatcmpl-{uuid.uuid4().hex}"
    created_time = int(time.time())

    # 发送初始角色增量
    yield f"data: {StreamResponse(id=stream_id, created=created_time, model=model, choices=[StreamChoice(delta={'role': 'assistant'})]).json()}\n\n"

    buffer = ""
    
    try:
        for chunk in response.iter_content(chunk_size=1024):
            if not chunk:
                continue
                
            chunk_text = chunk.decode("utf-8")
            log_debug(f"Received chunk: {chunk_text[:100]}..." if len(chunk_text) > 100 else chunk_text)
            buffer += chunk_text

            # 处理缓冲区中的完整事件块
            while "\n\n" in buffer:
                event_data, buffer = buffer.split("\n\n", 1)
                event_data = event_data.strip()
                
                if not event_data:
                    continue
                
                # 解析事件
                event_type = None
                data_json = None
                
                for line in event_data.split("\n"):
                    line = line.strip()
                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        try:
                            data_json = json.loads(line[5:].strip())
                        except json.JSONDecodeError:
                            log_debug(f"Failed to parse JSON: {line[5:].strip()}")
                
                if not event_type or not data_json:
                    continue
                
                if event_type == "add":
                    # 'text' 字段本身就是增量内容
                    delta = data_json.get("text", "")
                    if delta:
                        openai_response = StreamResponse(
                            id=stream_id,
                            created=created_time,
                            model=model,
                            choices=[StreamChoice(delta={"content": delta})],
                        )
                        yield f"data: {openai_response.json()}\n\n"
                
                elif event_type == "finish":
                    # 'finish' 事件标志着流的结束
                    log_debug("Received finish event.")
                    openai_response = StreamResponse(
                        id=stream_id,
                        created=created_time,
                        model=model,
                        choices=[StreamChoice(delta={}, finish_reason="stop")],
                    )
                    yield f"data: {openai_response.json()}\n\n"
                    yield "data: [DONE]\n\n"
                    return # 终止生成器
    
    except Exception as e:
        log_debug(f"Stream processing error: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    # 如果流意外中断，也发送终止信号
    log_debug("Stream finished unexpectedly, sending completion signal.")
    yield f"data: {StreamResponse(id=stream_id, created=created_time, model=model, choices=[StreamChoice(delta={}, finish_reason='stop')]).json()}\n\n"
    yield "data: [DONE]\n\n"


def _build_codegeex_non_stream_response(response, model: str) -> ChatCompletionResponse:
    """Build non-streaming response by accumulating stream data."""
    full_content = ""
    buffer = ""
    
    for chunk in response.iter_content(chunk_size=1024):
        if not chunk:
            continue
            
        buffer += chunk.decode("utf-8")
        
        # 处理缓冲区中的完整事件块
        while "\n\n" in buffer:
            event_data, buffer = buffer.split("\n\n", 1)
            event_data = event_data.strip()
            
            if not event_data:
                continue
            
            # 解析事件
            event_type = None
            data_json = None
            
            for line in event_data.split("\n"):
                line = line.strip()
                if line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("data:"):
                    try:
                        data_json = json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        continue
            
            if not event_type or not data_json:
                continue
            
            if event_type == "add":
                # 正确地累积增量文本
                full_content += data_json.get("text", "")
            
            elif event_type == "finish":
                # finish事件中的text是最终的完整文本，以此为准
                finish_text = data_json.get("text", "")
                if finish_text:
                    full_content = finish_text
                # 收到finish事件，可以提前结束解析
                return ChatCompletionResponse(
                    model=model,
                    choices=[
                        ChatCompletionChoice(
                            message=ChatMessage(
                                role="assistant",
                                content=full_content
                            )
                        )
                    ],
                )

    # 如果循环结束仍未返回（例如没有finish事件），则使用累积的内容
    return ChatCompletionResponse(
        model=model,
        choices=[
            ChatCompletionChoice(
                message=ChatMessage(
                    role="assistant",
                    content=full_content
                )
            )
        ],
    )


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest, _: None = Depends(authenticate_client)
):
    """Create chat completion using CodeGeeX backend"""
    if request.model not in CODEGEEX_MODELS:
        raise HTTPException(status_code=404, detail=f"Model '{request.model}' not found.")

    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided in the request.")
    
    log_debug(f"Processing request for model: {request.model}")
    
    # 转换消息格式
    try:
        prompt, history = _convert_messages_to_codegeex_format(request.messages)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process messages: {str(e)}")
    
    # 尝试所有令牌
    for attempt in range(len(CODEGEEX_TOKENS) + 1):  # +1 to handle the case of no tokens
        if attempt == len(CODEGEEX_TOKENS):
            raise HTTPException(
                status_code=503, 
                detail="All attempts to contact CodeGeeX API failed."
            )
            
        token = get_best_codegeex_token()
        if not token:
            raise HTTPException(
                status_code=503, 
                detail="No valid CodeGeeX tokens available."
            )

        try:
            # 构建请求
            payload = {
                "user_role": 0,
                "ide": "VSCode",
                "ide_version": "",
                "plugin_version": "",
                "prompt": prompt,
                "machineId": "",
                "talkId": f"{uuid.uuid4()}",
                "locale": "",
                "model": request.model,
                "agent": None,
                "candidates": {
                    "candidate_msg_id": "",
                    "candidate_type": "",
                    "selected_candidate": "",
                },
                "history": history,
            }

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Code/1.100.3 Chrome/132.0.6834.210 Electron/34.5.1 Safari/537.36",
                "Accept": "text/event-stream",
                "Accept-Encoding": "gzip, deflate, br, zstd",
                "Content-Type": "application/json",
                "code-token": token["token"],
            }

            log_debug(f"Sending request to CodeGeeX API with token ending in ...{token['token'][-4:]}")
            
            response = requests.post(
                "https://codegeex.cn/prod/code/chatCodeSseV3/chat",
                data=json.dumps(payload),
                headers=headers,
                stream=True,
                timeout=300.0,
            )
            response.raise_for_status()

            if request.stream:
                log_debug("Returning stream response")
                return StreamingResponse(
                    _codegeex_stream_generator(response, request.model),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )
            else:
                log_debug("Building non-stream response")
                return _build_codegeex_non_stream_response(response, request.model)

        except requests.HTTPError as e:
            status_code = getattr(e.response, "status_code", 500)
            error_detail = getattr(e.response, "text", str(e))
            log_debug(f"CodeGeeX API error ({status_code}): {error_detail}")

            with token_rotation_lock:
                if status_code in [401, 403]:
                    # 标记令牌为无效
                    token["is_valid"] = False
                    print(f"Token ...{token['token'][-4:]} marked as invalid due to auth error.")
                elif status_code in [429, 500, 502, 503, 504]:
                    # 增加错误计数
                    token["error_count"] += 1
                    print(f"Token ...{token['token'][-4:]} error count: {token['error_count']}")

        except Exception as e:
            log_debug(f"Request error: {e}")
            with token_rotation_lock:
                token["error_count"] += 1


async def error_stream_generator(error_detail: str, status_code: int):
    """Generate error stream response"""
    yield f'data: {json.dumps({"error": {"message": error_detail, "type": "codegeex_api_error", "code": status_code}})}\n\n'
    yield "data: [DONE]\n\n"


if __name__ == "__main__":
    import uvicorn

    # 设置环境变量以启用调试模式
    if os.environ.get("DEBUG_MODE", "").lower() == "true":
        DEBUG_MODE = True
        print("Debug mode enabled via environment variable")

    if not os.path.exists("codegeex.txt"):
        print("Warning: codegeex.txt not found. Creating a dummy file.")
        with open("codegeex.txt", "w", encoding="utf-8") as f:
            f.write(f"your-codegeex-token-here\n")
        print("Created dummy codegeex.txt. Please replace with valid CodeGeeX token.")

    if not os.path.exists("client_api_keys.json"):
        print("Warning: client_api_keys.json not found. Creating a dummy file.")
        dummy_key = f"sk-dummy-{uuid.uuid4().hex}"
        with open("client_api_keys.json", "w", encoding="utf-8") as f:
            json.dump([dummy_key], f, indent=2)
        print(f"Created dummy client_api_keys.json with key: {dummy_key}")

    load_client_api_keys()
    load_codegeex_tokens()

    print("\n--- CodeGeeX OpenAI API Adapter ---")
    print(f"Debug Mode: {DEBUG_MODE}")
    print("Endpoints:")
    print("  GET  /v1/models (Client API Key Auth)")
    print("  GET  /models (No Auth)")
    print("  POST /v1/chat/completions (Client API Key Auth)")
    print("  GET  /debug?enable=[true|false] (Toggle Debug Mode)")

    print(f"\nClient API Keys: {len(VALID_CLIENT_KEYS)}")
    if CODEGEEX_TOKENS:
        print(f"CodeGeeX Tokens: {len(CODEGEEX_TOKENS)}")
    else:
        print("CodeGeeX Tokens: None loaded. Check codegeex.txt.")
    
    print(f"Available models: {', '.join(CODEGEEX_MODELS)}")
    print("------------------------------------")

    uvicorn.run(app, host="0.0.0.0", port=3005)