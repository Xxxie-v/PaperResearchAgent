import os
import json
import asyncio
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, AsyncIterator, Tuple

import redis.asyncio as redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Key conventions
def k_user_queue(user_id: str) -> str:
    return f"user:{user_id}:queue"          # LIST of job_id

READY_QUEUE = "queue:ready"                 # LIST of job_id
ACTIVE_USERS = "users:active"               # SET of user_id

DOWNLOAD_QUEUE = "download_queue"           
PARSE_QUEUE = "parse_queue"                


def k_task(task_id: str) -> str:
    return f"task:{task_id}"

def k_job_pending(job_id: str) -> str:
    return f"job:{job_id}:tasks_pending"

def k_job_paper_pdf(job_id: str, paper_id: str) -> str:
    return f"job:{job_id}:paper:{paper_id}:pdf"  # 存 pdf bytes 或引用

def k_job_paper_parsed(job_id: str, paper_id: str) -> str:
    return f"job:{job_id}:paper:{paper_id}:parsed"  # 存解析文本或引用

def k_cache_pdf(paper_id: str) -> str:
    return f"cache:pdf:{paper_id}"

def k_cache_parsed(paper_id: str) -> str:
    return f"cache:parsed:{paper_id}"

def k_job_meta(job_id: str) -> str:
    return f"job:{job_id}:meta"             # HASH

def k_job_events(job_id: str) -> str:
    return f"job:{job_id}:events"           # STREAM

def k_user_inflight(user_id: str) -> str:
    return f"user:{user_id}:inflight"       # STRING (counter)

# ===== Task CRUD =====
async def save_task(r, task_id: str, payload: Dict[str, Any], *, ttl_s: int = 3600) -> None:
    # 统一用 JSON 存 payload，避免 hash 字段类型问题
    key = k_task(task_id)
    await r.set(key, json.dumps(payload, ensure_ascii=False))
    if ttl_s:
        await r.expire(key, ttl_s)

async def load_task(r, task_id: str) -> Optional[Dict[str, Any]]:
    raw = await r.get(k_task(task_id))
    if not raw:
        return None
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="ignore")
    try:
        return json.loads(raw)
    except Exception:
        return None

async def set_task_status(r, task_id: str, status: str, *, err: Optional[str] = None) -> None:
    # 简化：读-改-写（面试项目够用）；要原子可用 lua
    payload = await load_task(r, task_id) or {"task_id": task_id}
    payload["status"] = status
    if err:
        payload["error"] = err
    await save_task(r, task_id, payload)

# ===== pending 计数 =====
async def incr_pending(r, job_id: str, n: int) -> int:
    return await r.incrby(k_job_pending(job_id), int(n))

async def decr_pending(r, job_id: str, n: int = 1) -> int:
    return await r.decrby(k_job_pending(job_id), int(n))

async def get_pending(r, job_id: str) -> int:
    v = await r.get(k_job_pending(job_id))
    if v is None:
        return 0
    if isinstance(v, (bytes, bytearray)):
        v = v.decode("utf-8", errors="ignore")
    try:
        return int(v)
    except Exception:
        return 0


@dataclass
class Job:
    job_id: str
    user_id: str
    query: str
    max_papers: int = 50

# Redis 客户端 - JSON payload，用于任务数据
def get_redis_json() -> redis.Redis:
    return redis.from_url(REDIS_URL, decode_responses=True)

# Redis 客户端 - PDF bytes，用于缓存 PDF
def get_redis_bytes() -> redis.Redis:
    return redis.from_url(REDIS_URL, decode_responses=False)

# -------------------------
# inflight(K) with Lua
# -------------------------
_ACQUIRE_LUA = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])

local v = redis.call("INCR", key)
if v > limit then
  redis.call("DECR", key)
  return 0
end
return 1
"""

_RELEASE_LUA = """
local key = KEYS[1]
local v = redis.call("DECR", key)
if v < 0 then
  redis.call("SET", key, 0)
  return 0
end
return v
"""


async def try_acquire_inflight(r: redis.Redis, user_id: str, limit_k: int) -> bool:
    key = k_user_inflight(user_id)
    ok = await r.eval(_ACQUIRE_LUA, 1, key, str(limit_k))
    return ok == 1


async def release_inflight(r: redis.Redis, user_id: str) -> None:
    key = k_user_inflight(user_id)
    await r.eval(_RELEASE_LUA, 1, key)


# -------------------------
# Job submit & meta
# -------------------------
async def submit_job(r: redis.Redis, user_id: str, query: str, max_papers: int = 50) -> Job:
    # 简单 job_id：用 Redis 自增或 uuid 都行，这里用 INCR
    job_id = str(await r.incr("jobs:seq"))
    job = Job(job_id=job_id, user_id=user_id, query=query, max_papers=max_papers)

    # 保存元数据
    await r.hset(
        k_job_meta(job_id),
        mapping={
            "user_id": user_id,
            "query": query,
            "max_papers": str(max_papers),
            "status": "queued",
        },
    )

    # 把用户标记为活跃（供 RR feeder 扫描）
    await r.sadd(ACTIVE_USERS, user_id)

    # 入 per-user queue
    await r.rpush(k_user_queue(user_id), job_id)

    # 创建一条“queued”事件，保证 stream 存在且 SSE 可立刻读到
    await r.xadd(k_job_events(job_id), {"data": json.dumps({"step": "queued", "state": "queued"})})

    return job


async def load_job(r: redis.Redis, job_id: str) -> Job:
    meta = await r.hgetall(k_job_meta(job_id))
    if not meta:
        raise RuntimeError(f"job meta not found: {job_id}")
    return Job(
        job_id=job_id,
        user_id=meta["user_id"],
        query=meta["query"],
        max_papers=int(meta.get("max_papers", "50")),
    )


async def set_job_status(r: redis.Redis, job_id: str, status: str) -> None:
    await r.hset(k_job_meta(job_id), "status", status)


# -------------------------
# Events: adapt state_queue.put -> Redis Stream XADD
# -------------------------
class RedisStateQueue:
    """兼容你现有 orchestrator/state_queue.put(...) 的最小适配器"""

    def __init__(self, r: redis.Redis, job_id: str):
        self.r = r
        self.job_id = job_id

    async def put(self, item: Any) -> None:
        # item 期望是 BackToFrontData（有 model_dump_json）
        if hasattr(item, "model_dump_json"):
            payload = item.model_dump_json()
        else:
            payload = json.dumps(item, ensure_ascii=False)
        await self.r.xadd(k_job_events(self.job_id), {"data": payload})


async def stream_events(
    r: redis.Redis, job_id: str, start_id: str = "0-0", block_ms: int = 15000
) -> AsyncIterator[str]:
    """
    SSE 用：持续从 Redis Stream 读 data 字段。
    返回的是每条消息的 JSON 字符串。
    """
    stream_key = k_job_events(job_id)
    last_id = start_id

    while True:
        resp = await r.xread({stream_key: last_id}, count=50, block=block_ms)
        if not resp:
            continue
        # resp: [(stream_key, [(id, {"data": ...}), ...])]
        _, entries = resp[0]
        for entry_id, fields in entries:
            last_id = entry_id
            data = fields.get("data")
            if data is not None:
                yield data