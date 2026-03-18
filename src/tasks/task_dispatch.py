# src/tasks/task_dispatch.py

import asyncio
import hashlib
import uuid
from typing import Any, Dict, List
from urllib.parse import urlparse

from src.infra.redis_runtime import (
    DOWNLOAD_QUEUE,
    incr_pending,
    k_cache_parsed,
    k_job_paper_parsed,
    save_task,
)

def _host_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

def _uid(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

async def dispatch_download_tasks(
    r,
    *,
    job_id: str,
    user_id: str,
    papers: List[Dict[str, Any]],
    top_k: int = 20,
    task_ttl_s: int = 3600,
) -> int:
    """
    对 papers 投递下载任务：
    - 若 cache:parsed:{paper_id} 命中：直接写 job 结果，跳过 task
    - 未命中：创建 download task -> RPUSH download_queue
    返回 pending 数量
    """
    pending = 0
    for p in papers[:top_k]:
        paper_id = (p.get("paper_id") or "").strip()
        pdf_url = (p.get("pdf_url") or "").strip()
        if not paper_id or not pdf_url:
            continue

        # 热门解析缓存命中：直接写 job 结果
        cached_parsed = await r.get(k_cache_parsed(paper_id))
        if cached_parsed:
            await r.set(k_job_paper_parsed(job_id, paper_id), cached_parsed)
            continue

        task_id = str(uuid.uuid4())
        payload = {
            "task_id": task_id,
            "type": "download",
            "status": "queued",
            "job_id": job_id,
            "user_id": user_id,
            "paper_id": paper_id,
            "pdf_url": pdf_url,
            "host": _host_of(pdf_url),
            "attempt": 0,
        }
        await save_task(r, task_id, payload, ttl_s=task_ttl_s)
        await r.rpush(DOWNLOAD_QUEUE, task_id)
        pending += 1

    if pending > 0:
        await incr_pending(r, job_id, pending)

    return pending

async def wait_tasks_done(
    r,
    *,
    job_id: str,
    timeout_s: int = 300,
    poll_s: float = 0.5,
) -> bool:
    """
    轮询 pending 直到归零或超时（面试项目足够）。
    """
    deadline = asyncio.get_event_loop().time() + timeout_s
    from src.infra.redis_runtime import get_pending

    while asyncio.get_event_loop().time() < deadline:
        left = await get_pending(r, job_id)
        if left <= 0:
            return True
        await asyncio.sleep(poll_s)
    return False