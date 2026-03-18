import asyncio
import os
import random
from collections import defaultdict
from typing import Dict, Optional
from urllib.parse import urlparse
import aiohttp
from src.utils.log_utils import setup_logger
logger = setup_logger(__name__)

from src.infra.redis_runtime import (
    get_redis_json,
    get_redis_bytes,
    DOWNLOAD_QUEUE,
    PARSE_QUEUE,
    load_task,
    save_task,
    set_task_status,
    k_cache_pdf,
    try_acquire_inflight,
    release_inflight,
    decr_pending,
)

# ===== 配置 =====
HOST_CONCURRENCY = int(os.getenv("HOST_CONCURRENCY", "3"))
HTTP_TIMEOUT_S = int(os.getenv("HTTP_TIMEOUT_S", "30"))
MAX_RETRY = int(os.getenv("DOWNLOAD_MAX_RETRY", "4"))
BACKOFF_BASE = float(os.getenv("BACKOFF_BASE", "0.5"))
BACKOFF_CAP = float(os.getenv("BACKOFF_CAP", "8.0"))
PDF_CACHE_TTL = int(os.getenv("PDF_CACHE_TTL", "86400"))  # 1 day
USER_INFLIGHT_K = int(os.getenv("USER_INFLIGHT_K", "2"))  # 单用户最大并发下载数

_host_sems: Dict[str, asyncio.Semaphore] = defaultdict(lambda: asyncio.Semaphore(HOST_CONCURRENCY))


def _host_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return "unknown"


def _should_retry(status: Optional[int], exc: Optional[Exception]) -> bool:
    if exc is not None:
        return True
    if status is None:
        return True
    if status == 429:
        return True
    if 500 <= status <= 599:
        return True
    return False


def _backoff_s(attempt: int) -> float:
    base = min(BACKOFF_BASE * (2 ** max(0, attempt)), BACKOFF_CAP)
    return base + random.random() * base  # jitter


async def _download_bytes(session: aiohttp.ClientSession, task_id: str, url: str) -> bytes:
    async with session.get(url) as resp:
        status = resp.status
        if status < 200 or status >= 300:
            logger.error(f"下载任务 {task_id} 失败，HTTP {status}")
            raise aiohttp.ClientResponseError(
                request_info=resp.request_info,
                history=resp.history,
                status=status,
                message=f"HTTP {status}",
                headers=resp.headers,
            )
        logger.info(f"下载任务 {task_id} 成功，HTTP {status}")
        return await resp.read()


async def download_worker_loop(worker_id: int):
    r_json = get_redis_json()   # 用于任务数据
    r_bytes = get_redis_bytes() # 用于 PDF 缓存
    timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT_S)
    
    logger.info(f"下载 worker {worker_id} 启动")

    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            
            item = await r_json.blpop(DOWNLOAD_QUEUE, timeout=30)
            if item is None:
                continue
            _, task_id = item
            if isinstance(task_id, (bytes, bytearray)):
                task_id = task_id.decode("utf-8", errors="ignore")
            

            task = await load_task(r_json, task_id)
            
            if not task:
                continue

            user_id = task.get("user_id")
            user_id = f"{user_id}:download"
            job_id = task.get("job_id")

            # ---- 用户级 inflight 限流 ----
            ok = await try_acquire_inflight(r_json, user_id, limit_k=USER_INFLIGHT_K)
            
            if not ok:
                await r_json.rpush(DOWNLOAD_QUEUE, task_id)
                await asyncio.sleep(0.2 + random.random() * 0.2)
                continue

            pdf_url = task.get("pdf_url") or ""
            paper_id = task.get("paper_id") or ""
            host = (task.get("host") or _host_of(pdf_url)).lower()
            attempt = int(task.get("attempt") or 0)
            
            
            # ---- PDF cache 命中 ----
            cached = await r_bytes.get(k_cache_pdf(paper_id))
            if cached:
                
                logger.info(f"下载任务 {task_id} 从缓存命中 {paper_id}")
                await set_task_status(r_json, task_id, "downloaded_cache_hit")
                parse_task = dict(task)
                parse_task["type"] = "parse"
                parse_task["status"] = "queued"
                await save_task(r_json, task_id, parse_task)
                await r_json.rpush(PARSE_QUEUE, task_id)
                await release_inflight(r_json, user_id)
                
                if job_id:
                    await decr_pending(r_json, job_id, 1)
                continue

            await set_task_status(r_json, task_id, "downloading")

            sem = _host_sems[host]
            status_code = None
            exc = None

            try:
                async with sem:
                    logger.info(f"下载任务 {task_id} 从 {pdf_url}")
                    data = await _download_bytes(session, task_id, pdf_url)

                # 写入 PDF cache (bytes)
                await r_bytes.set(k_cache_pdf(paper_id), data, ex=PDF_CACHE_TTL)
                await set_task_status(r_json, task_id, "downloaded")

                # 投递 parse task
                parse_task = dict(task)
                parse_task["type"] = "parse"
                parse_task["status"] = "queued"
                await save_task(r_json, task_id, parse_task)
                await r_json.rpush(PARSE_QUEUE, task_id)

                if job_id:
                    await decr_pending(r_json, job_id, 1)

            except aiohttp.ClientResponseError as e:
                status_code = getattr(e, "status", None)
                exc = e
                logger.error(f"下载任务 {task_id} 失败，HTTP {status_code}，重试 {attempt} 次")
            except Exception as e:
                exc = e
                logger.error(f"下载任务 {task_id} 失败，异常 {exc}，重试 {attempt} 次")
            finally:
                await release_inflight(r_json, user_id)

            # ---- 重试逻辑 ----
            if exc is not None:
                if _should_retry(status_code, exc) and attempt < MAX_RETRY:
                    logger.error(f"下载任务 {task_id} 失败，HTTP {status_code}，重试 {attempt} 次")
                    attempt += 1
                    task["attempt"] = attempt
                    task["status"] = "retrying"
                    await save_task(r_json, task_id, task)
                    await asyncio.sleep(_backoff_s(attempt))
                    await r_json.rpush(DOWNLOAD_QUEUE, task_id)
                else:
                    await set_task_status(r_json, task_id, "failed", err=str(exc))
                    if job_id:
                        await decr_pending(r_json, job_id, 1)


async def main():
    n = int(os.getenv("DOWNLOAD_WORKERS", "8"))
    
    logger.info(f"启动 {n} 个下载 worker")
    tasks = [asyncio.create_task(download_worker_loop(i)) for i in range(n)]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())