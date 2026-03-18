import uuid
from typing import List, Dict, Optional
from src.infra.redis_runtime import (
    get_redis_json,
    save_task,
    DOWNLOAD_QUEUE,
    incr_pending,
    k_job_meta,
)
from src.utils.log_utils import setup_logger
logger = setup_logger(__name__)

async def submit_download_tasks(
    papers: List[Dict],
    state_queue,
    max_papers: int = 50,
    job_id: Optional[str] = None
) -> str:
    """
    根据论文列表生成下载任务并投递到 DOWNLOAD_QUEUE
    兼容 Redis 体系和 orchestrator state_queue

    参数：
        papers: List[Dict]，每篇论文至少包含 paper_id 和 pdf_url
        state_queue: RedisStateQueue 对象，内部包含 job_id 和 Redis 客户端
        max_papers: 最大论文数量
        job_id: 可选，已有 job_id，如果 None 使用 state_queue.job_id

    返回：
        job_id: 所属 Job ID
    """
    if not papers:
        return None
    
    user_id = ""
    r = state_queue.r
    if job_id is None:
        job_id = state_queue.job_id
        user_id = await r.hget(k_job_meta(job_id), "user_id") 
        if user_id ==None:
            print(f"job_id {job_id} 没有 user_id")

    for paper in papers:
        paper_id = paper.get("paper_id")
        pdf_url = paper.get("pdf_url")
        paper_title = paper.get("title")
        paper_abstract = paper.get("summary")
        if not paper_id or not pdf_url:
            continue

        task_id = str(uuid.uuid4())
        task = {
            "task_id": task_id,
            "user_id": user_id,
            "paper_id": paper_id,
            "paper_title": paper_title,
            "paper_abstract": paper_abstract,
            "pdf_url": pdf_url,
            "job_id": job_id,
            "attempt": 0,
            "status": "queued"
        }
        

        # 保存 task 到 Redis
        await save_task(r, task_id, task)
        # 投递到下载队列
        
        logger.info(f"投递下载任务 {task_id} 到队列 {DOWNLOAD_QUEUE}")
        
        await r.rpush(DOWNLOAD_QUEUE, task_id)
        # 增加 Job pending 计数
        await incr_pending(r, job_id, 2)

    return job_id