import asyncio
import json
import random

from src.infra.redis_runtime import (
    get_redis_json,
    READY_QUEUE,
    load_job,
    set_job_status,
    try_acquire_inflight,
    release_inflight,
    RedisStateQueue,
    k_user_queue,
)

from src.utils.log_utils import setup_logger
from src.core.state_models import BackToFrontData, ExecutionState  # 你已有
logger = setup_logger(name='main', log_file='project.log')

async def worker_loop(worker_id: int, inflight_k: int = 2):
    r = get_redis_json()

    from src.agents.orchestrator import PaperAgentOrchestrator  # 延迟 import 避免循环依赖

    while True:
        logger.info(f"[Worker {worker_id}] 开始等待新任务")
        # 阻塞取 ready job_id
        item = await r.blpop(READY_QUEUE, timeout=30)
        if item is None:
            continue
        _, job_id = item

        job = await load_job(r, job_id)
        logger.info(f"[Worker {worker_id}] 开始处理任务 {job_id}")
        
        ok = await try_acquire_inflight(r, job.user_id, limit_k=inflight_k)
        if not ok:
            # 抢不到槽位：放回该用户队列末尾，稍后再调度
            await r.rpush(k_user_queue(job.user_id), job.job_id)
            await asyncio.sleep(0.2 + random.random() * 0.2)
            continue

        state_queue = RedisStateQueue(r, job.job_id)

        try:
            await set_job_status(r, job.job_id, "running")
            await state_queue.put(BackToFrontData(step="worker_start", state="running", data={"worker": worker_id}))
            logger.info(f"[Worker {worker_id}] 开始执行任务 {job_id}")
            
            orchestrator = PaperAgentOrchestrator(state_queue=state_queue)
            
            await orchestrator.run(user_request=job.query, max_papers=job.max_papers)

            # 你 orchestrator.run() 里本来会 put FINISHED 事件；这里可选再落个 meta
            await set_job_status(r, job.job_id, "finished")

        except Exception as e:
            logger.error(f"[Worker {worker_id}] 任务 {job_id} 执行失败: {str(e)}")
            await set_job_status(r, job.job_id, "failed")
            await state_queue.put(BackToFrontData(step="error", state="failed", data={"error": str(e)}))

        finally:
            await release_inflight(r, job.user_id)


async def main():
    # 单机多进程：你可以起多个 worker_main.py 进程
    # 每个进程内部也可以起多个协程 worker（视 IO/CPU 情况）
    workers = [asyncio.create_task(worker_loop(worker_id=i, inflight_k=2)) for i in range(4)]
    await asyncio.gather(*workers)


if __name__ == "__main__":
    asyncio.run(main())