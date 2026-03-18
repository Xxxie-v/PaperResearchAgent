import asyncio
import random
import redis.asyncio as redis

from src.infra.redis_runtime import (
    get_redis_json,
    ACTIVE_USERS,
    READY_QUEUE,
    k_user_queue,
)

# 空队列次数阈值，超过就将用户从 ACTIVE_USERS 移除，避免长期轮询空用户
EMPTY_THRESHOLD = 50

async def rr_feeder_loop(poll_sleep: float = 0.02):
    r = get_redis_json()
    idx = 0
    empty_counts = {}  # uid -> empty hits

    while True:
        user_ids = list(await r.smembers(ACTIVE_USERS))
        if not user_ids:
            await asyncio.sleep(poll_sleep)
            continue

        uid = user_ids[idx % len(user_ids)]
        idx += 1

        # 从该用户队列拿一个 job
        job_id = await r.lpop(k_user_queue(uid))
        if job_id is None:
            # 空队列统计，避免 feeder 永久扫描空用户
            c = empty_counts.get(uid, 0) + 1
            empty_counts[uid] = c
            if c >= EMPTY_THRESHOLD:
                await r.srem(ACTIVE_USERS, uid)
                empty_counts.pop(uid, None)
            await asyncio.sleep(poll_sleep)
            continue

        # 把 job 放进全局 ready 队列（worker 就从这里 BLPOP）
        await r.rpush(READY_QUEUE, job_id)

        # 该用户确实有任务，清空空计数
        empty_counts.pop(uid, None)

        # 可选：轻微 sleep + jitter，减少 Redis 压力/抖动
        if poll_sleep > 0:
            await asyncio.sleep(poll_sleep + random.random() * poll_sleep)

async def main():
    await rr_feeder_loop()

if __name__ == "__main__":
    asyncio.run(main())