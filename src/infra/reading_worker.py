import asyncio
import uuid
import base64
import random
from typing import Dict, Any
import json
import pdfplumber
from io import BytesIO
from src.infra.redis_runtime import (
    get_redis_json,
    get_redis_bytes,
    PARSE_QUEUE,
    load_task,
    save_task,
    set_task_status,
    try_acquire_inflight,
    release_inflight,
    decr_pending,
    k_cache_pdf,
)
from src.utils.log_utils import setup_logger
from src.core.model_client import create_reading_model_client
from autogen_agentchat.agents import AssistantAgent
from src.core.prompts import reading_agent_prompt
from src.core.state_models import ExtractedPaperData
from src.knowledge.knowledge import knowledge_base

logger = setup_logger(__name__)

def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))

def _extract_text_with_spaces(page) -> str:
    words = page.extract_words(x_tolerance=2, y_tolerance=2)
    if not words:
        return page.extract_text(x_tolerance=2, y_tolerance=2, use_text_flow=True) or ""
    lines = []
    for w in words:
        placed = False
        for line in lines:
            if abs(w["top"] - line[0]["top"]) <= 3:
                line.append(w)
                placed = True
                break
        if not placed:
            lines.append([w])
    out_lines = []
    for line in sorted(lines, key=lambda l: l[0]["top"]):
        line_sorted = sorted(line, key=lambda w: w["x0"])
        out_lines.append(" ".join(w["text"] for w in line_sorted))
    return "\n".join(out_lines)

# --------------------------
# 配置
# --------------------------
PARSE_CONCURRENCY = 4
USER_INFLIGHT_K = 5
MAX_RETRY = 3

# 初始化阅读 agent


# --------------------------
# Parse Worker
# --------------------------
async def parse_worker(worker_id: int, concurrency: int = PARSE_CONCURRENCY):
    r_json = get_redis_json()
    r_bytes = get_redis_bytes()
    sem = asyncio.Semaphore(concurrency)

    logger.info(f"[Parse Worker {worker_id}] 启动")


    while True:
        item = await r_json.blpop(PARSE_QUEUE, timeout=30)
        if not item:
            continue
        _, task_id = item
        task = await load_task(r_json, task_id)
        if not task:
            continue
        
        model_client = create_reading_model_client()
        read_agent = AssistantAgent(
            name="read_agent",
            model_client=model_client,
            system_message=reading_agent_prompt,
            output_content_type=ExtractedPaperData,
            model_client_stream=True
            )


        user_id = f"{task.get('user_id')}:parse"
        job_id = task.get("job_id")
        paper_id = task.get("paper_id")
        attempt = int(task.get("attempt", 0))
        paper_title = task.get("paper_title")
        paper_abstract = task.get("paper_abstract")

        logger.info(f"[Parse Worker {worker_id}] 处理 Paper {paper_id}，attempt {attempt}")
        # 用户级 inflight 限流
        ok = await try_acquire_inflight(r_json, user_id, limit_k=USER_INFLIGHT_K)
        if not ok:
            await r_json.rpush(PARSE_QUEUE, task_id)
            await asyncio.sleep(0.2 + 0.2 * random.random())
            logger.info(f"[Parse Worker {worker_id}] 处理 Paper {paper_id}，attempt {attempt} 失败，用户 inflight 限流")
            continue
        
        logger.info(f"[Parse Worker {worker_id}] 开始处理 Paper {paper_id}，attempt {attempt}")


        async with sem:
            try:
                # 读取 PDF bytes
                pdf_bytes = await r_bytes.get(k_cache_pdf(paper_id))
                if not pdf_bytes:
                    logger.warning(f"[Parse Worker {worker_id}] Paper {paper_id} PDF 未命中缓存")
                    await set_task_status(r_json, task_id, "failed", err="PDF missing")
                    if job_id:
                        await decr_pending(r_json, job_id, 1)
                    continue
                pdf_file = BytesIO(pdf_bytes)
                with pdfplumber.open(pdf_file) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += _extract_text_with_spaces(page) + "\n"
                
                task_payload = {
                    "paper_id": paper_id,
                    "pdf_b64": text,
                    "title": paper_title,
                    "abstract": paper_abstract,
                }
                # 调用阅读 agent
                result = await read_agent.run(task=json.dumps(task_payload))
                
                if hasattr(result, "model_dump"):
                    parsed_paper = result.model_dump()
                else:
                    parsed_paper = result
                out = parsed_paper["messages"][-1]["content"]
                logger.info(f"[Parse Worker {worker_id}] Paper {paper_id} 解析完成，attempt {attempt}")
                
                # -------------------------------
                # 按 job_id 聚合保存
                # -------------------------------
                job_key = f"job:{job_id}:parsed_results"
                job_results = await r_json.get(job_key)
                if job_results:
                    job_results = json.loads(job_results)
                else:
                    job_results = []

                job_results.append({
                    "paper_id": paper_id,
                    "title": paper_title,
                    "abstract": paper_abstract,
                    "parsed_data": _json_safe(out)
                })

                await r_json.set(job_key, json.dumps(job_results, ensure_ascii=False))
                logger.info(f"[Parse Worker {worker_id}] 已保存 Paper {paper_id} 到 job {job_id}")
                
                
                # 写入知识库
               # await knowledge_base.add_processed_content(parsed_paper)  # 替换为你的 add_papers_to_kb 实现

                # 更新 task 状态
                task["parsed_data"] = _json_safe(out)
                await save_task(r_json, task_id, _json_safe(task))
                await set_task_status(r_json, task_id, "parsed")

                if job_id:
                    await decr_pending(r_json, job_id, 1)

            except Exception as e:
                logger.error(f"[Parse Worker {worker_id}] Paper {paper_id} 解析失败: {e}, attempt {attempt}")
                msg =""
                # 有些 SDK 异常会把 info 放在 e.args[0] 或 e.args
                if hasattr(e, "args") and e.args:
                    msg = str(e.args[0]).lower()
                else:
                    msg = str(e).lower()
                    
                  # 判断是否是 max_prompt_tokens 超限
                if "max_prompt_tokens" in msg or "number of input tokens" in msg:
                    logger.warning(f"[Parse Worker {worker_id}] Paper {paper_id} token 超限，降级只传 title/abstract")
                    # 构造降级 task
                    task_payload_small = {
                        "paper_id": paper_id,
                        "title": paper_title,
                        "abstract": paper_abstract,
                    }
                    print(task_payload_small)
                    
                    try:
                        result = await read_agent.run(task=json.dumps(task_payload_small))
                        
                        if hasattr(result, "model_dump"):
                            parsed_paper = result.model_dump()
                        else:
                            parsed_paper = result
                        out = parsed_paper["messages"][-1]["content"]
                        logger.info(f"[Parse Worker {worker_id}] Paper {paper_id} 解析完成，attempt {attempt}")
                        
                        # -------------------------------
                        # 按 job_id 聚合保存
                        # -------------------------------
                        job_key = f"job:{job_id}:parsed_results"
                        job_results = await r_json.get(job_key)
                        if job_results:
                            job_results = json.loads(job_results)
                        else:
                            job_results = []

                        job_results.append({
                            "paper_id": paper_id,
                            "title": paper_title,
                            "abstract": paper_abstract,
                            "parsed_data": _json_safe(out)
                        })

                        await r_json.set(job_key, json.dumps(job_results, ensure_ascii=False))
                        logger.info(f"[Parse Worker {worker_id}] 已保存 Paper {paper_id} 到 job {job_id}")
                        
                        if job_id:
                            await decr_pending(r_json, job_id, 1)
                        
                        
                    except Exception as e2:
                        logger.error(f"[Parse Worker {worker_id}] Paper {paper_id} 降级解析失败: {e2}, attempt {attempt}")
                        if attempt < MAX_RETRY:
                            task["attempt"] = attempt + 1
                            task["status"] = "retrying"
                            await save_task(r_json, task_id, _json_safe(task))
                            await asyncio.sleep(0.5 + random.random())
                            await r_json.rpush(PARSE_QUEUE, task_id)
                        else:
                            await set_task_status(r_json, task_id, "failed")
                            if job_id:
                                await decr_pending(r_json, job_id, 1)
                        continue
                    
                    
                if attempt < MAX_RETRY:
                    task["attempt"] = attempt + 1
                    task["status"] = "retrying"
                    await save_task(r_json, task_id, _json_safe(task))
                    await asyncio.sleep(0.5 + random.random())
                    await r_json.rpush(PARSE_QUEUE, task_id)
                else:
                    await set_task_status(r_json, task_id, "failed")
                    if job_id:
                        await decr_pending(r_json, job_id, 1)
            finally:
                await release_inflight(r_json, user_id)


# --------------------------
# 启动多个 parse Worker
# --------------------------
async def main():
    n = 4
    # #清除现在队列里的所有任务
    r_json = get_redis_json()
    await r_json.ltrim(PARSE_QUEUE, 0, 0)
    workers = [asyncio.create_task(parse_worker(i)) for i in range(n)]
    await asyncio.gather(*workers)


if __name__ == "__main__":
    asyncio.run(main())