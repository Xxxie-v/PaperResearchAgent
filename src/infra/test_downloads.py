import asyncio
import json
import logging
from typing import Optional, Dict, Any, List

from src.infra.redis_runtime import get_redis_json, get_pending
from src.core.config import config
from src.knowledge.knowledge import knowledge_base
from autogen_agentchat.agents import AssistantAgent
from pydantic import BaseModel, Field, field_validator

from src.utils.log_utils import setup_logger
from src.core.prompts import reading_agent_prompt
from src.core.model_client import create_reading_model_client
from src.core.state_models import BackToFrontData, State, ExecutionState
from src.services.chroma_client import ChromaClient
from src.knowledge.knowledge import knowledge_base
from src.core.config import config
from src.infra.redis_runtime import get_pending
from src.core.prompts import analyse_agent_prompt
from src.core.state_models import ExtractedPaperData

logger = logging.getLogger(__name__)


model_client = create_reading_model_client()
analyse_agent = AssistantAgent(
    name="analyse_agent",
    model_client=model_client,
    system_message=analyse_agent_prompt,
    model_client_stream=True
)

async def analyse_node():
    """从临时知识库读取论文并生成总结报告"""
    
    # 获取临时知识库 db_id
    tmp_db_id = config.get("tmp_db_id")

    if not tmp_db_id:
        logger.error("没有找到临时知识库 db_id，请先保存论文到临时知识库")
        return
    logger.info(f"从临时知识库 {tmp_db_id} 读取所有论文文档")
    # 1. 从知识库读取所有论文文档
    
    docs = knowledge_base.get_kb(tmp_db_id)
    
    db_info = docs.get_database_info(tmp_db_id)
    
    collection = await docs._get_chroma_collection(tmp_db_id)
    all_docs = collection.get(include=["documents", "metadatas"])
    
    task_payload = {
        "papers": all_docs
    }
    
    logger.info(f"[Job {tmp_db_id}] 分析节点开始执行")
    try:
        result = await analyse_agent.run(task=json.dumps(task_payload))
    except Exception as e:
        logger.error(f"调用 analyse_agent 失败: {e}")
        return
    
    # 6. 处理结果
    if hasattr(result, "model_dump"):
        parsed_result = result.model_dump()
    else:
        parsed_result = result

    # 7. 输出分析报告
    content = parsed_result["messages"][-1]["content"]
    print("\n===== 分析报告 =====\n")
    print(content)
    
  
    

# ====== 工具函数：保存 job 解析结果到 KB ======
async def save_job_results_to_kb(job_id: str):
    """
    从 Redis 读取 job 聚合的解析结果，并保存到知识库
    """
    r_json = get_redis_json()
    job_key = f"job:{job_id}:parsed_results"

    # 轮询等待 job 所有 task 完成
    while True:
        pending = await get_pending(r_json, job_id)
        if pending is None:
            logger.warning(f"[Job {job_id}] pending_count 不存在，继续等待...")
            await asyncio.sleep(5)
            continue

        pending = int(pending)
        if pending > 0:
            logger.info(f"[Job {job_id}] 还有 {pending} 个 task 未完成，等待中...")
            await asyncio.sleep(5)
            continue
        else:
            logger.info(f"[Job {job_id}] 所有 task 已完成")
            break

    # 读取聚合结果
    papers_raw = await r_json.get(job_key)
    if not papers_raw:
        logger.warning(f"[Job {job_id}] Redis 中没有解析结果")
        return

    try:
        papers: List[Dict[str, Any]] = json.loads(papers_raw)
    except json.JSONDecodeError:
        logger.error(f"[Job {job_id}] Redis 数据 JSON 解析失败")
        return

    # 保存到知识库
    embedding_dic = config.get("default-embedding-model")
    embedding_provider = embedding_dic.get("model-provider")
    provider_dic = config.get(embedding_provider)

    embed_info = {
        "name": embedding_dic.get("model"),
        "dimension": embedding_dic.get("dimension"),
        "base_url": provider_dic.get("base_url"),
        "api_key": provider_dic.get("api_key"),
    }

    kb_type = config.get("KB_TYPE")
    database_info = await knowledge_base.create_database(
        "临时知识库",
        "用于存储临时提取的论文数据，仅用于本次报告的生成，用完即删",
        kb_type=kb_type,
        embed_info=embed_info,
        llm_info=None,
    )
    db_id = database_info["db_id"]
    config.set("tmp_db_id", db_id)

    # 将每篇论文序列化
    documents = [json.dumps(p, ensure_ascii=False) for p in papers]
    ids = [str(i) for i in range(len(documents))]

    await knowledge_base.add_processed_content(db_id, {
        "documents": documents,
        "ids": ids,
    })

    logger.info(f"[Job {job_id}] 成功将 {len(papers)} 篇论文保存到知识库 (db_id={db_id})")
    await analyse_node()
    
    


# ====== 测试运行 ======
if __name__ == "__main__":
    test_job_id = "203"

    asyncio.run(save_job_results_to_kb(test_job_id))