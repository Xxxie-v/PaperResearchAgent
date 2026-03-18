import asyncio
import json
import re
import ast
from typing import List, Optional, Dict, Any

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

logger = setup_logger(__name__)

# ====== Pydantic 数据模型 ======
class KeyMethodology(BaseModel):
    name: Optional[str] = Field(default=None, description="方法名称")
    principle: Optional[str] = Field(default=None, description="核心原理")
    novelty: Optional[str] = Field(default=None, description="创新点")

class ExtractedPaperData(BaseModel):
    core_problem: str = Field(default="", description="核心问题")
    key_methodology: KeyMethodology = Field(default=None, description="关键方法")
    datasets_used: List[str] = Field(default=[], description="使用的数据集")
    evaluation_metrics: List[str] = Field(default=[], description="评估指标")
    main_results: str = Field(default="", description="主要结果")
    limitations: str = Field(default="", description="局限性")
    contributions: List[str] = Field(default=[], description="贡献")

    @field_validator("datasets_used", "evaluation_metrics", "contributions", mode="before")
    @classmethod
    def _validate_list_fields(cls, v):
        if v is None or v == "":
            return []
        if isinstance(v, str):
            return [v]
        return v

    @field_validator("core_problem", "main_results", "limitations", mode="before")
    @classmethod
    def _validate_str_fields(cls, v):
        if v is None:
            return ""
        return str(v)

class ExtractedPapersData(BaseModel):
    papers: List[ExtractedPaperData] = Field(default=[], description="提取的论文数据列表")

# ====== 初始化阅读 Agent ======
model_client = create_reading_model_client()
read_agent = AssistantAgent(
    name="read_agent",
    model_client=model_client,
    system_message=reading_agent_prompt,
    output_content_type=ExtractedPaperData,
    model_client_stream=True
)

# ====== 工具函数 ======
def sanitize_metadata(paper: Dict[str, Any]) -> Dict[str, Any]:
    new_meta = {}
    for k, v in paper.items():
        if v is None:
            continue
        if isinstance(v, list):
            new_meta[k] = ", ".join(str(x) for x in v)
        elif isinstance(v, dict):
            new_meta[k] = json.dumps(v, ensure_ascii=False)
        else:
            new_meta[k] = v
    return new_meta

async def add_papers_to_kb(papers: Optional[List[Dict[str, Any]]]):
    """将提取的论文数据添加到知识库"""
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

    documents = [json.dumps(p, ensure_ascii=False) for p in papers]

    ids = [str(i) for i in range(len(documents))]

    await knowledge_base.add_processed_content(db_id, {
        "documents": documents,
        "ids": ids,
    })

# ====== 阅读节点 ======
#修改阅读节点为，读取job中的任务是否都完成，若完成，则开始处理
async def reading_node(state: State) -> State:
    """处理 Redis 中论文信息，调用阅读 Agent 生成报告"""
    state_queue = state["state_queue"]
    current_state = state["value"]
    current_state.current_step = ExecutionState.READING
    job_id = state_queue.job_id
    job_key = f"job:{job_id}:parsed_results"
    r_json = state_queue.r

    await state_queue.put(BackToFrontData(step=ExecutionState.READING, state="initializing", data=None))
    logger.info(f"[Job {job_id}] 开始读取论文数据")
    while True:
        pending = await get_pending(r_json, job_id)
        if pending is None:
            logger.warning(f"[Job {job_id}] pending_count 不存在，继续等待")
            await asyncio.sleep(10)
            continue

        pending = int(pending)
        if pending > 0:
            logger.info(f"[Job {job_id}] 还有 {pending} 个 task 未完成，等待中...")
            await asyncio.sleep(10)  # 可调整等待间隔
            continue
        else:
            logger.info(f"[Job {job_id}] 所有 task 已完成")
            break
        
    logger.info(f"[Job {job_id}] 开始保存到数据库")
    papers = await r_json.get(job_key)
    
    try:
        papers = json.loads(papers)
    except json.JSONDecodeError:
        logger.error(f"[Job {job_id}] Redis 数据 JSON 解析失败")
        return
    
    await add_papers_to_kb(papers)
    logger.info(f"[Job {job_id}] 论文数据已保存到数据库")
    await state_queue.put(BackToFrontData(step=ExecutionState.READING, state="completed", data="读取完成，已保存到知识库"))

    return {"value": current_state}
