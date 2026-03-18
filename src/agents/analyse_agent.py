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
from src.core.prompts import analyse_agent_prompt
from src.core.state_models import ExtractedPaperData

logger = setup_logger(__name__)

model_client = create_reading_model_client()
analyse_agent = AssistantAgent(
    name="analyse_agent",
    model_client=model_client,
    system_message=analyse_agent_prompt,
    model_client_stream=True
)

async def analyse_node(state: State)-> State:
    """从临时知识库读取论文并生成总结报告"""
    state_queue = state["state_queue"]
    current_state = state["value"]
    current_state.current_step = ExecutionState.ANALYZING
    await state_queue.put(BackToFrontData(step=ExecutionState.ANALYZING,state="initializing",data=None))
    # 获取临时知识库 db_id
    tmp_db_id = config.get("tmp_db_id")

    if not tmp_db_id:
        logger.error("没有找到临时知识库 db_id，请先保存论文到临时知识库")
        return
    logger.info(f"从临时知识库 {tmp_db_id} 读取所有论文文档")
    # 1. 从知识库读取所有论文文档
    
    docs = knowledge_base.get_kb(tmp_db_id)
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
    if parsed_result["messages"]:
        content = parsed_result["messages"][-1]["content"]
        logger.info("分析成功")
    else:
        logger.warning("分析结果为空")
        content = "分析结果为空"

    await state_queue.put(BackToFrontData(step=ExecutionState.ANALYZING,state="completed",data=f"分析完成，报告：{content}"))

    return {"value": current_state}


if __name__ == "__main__":
    asyncio.run(analyse_node())
