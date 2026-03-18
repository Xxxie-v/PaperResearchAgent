from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from src.agents.userproxy_agent import WebUserProxyAgent,userProxyAgent
from pydantic import BaseModel, Field
from typing import Optional,List
import re
import ast
from datetime import datetime, timedelta
from src.utils.log_utils import setup_logger
from src.tasks.paper_search import PaperSearcher
from src.core.state_models import State,ExecutionState
from src.core.prompts import search_agent_prompt
from src.core.state_models import BackToFrontData
from src.tasks.download_papers import submit_download_tasks
from src.core.model_client import create_search_model_client
from src.infra.redis_runtime import incr_pending,get_pending
import asyncio

logger = setup_logger(__name__)


model_client = create_search_model_client()


# 创建一个查询条件类，包括查询内容、主题、时间范围等信息，用于存储用户的查询需求
class SearchQuery(BaseModel):
    """查询条件类，存储用户查询需求"""
    querys: List[str] = Field(default=list, description="查询条件列表")
    start_date: Optional[str] = Field(default=None, description="开始时间, 格式: YYYY-MM-DD")
    end_date: Optional[str] = Field(default=None, description="结束时间, 格式: YYYY-MM-DD")

search_agent = AssistantAgent(
    name="search_agent",
    model_client=model_client,
    system_message=search_agent_prompt,
    output_content_type=SearchQuery
)

def parse_search_query(s: str) -> SearchQuery:
    """将前端传回的字符串转为 SearchQuery 对象"""
    # 提取 querys（使用 ast.literal_eval 保证安全）
    querys_match = re.search(r"querys\s*=\s*(\[[^\]]*\])", s)
    start_match = re.search(r"start_date\s*=\s*'([^']*)'", s)
    end_match = re.search(r"end_date\s*=\s*'([^']*)'", s)

    querys = []
    if querys_match:
        try:
            querys = ast.literal_eval(querys_match.group(1))
        except Exception:
            querys = []

    start_date = start_match.group(1) if start_match else None
    end_date = end_match.group(1) if end_match else None

    return SearchQuery(querys=querys, start_date=start_date, end_date=end_date)

async def search_node(state: State) -> State:
    
    """搜索论文节点"""
    state_queue = None
    logger.info(f"[Job {state['state_queue'].job_id}] 搜索节点开始执行")
    try:
        state_queue = state["state_queue"]
        current_state = state["value"]
        current_state.current_step = ExecutionState.SEARCHING
        await state_queue.put(BackToFrontData(step=ExecutionState.SEARCHING,state="initializing",data=None))

        prompt = f"""
            用户输入:{current_state.user_request}
            基准时间:{datetime.now().strftime("%Y-%m-%d")}
            """
        response = await search_agent.run(task = prompt)
        
        search_query = response.messages[-1].content
        
        #await state_queue.put(BackToFrontData(step=ExecutionState.SEARCHING,state="user_review",data=f"{search_query}"))
        # logger.info("send to human review...") 
        
        # result = await userProxyAgent.on_messages(
        #     [TextMessage(content="请人工审核：查询条件是否符合？", source="AI")],
        #     cancellation_token=CancellationToken()
        # )
        
        result = TextMessage(content=str(search_query), source="fallback")
        # logger.info("human replied: %s", result.content)
        search_query = parse_search_query(result.content)

        # 调用检索服务
    
        paper_searcher = PaperSearcher()
        results = await paper_searcher.search_papers(
            querys = search_query.querys,
            start_date = search_query.start_date,
            end_date = search_query.end_date,
        )
        #20篇论文 list [{'paper_id': '2411.15594v6', 'title': 'A Survey on LLM-as-a-Judge', 'authors': ['Jiawei Gu', 'Xuhui Jiang', 'Zhichao Shi', 'Hexiang Tan', 'Xuehao Zhai', 'Chengjin Xu', 'Wei Li', 'Yinghan Shen', 'Shengjie Ma', 'Honghao Liu', 'Saizhuo Wang', 'Kun Zhang', 'Yuanzhuo Wang', 'Wen Gao', 'Lionel Ni', 'Jian Guo'], 'summary': 'Accurate and consistent evaluation is crucial for decision-making across numerous fields, yet it remains a challenging task due to inherent subjectivity, variability, and scale. Large Language Models (LLMs) have achieved remarkable success across diverse domains, leading to the emergence of "LLM-as-a-Judge," where LLMs are employed as evaluators for complex tasks. With their ability to process diverse data types and provide scalable, cost-effective, and consistent assessments, LLMs present a compelling alternative to traditional expert-driven evaluations. However, ensuring the reliability of LLM-as-a-Judge systems remains a significant challenge that requires careful design and standardization. This paper provides a comprehensive survey of LLM-as-a-Judge, addressing the core question: How can reliable LLM-as-a-Judge systems be built? We explore strategies to enhance reliability, including improving consistency, mitigating biases, and adapting to diverse assessment scenarios. Additionally, we propose methodologies for evaluating the reliability of LLM-as-a-Judge systems, supported by a novel benchmark designed for this purpose. To advance the development and real-world deployment of LLM-as-a-Judge systems, we also discussed practical applications, challenges, and future directions. This survey serves as a foundational reference for researchers and practitioners in this rapidly evolving field.', 'published': 2024, 'published_date': '2024-11-23T16:03:35+00:00', 'url': 'http://arxiv.org/abs/2411.15594v6', 'pdf_url': 'https://arxiv.org/pdf/2411.15594v6', 
        # 'primary_category': 'cs.CL', 'categories': ['cs.CL', 'cs.AI'], 'doi': None, 'score': 0.712576687335968, 'citations': 1092} 。。。。。]       
       
        current_state.search_results = results
        logger.info(f"开始下载 {len(results)} 篇论文")

        await submit_download_tasks(results,state_queue)
        r = state_queue.r
        pending = await get_pending(r, state_queue.job_id)
        while pending is None or int(pending) > len(results):
            logger.info(f"[Job {state_queue.job_id}] 还有 {pending-len(results)} 个 下载未完成，等待中...")
            await asyncio.sleep(5)  # 可调整等待间隔
            pending = await get_pending(r, state_queue.job_id)
        logger.info(f"[Job {state_queue.job_id}] 所有论文下载完成,开始解析")
        
        if len(results) > 0:
            await state_queue.put(BackToFrontData(step=ExecutionState.SEARCHING,state="completed",data=f"论文搜索完成，返回最相关的 {len(results)} 篇论文"))
        else:
            await state_queue.put(BackToFrontData(step=ExecutionState.SEARCHING,state="error",data="没有找到相关论文,请尝试其他查询条件"))
            current_state.error.search_node_error = "没有找到相关论文,请尝试其他查询条件"
            
        
        return {"value": current_state}
    
    
    
            
    except Exception as e:
        err_msg = f"Search failed: {str(e)}"
        state["value"].error.search_node_error = err_msg
        await state_queue.put(BackToFrontData(step=ExecutionState.SEARCHING,state="error",data=err_msg))
        return state