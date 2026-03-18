from time import sleep
from src.utils.log_utils import setup_logger
from src.utils.tool_utils import handlerChunk
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse
from src.agents.userproxy_agent import WebUserProxyAgent, userProxyAgent
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from src.knowledge.knowledge_router import knowledge
from fastapi import APIRouter
import json
import asyncio
from src.core.state_models import BackToFrontData
import uuid

from src.infra.redis_runtime import get_redis_json, submit_job, stream_events
# 设置日志
logger = setup_logger(name='main', log_file='project.log')

app = FastAPI()
app.include_router(knowledge)
# === CORS 配置（开发时可用 "*"，生产请限定具体域名） ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

state_queue = asyncio.Queue()

# agent = WebUserProxyAgent("user_proxy")

@app.post("/send_input")
async def send_input(data: dict):
    user_input = data.get("input")
    userProxyAgent.set_user_input(user_input)
    return JSONResponse({"status": 200, "msg": "已收到人工输入"})

# @app.get('/api/research')
# async def research_stream(query: str):
#     from src.agents.orchestrator import PaperAgentOrchestrator
#     from src.core.state_models import State,ExecutionState
#     async def event_generator():
#         while True:
#             state = await state_queue.get()
#             yield {"data": f"{state.model_dump_json()}"}
    
#     # 启动事件生成器（此时已开始监听队列）
#     event_source = EventSourceResponse(event_generator(), media_type="text/event-stream")

#     # 初始化业务流程控制器
#     orchestrator = PaperAgentOrchestrator(state_queue = state_queue)
    
#     # 启动异步任务
#     asyncio.create_task(orchestrator.run(user_request=query))

#     return event_source

@app.get("/api/research")
async def research_stream(query: str, user_id: str = "default"):
    r = get_redis_json()
    user_id = str(uuid.uuid4())

    job = await submit_job(r, user_id=user_id, query=query, max_papers=50)

    async def event_generator():
        # 先告诉前端 job_id
        yield {"data": json.dumps({"type": "job", "job_id": job.job_id}, ensure_ascii=False)}

        async for data in stream_events(r, job.job_id, start_id="0-0"):
            yield {"data": data}

            # 可选：如果你 BackToFrontData 里 finished/failed 有固定字段，可以在这里 break
            # 这里不强依赖字段，避免误判。需要的话你把 BackToFrontData 格式贴我，我给你加上。

    return EventSourceResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    