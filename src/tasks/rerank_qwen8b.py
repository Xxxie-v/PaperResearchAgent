import requests
import asyncio
import aiohttp
import os
from typing import Any, Dict, List, Optional, Union
import re
from serpapi import GoogleSearch

class SiliconFlowRerankError(RuntimeError):
    """SiliconFlow rerank API 调用失败时抛出。"""


def build_documents(papers: List[Dict[str, any]], sep: str = "\n") -> List[str]:
    """根据论文列表构造 rerank 文档串，拼接 title 与 summary 并保持索引对应。

    参数:
        papers: 论文列表，元素包含 title 与 summary。
        sep: 标题与摘要之间的分隔符。

    返回:
        与 papers 一一对应的文档字符串列表。
    """
    documents = []
    for p in papers:
        title = (p.get("title") or "").strip()
        summary = (p.get("summary") or "").strip()
        doc = f"{title}{sep}{summary}".strip()
        documents.append(doc)
    return documents

def to_rerank_query(querys):
    """将查询列表或字符串清洗为可用于重排序的纯文本查询。

    参数:
        querys: 查询字符串或查询列表。

    返回:
        清洗后的查询文本。
    """
    # 1) list -> str
    if isinstance(querys, list):
        q = querys[0] if len(querys) == 1 else " OR ".join(f"({x})" for x in querys)
    else:
        q = str(querys)

    # 2) 反转义
    q = q.replace(r"\"", '"')

    # 3) 去掉布尔操作符与括号
    q = re.sub(r"\b(AND|OR|NOT)\b", " ", q, flags=re.IGNORECASE)
    q = q.replace("(", " ").replace(")", " ")

    # 4) 去掉多余引号（保留短语内容即可）
    q = q.replace('"', ' ')

    # 5) 压缩空白
    q = re.sub(r"\s+", " ", q).strip()
    return q

def rerank_bucket(papers, eps=0.05):
    """按 score 分桶并在桶内按 citations 排序，eps 为同桶的分数阈值。

    参数:
        papers: 已包含 score 与 citations 的论文列表。
        eps: 分数差小于等于该值的论文归为同桶。

    返回:
        排序后的论文列表。
    """
    papers.sort(key=lambda x: x["score"], reverse=True)
    out, bucket = [], [papers[0]] if papers else []
    for p in papers[1:]:
        if abs(p["score"] - bucket[-1]["score"]) <= eps:
            bucket.append(p)
        else:
            bucket.sort(key=lambda x: x.get("citations", 0), reverse=True)
            out.extend(bucket)
            bucket = [p]
    if bucket:
        bucket.sort(key=lambda x: x.get("citations", 0), reverse=True)
        out.extend(bucket)
    return out

def norm(s: str) -> str:
    """将文本标准化为小写、去标点并压缩空白，便于相似度计算。

    参数:
        s: 原始文本。

    返回:
        规范化后的文本。
    """
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)  # 去标点
    return s

def simple_similarity(a: str, b: str) -> float:
    """基于词集合的重叠率计算粗粒度相似度，空集合返回 0。

    参数:
        a: 文本 A。
        b: 文本 B。

    返回:
        相似度分数（0-1）。
    """
    # token overlap ratio（很粗，但比全等强）
    A = set(norm(a).split())
    B = set(norm(b).split())
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

async def _fetch_serpapi(
    session: aiohttp.ClientSession,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """调用 SerpAPI Scholar 接口并返回 JSON 响应。

    参数:
        session: aiohttp 会话对象。
        params: SerpAPI 请求参数。

    返回:
        SerpAPI 返回的 JSON 字典。
    """
    async with session.get(os.getenv("SERPAPI_URL"), params=params) as resp:
        resp.raise_for_status()
        return await resp.json()

async def _fill_one_citation(
    sem: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    paper: Dict[str, Any],
    *,
    api_key: str,
    max_check: int = 5,
    sim_threshold: float = 0.8,
) -> None:
    """为单篇论文查询 Scholar 并填充 citations，基于相似度筛选结果。

    参数:
        sem: 并发控制信号量。
        session: aiohttp 会话对象。
        paper: 单篇论文信息字典。
        api_key: SerpAPI 的 API Key。
        max_check: 参与匹配的候选结果数量上限。
        sim_threshold: 标题相似度阈值。

    返回:
        None。
    """
    # 规范化标题并初始化引用次数
    title = (paper.get("title") or "").strip()
    paper["citations"] = 0
    if not title:
        return

    # 构造 Google Scholar 查询参数
    params = {
        "engine": "google_scholar",
        "q": f"\"{title}\"",
        "hl": "en",
        "api_key": api_key,
        "num": 10,
    }

    # 在并发控制下请求 SerpAPI
    try:
        async with sem:
            data = await _fetch_serpapi(session, params)
    except Exception:
        return

    organic_results = data.get("organic_results", [])
    if not isinstance(organic_results, list) or not organic_results:
        return

    best_item = None
    best_sim = 0.0
    # 在前 max_check 条结果中挑相似度最高的标题
    for item in organic_results[:max_check]:
        sim = simple_similarity(title, item.get("title", ""))
        if sim > best_sim:
            best_sim = sim
            best_item = item

    # 相似度达标才写入引用次数
    if best_item and best_sim >= sim_threshold:
        paper["citations"] = int(
            (((best_item.get("inline_links") or {}).get("cited_by") or {}).get("total") or 0)
        )


async def add_citations_async(
    papers: List[Dict[str, Any]],
    *,
    concurrency: int = 3,
    timeout_s: int = 30,
    max_check: int = 5,
    sim_threshold: float = 0.8,
) -> List[Dict[str, Any]]:
    """并发批量补全论文 citations，控制并发与超时并复用会话。

    参数:
        papers: 论文列表。
        concurrency: 并发请求数量。
        timeout_s: 请求超时秒数。
        max_check: 参与匹配的候选结果数量上限。
        sim_threshold: 标题相似度阈值。

    返回:
        补全 citations 后的论文列表。
    """
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise RuntimeError("SERPAPI_API_KEY is not set")

    # 控制并发与请求超时
    sem = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=timeout_s)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            _fill_one_citation(
                sem, session, p,
                api_key=api_key,
                max_check=max_check,
                sim_threshold=sim_threshold,
            )
            for p in papers
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    return papers


async def siliconflow_rerank_async(
    *,
    token: Optional[str] = None,
    model: str = "Qwen/Qwen3-Reranker-8B",
    base_url: str = os.getenv("RERANK_URL"),
    query: str,                                                                                                                        
    documents: List[str],
    top_n: Optional[int] = None,
    timeout_s: int = 60,
) -> Dict[str, Any]:
    """调用 SiliconFlow rerank 接口，组装请求并返回 JSON 结果。

    参数:
        token: SiliconFlow API Token，可为空并从环境变量读取。
        model: 重排模型名称。
        base_url: 接口地址。
        query: 查询文本。
        documents: 待排序的文档列表。
        top_n: 返回结果数量上限。
        timeout_s: 请求超时秒数。

    返回:
        接口响应的 JSON 字典。
    """

    token = token or os.getenv("SILICONFLOW_API_KEY")
    if not token:
        raise ValueError("token 未提供，且环境变量 SILICONFLOW_API_KEY 不存在")
    
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload: Dict[str, Any] = {"model": model, "query": query, "documents": documents}
    if top_n is not None:
        payload["top_n"] = int(top_n)

    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as sess:
        async with sess.post(base_url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            return await resp.json()



async def siliconflow_rerank_documents_async(
    querys: Any,
    papers: List[Dict[str, Any]],
    *,
    token: Optional[str] = None,
    model: str = "Qwen/Qwen3-Reranker-8B",
    base_url: str = os.getenv("RERANK_URL"),
    top_k: Optional[int] = None,
    threshold: Optional[float] = None,
    use_cite_rerank: bool = False,
    bucket_eps: float = 0.05,
    concurrency: int = 3,
) -> List[Dict[str, Any]]:
    """构造查询与文档进行重排，按阈值与 top_k 过滤并可选引用二次排序。

    参数:
        querys: 查询字符串或查询列表。
        papers: 论文列表。
        token: SiliconFlow API Token。
        model: 重排模型名称。
        base_url: 接口地址。
        top_k: 最多返回的结果数量。
        threshold: 相关性分数阈值。
        use_cite_rerank: 是否按引用数二次排序。
        bucket_eps: 分桶时的分数阈值。
        concurrency: 引用查询并发数。

    返回:
        重排后的论文列表。
    """
    if not papers:
        return []

    # 统一查询为适合模型的文本，并构造文档字符串
    query = to_rerank_query(querys)
    documents = build_documents(papers)

    # 调用重排序模型，返回每个文档的相关性得分
    data = await siliconflow_rerank_async(
        token=token,
        model=model,
        base_url=base_url,
        query=query,
        documents=documents,
        top_n=(len(documents) if top_k is None else int(top_k)),
    )

    # 解析 results 字段并做结构校验
    results = data.get("results")
    if not isinstance(results, list):
        raise SiliconFlowRerankError(f"无法解析 results 字段，响应为: {data}")

    out: List[Dict[str, Any]] = []
    for item in results:
        # 拉取原始索引与相关性分数
        idx = item.get("index")
        score = item.get("relevance_score")
        if not isinstance(idx, int) or score is None:
            continue

        # 阈值过滤并写回 score
        score_f = float(score)
        if threshold is not None and score_f < float(threshold):
            continue

        paper = papers[idx]
        paper["score"] = score_f
        out.append(paper)

        # 提前满足 top_k 即停止
        if top_k is not None and len(out) >= int(top_k):
            break

    # 可选基于引用数做二次排序
    # if use_cite_rerank and out:        # 只保留前 M
    #     await add_citations_async(out, concurrency=concurrency)
    #     out = rerank_bucket(out, eps=bucket_eps)

    return out




# def siliconflow_rerank(
#     query: str,
#     documents: List[str],
#     *,
#     token: Optional[str] = None,
#     model: str = "Qwen/Qwen3-Reranker-8B",
#     base_url: str = os.getenv("RERANK_URL"),
#     top_n: Optional[int] = None,
#     extra_payload: Optional[Dict[str, Any]] = None,
#     session: Optional[requests.Session] = None,
# ) -> Dict[str, Any]:
#     """
#     调用 SiliconFlow Rerank API，对 documents 进行相关性重排。

#     参数:
#         query: 查询文本
#         documents: 候选文档列表（字符串列表）
#         token: API token；不传则尝试读取环境变量 SILICONFLOW_API_KEY
#         model: rerank 模型名
#         base_url: rerank endpoint
#         top_n: 可选，限制返回的结果数量（若接口支持该字段）
#         timeout: 超时设置；可传单个 float 或 (connect, read)
#         extra_payload: 额外 payload 字段（用于兼容接口扩展）
#         session: 可选 requests.Session，用于连接复用

#     返回:
#         解析后的 JSON 字典（包含原始字段，如 results 等）

#     异常:
#         SiliconFlowRerankError: 非 2xx 或返回非 JSON / JSON 中包含错误信息
#         ValueError: 输入参数不合法
#     """
#     if not query or not query.strip():
#         raise ValueError("query 不能为空")
#     if not isinstance(documents, list) or not documents:
#         raise ValueError("documents 必须是非空 list")
#     if any(not isinstance(d, str) for d in documents):
#         raise ValueError("documents 必须是 string 列表")

#     token = token or os.getenv("SILICONFLOW_API_KEY")
#     if not token:
#         raise ValueError("token 未提供，且环境变量 SILICONFLOW_API_KEY 不存在")

#     payload: Dict[str, Any] = {
#         "model": model,
#         "query": query,
#         "documents": documents,
#     }

#     # 有些 rerank 接口支持 top_n / top_k；不确定官方是否支持时，用 extra_payload 更稳
#     if top_n is not None:
#         payload["top_n"] = int(top_n)

#     if extra_payload:
#         payload.update(extra_payload)

#     headers = {
#         "Authorization": f"Bearer {token}",
#         "Content-Type": "application/json",
#     }

#     try:
#         resp = requests.post(base_url, json=payload, headers=headers)
#     except requests.RequestException as e:
#         raise SiliconFlowRerankError(f"请求失败: {e}") from e

#     # 非 2xx 直接报错并带上响应文本
#     if not (200 <= resp.status_code < 300):
#         raise SiliconFlowRerankError(
#             f"HTTP {resp.status_code}: {resp.text}"
#         )

#     # 尝试解析 JSON
#     try:
#         data = resp.json()
#     except ValueError as e:
#         raise SiliconFlowRerankError(f"响应不是合法 JSON: {resp.text}") from e

#     return data

# def add_citations(papers: List[Dict[str, any]]) -> List[Dict[str, any]]:
#     """
#     为论文列表添加引用数字段。
#     """

#     for paper in papers:
        
#         title = paper.get("title", "")
#         paper["citations"] = 0 
        
#         # 调用 SerpAPI 搜索标题
#         params = {
#         "engine": "google_scholar",
#         "q":f'"{title}"',
#         "hl": "en",
#         "api_key": os.getenv("SERPAPI_API_KEY"),
#         }
#         search = GoogleSearch(params)
#         results = search.get_dict()
#         # 解析返回结果organic_results
#         organic_results = results["organic_results"]
        
#         best_item = None
#         best_sim = 0.0
#         for item in organic_results[:5]:
#             # 计算标题相似度
#             sim = simple_similarity(title, item.get("title", ""))
#             if sim > best_sim:
#                 best_sim = sim
#                 best_item = item

#         if best_item and best_sim >= 0.8:  # 阈值可调
#             paper["citations"] = int((((best_item.get("inline_links") or {}).get("cited_by") or {}).get("total") or 0))
        
#     return papers

# def siliconflow_rerank_documents(
#     querys: str,
#     papers: List[Dict[str, any]],
#     *,
#     top_k: Optional[int] = None,
#     threshold: Optional[float] = None,
#     use_cite_rerank: bool = False,
#     **kwargs: Any,
    
# ) -> List[Dict[str, Any]]:
#     """
#     返回已按相关性排序的top_k个文档列表（来自返回体里的 results 顺序）。
#     """
    
#     query = to_rerank_query(querys)
#     documents = build_documents(papers)
    
#     data = siliconflow_rerank(query, documents, **kwargs)
#     results = data.get("results")
#     if not isinstance(results, list):
#         raise SiliconFlowRerankError(f"无法解析 results 字段，响应为: {data}")

#     out: List[Dict[str, Any]] = []
#     for rank, item in enumerate(results, start=1):
#         idx = item.get("index")
#         score = item.get("relevance_score")

#         if not isinstance(idx, int):
#             continue
#         if score is None:
#             continue
#         score_f = float(score)

#         if threshold is not None and score_f < float(threshold):
#             continue
        
#         paper = papers[idx]
#         paper["score"] = score_f
        
#         out.append(paper)

#         if top_k is not None and len(out) >= int(top_k):
#             break
        
#     # 如果按引用数排序，添加引用数字段并排序
#     if use_cite_rerank:
#         out = add_citations(out)
#         out = rerank_bucket(out)
#     return out

