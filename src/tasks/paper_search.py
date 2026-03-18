import arxiv
import logging
from typing import List, Dict, Optional, Union
from datetime import datetime, timedelta
from time import timezone
from src.utils.log_utils import setup_logger
import re
import os
from rank_bm25 import BM25Okapi
from src.tasks.rerank_qwen8b import siliconflow_rerank_documents_async

logger = setup_logger(__name__)



class PaperSearcher:
    """论文搜索器，使用arxiv库搜索论文"""
    
    def __init__(self):
        """初始化论文搜索器"""
        
        pass
    
    async def search_papers(self, 
                      querys: List[str], 
                      max_results: int = 100, 
                      sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance, 
                      sort_order: arxiv.SortOrder = arxiv.SortOrder.Descending, 
                      start_date: Optional[Union[str, datetime]] = None, 
                      end_date: Optional[Union[str, datetime]] = None) -> List[Dict]:
        """
        搜索arXiv论文
        
        参数:
            querys: 搜索关键词
            max_results: 最大返回结果数量
            sort_by: 排序方式 (Relevance, LastUpdatedDate, SubmittedDate)
            sort_order: 排序顺序 (Ascending, Descending)
            start_date: 开始日期，可以是字符串(YYYY-MM-DD)或datetime对象
            end_date: 结束日期，可以是字符串(YYYY-MM-DD)或datetime对象
        
        返回:
            论文列表，每项包含论文的详细信息
        """
        # querys = ['artificial intelligence', 'AI', 'llm', 'machine learning', 'deep learning']
        try:
            # 构建搜索查询
            search_query=querys

            # 添加日期范围过滤（submittedDate 使用 UTC/GMT，精确到分钟）
            if start_date or end_date:
                start_date_str = self._format_date(start_date) if start_date else "190001010000"
                end_date_str = self._format_date(end_date) if end_date else datetime.now().strftime("%Y%m%d2359")
                search_query = f"{search_query} AND submittedDate:[{start_date_str} TO {end_date_str}]"

            logger.info(f"开始搜索论文: query='{search_query}', max_results={max_results}, sort_by={sort_by}")


            logger.info(f"论文搜索查询条件: {search_query}")

            # 创建搜索对象
            try:
                search = arxiv.Search(
                    query=search_query,
                    max_results=max_results,
                    sort_by=sort_by,
                    sort_order=sort_order
                )
            except Exception as e:
                logger.error(f"创建arxiv搜索对象失败: {str(e)}")
                return []
            
            # logger.info(f"论文搜索结果为：{search.results()}")
            # 执行搜索并解析结果
            # 使用新方法格式化论文列表
            papers = self.format_papers_list(search.results())
            
            logger.info(f"论文搜索完成，共找到 {len(papers)} 篇论文")
            
            # #使用BM25模型对论文进行排序
            # logger.info(f"开始使用BM25模型对论文进行排序")
            # rerank_papers = bm25_rerank(papers, querys[0])
            # print(rerank_papers[0])
            
            
            # 使用重排序模型对论文进行排序
            logger.info(f"开始使用重排序模型对论文进行排序")
            rerank_papers = await siliconflow_rerank_documents_async(querys, papers, top_k=5,use_cite_rerank=os.getenv("USE_CITE_RERANK") == "True")
            
            logger.info(f"重排序模型对论文进行排序完成，共找到 {len(rerank_papers)} 篇论文")
            
            return rerank_papers
        except Exception as e:
            logger.error(f"论文搜索失败: {str(e)}")
            raise
    

    
    def format_papers_list(self, search_results) -> List[Dict]:
        """
        将搜索结果（迭代器或列表）格式化为论文信息字典列表
        
        参数:
            search_results: arxiv搜索结果对象（可能是迭代器）
        
        返回:
            格式化后的论文信息字典列表
        """
        # 将迭代器转换为列表以便后续处理
        results_list = list(search_results)
        
        # 格式化论文列表
        formatted_papers = [self._parse_paper_result(result) for result in results_list]
        
        logger.info(f"开始格式化论文列表，共 {len(results_list)} 篇论文")
        return formatted_papers

    def search_by_author(self, 
                        author_name: str, 
                        limit: int = 10) -> List[Dict]:
        """
        按作者搜索论文
        
        参数:
            author_name: 作者姓名
            limit: 返回结果数量限制
        
        返回:
            论文列表
        """
        logger.info(f"按作者搜索论文: author='{author_name}', limit={limit}")
        
        # 使用作者字段搜索
        query = f"au:{author_name}"
        return self.search_papers(
            query=query,
            max_results=limit,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
    
    def _parse_paper_result(self, result: arxiv.Result) -> Dict:
        """
        解析arXiv搜索结果
        
        参数:
            result: arxiv.Result对象
        
        返回:
            包含论文信息的字典
        """
        # 从结果URL中提取论文ID
        paper_id = result.get_short_id()
        
        # 提取发布年份
        published_year = result.published.year if result.published else None
        
        return {
            "paper_id": paper_id,
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "summary": result.summary,
            "published": published_year,
            "published_date": result.published.isoformat() if result.published else None,
            "url": result.entry_id,
            "pdf_url": result.pdf_url,
            "primary_category": result.primary_category,
            "categories": result.categories,
            "doi": result.doi if hasattr(result, 'doi') else None
        }
    
    def  _format_date(self, date: Union[str, datetime]) -> str:
        """
        输入:
        - datetime
        - str: 'YYYY-MM-DD' / 'YYYYMMDD'
                可选带时间：'YYYY-MM-DD HH:MM' / 'YYYYMMDDHHMM' / 'YYYY-MM-DD HHMM'
        输出:
        - 'YYYYMMDDHHMM'
        约定:
        - 没给时间默认 0000
        - datetime 不带 tz 的按其自身值格式化（不做时区转换）
        """
        if isinstance(date, datetime):
            return date.strftime("%Y%m%d%H%M")

        s = str(date).strip()
        if not s:
            return datetime.now().strftime("%Y%m%d%H%M")

        # 统一分隔符
        s2 = s.replace("/", "-").replace(".", "-")

        fmts = [
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d %H%M",
            "%Y%m%d%H%M",
            "%Y-%m-%d",
            "%Y%m%d",
        ]
        for fmt in fmts:
            try:
                dt = datetime.strptime(s2, fmt)
                # 如果只有日期，补 00:00
                if fmt in ("%Y-%m-%d", "%Y%m%d"):
                    dt = dt.replace(hour=0, minute=0)
                return dt.strftime("%Y%m%d%H%M")
            except ValueError:
                continue

        # 解析不了就退回当前时间（也可以改成 raise ValueError 更早暴露问题）
        return datetime.now().strftime("%Y%m%d%H%M")


# -----------------------------
# 1) 解析 Boolean 表达式 -> 关键词/短语
# -----------------------------
_BOOL_OPS = {"AND", "OR", "NOT"}

def extract_query_terms(boolean_expr: str) -> tuple[List[str], List[str]]:
    """
    输入: 英文 Boolean 检索式，例如:
      transformer AND ("survey" OR "review" OR "overview" OR "tutorial")

    输出:
      - phrases: ["survey", "review", ...]  (来自双引号包裹的短语)
      - terms:   ["transformer", ...]      (其他单词)
    规则:
      - 移除 AND/OR/NOT 和括号
      - 双引号内内容作为 phrase
      - 其余按单词抽取
    """
    s = (boolean_expr or "").strip()
    if not s:
        return [], []

    # 把 \" 还原成 "
    s = s.replace(r"\"", '"')

    # 先抽取双引号短语
    phrases = re.findall(r'"([^"]+)"', s)
    phrases = [p.strip().lower() for p in phrases if p.strip()]

    # 移除短语，避免短语内词被重复抽取
    s_wo_phrases = re.sub(r'"[^"]+"', " ", s)

    # 去掉括号
    s_wo_phrases = s_wo_phrases.replace("(", " ").replace(")", " ")

    # 抽取单词 token（保留连字符 transformer-based）
    raw_tokens = re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?", s_wo_phrases)

    terms = []
    for t in raw_tokens:
        up = t.upper()
        if up in _BOOL_OPS:
            continue
        terms.append(t.lower())

    # 去重但保持顺序
    def uniq(seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    return uniq(phrases), uniq(terms)


# -----------------------------
# 2) 文档分词
# -----------------------------
def tokenize_text(text: str) -> List[str]:
    """
    简单 tokenizer：小写、抽取英文/数字/连字符 token。
    后续想更准可加停用词表、词干化等。
    """
    if not text:
        return []
    text = text.lower()
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", text)


# -----------------------------
# 3) BM25 rerank 主函数
# -----------------------------
def bm25_rerank(
    papers: List[Dict],
    boolean_query: str,
    title_weight: float = 2.0,
    abstract_weight: float = 1.0,
    phrase_boost: float = 0.5,
) -> List[Dict]:
    """
    对 arXiv 召回结果做 BM25 重排序。

    - papers: 你的 arXiv dict 列表（至少包含 title/summary）
    - boolean_query: 上游给的 Boolean 检索式
    - title_weight / abstract_weight: 让标题更重要（常用）
    - phrase_boost: 若短语在原文出现，给额外加分（轻量“短语偏好”）

    返回: 按 rel_score 降序排序后的 papers（同时每篇写入 rel_score）
    """
    phrases, terms = extract_query_terms(boolean_query)

    # BM25 的 query tokens：terms + 把短语拆成 token（BM25 本质是 bag-of-words）
    query_tokens = []
    query_tokens.extend(terms)
    for ph in phrases:
        query_tokens.extend(tokenize_text(ph))

    # 构建语料：标题 token 复制多份实现“权重”
    corpus_tokens = []
    raw_texts = []  # 用于 phrase boost
    for p in papers:
        title = p.get("title", "") or ""
        abs_ = p.get("summary", "") or ""
        raw = f"{title}\n{abs_}"
        raw_texts.append(raw.lower())

        t_tokens = tokenize_text(title)
        a_tokens = tokenize_text(abs_)

        # 通过复制 token 模拟字段权重（简单有效）
        doc_tokens = []
        doc_tokens.extend(t_tokens * int(max(1, round(title_weight))))
        doc_tokens.extend(a_tokens * int(max(1, round(abstract_weight))))

        corpus_tokens.append(doc_tokens)

    if not corpus_tokens:
        return papers

    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(query_tokens)  # numpy array-like

    # 写回 rel_score，并做短语 boost（可选）
    for i, p in enumerate(papers):
        score = float(scores[i])
        if phrases:
            # 若短语原文直接出现（不做复杂匹配），给一点加成
            hit = 0
            text = raw_texts[i]
            for ph in phrases:
                if ph in text:
                    hit += 1
            score += phrase_boost * hit
        p["rel_score"] = score

    papers.sort(key=lambda x: x.get("rel_score", 0.0), reverse=True)
    return papers

# 示例用法
if __name__ == "__main__":
    data = PaperSearcher()._format_date("2023")
    print(data)