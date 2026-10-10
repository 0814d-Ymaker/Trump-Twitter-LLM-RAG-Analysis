import json
import math
import re
import html
import urllib.request
from pathlib import Path
import heapq


# =========================================================
# 基本设置
# =========================================================

BASE_DIR = Path(__file__).parent

INDEX_FILE = BASE_DIR / "Trump_rag_index_v1.jsonl"

OUTPUT_FILE = BASE_DIR / "Trump_rag_last_answer.txt"

EMBEDDING_API_URL = "http://127.0.0.1:1234/v1/embeddings"

CHAT_API_URL = "http://127.0.0.1:1234/api/v1/chat"

EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"

CHAT_MODEL = "qwen3.5-4b"

EXPECTED_DIMENSION = 768

TOP_K = 8


# =========================================================
# 原因类别
# =========================================================

REASON_OPTIONS = {

    "1": {
        "stance": "support",
        "code": "S1",
        "name": "个人领导力信任与情感认同"
    },

    "2": {
        "stance": "support",
        "code": "S2",
        "name": "反建制、司法质疑与政治对手否定"
    },

    "3": {
        "stance": "support",
        "code": "S4",
        "name": "经济、社会与国内政策表现"
    },

    "4": {
        "stance": "anti",
        "code": "A3",
        "name": "治理能力、行政表现与信息诚信"
    },

    "5": {
        "stance": "anti",
        "code": "A1",
        "name": "个人品行、道德与违法犯罪指控"
    },

    "6": {
        "stance": "anti",
        "code": "A2",
        "name": "法治、司法与民主程序争议"
    }
}


# =========================================================
# 查询文本转Embedding
# =========================================================

def embed_query(query):

    body = {
        "model": EMBEDDING_MODEL,
        "input": "search_query: " + query
    }

    request = urllib.request.Request(
        EMBEDDING_API_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    with urllib.request.urlopen(
        request,
        timeout=60
    ) as response:

        result = json.loads(
            response.read().decode("utf-8")
        )

    embedding = result["data"][0]["embedding"]

    if len(embedding) != EXPECTED_DIMENSION:

        raise ValueError(
            f"查询向量维度错误："
            f"预期 {EXPECTED_DIMENSION}，"
            f"实际 {len(embedding)}"
        )

    return embedding


# =========================================================
# 余弦相似度
# =========================================================

def cosine_similarity(a, b):

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for x, y in zip(a, b):

        dot += x * y

        norm_a += x * x

        norm_b += y * y

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return (
        dot
        /
        (
            math.sqrt(norm_a)
            *
            math.sqrt(norm_b)
        )
    )


# =========================================================
# Metadata Filter + Vector Retrieval
# =========================================================

def retrieve(
    query_embedding,
    target_stance,
    target_reason
):

    if not INDEX_FILE.exists():

        raise FileNotFoundError(
            f"找不到RAG索引文件：\n"
            f"{INDEX_FILE}"
        )

    best_results = []

    total_count = 0

    candidate_count = 0

    with INDEX_FILE.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)

            except json.JSONDecodeError:
                continue

            total_count += 1

            stance = str(
                item.get(
                    "stance",
                    ""
                )
            ).strip()

            reason = str(
                item.get(
                    "primary_reason_code",
                    ""
                )
            ).strip()

            # -------------------------------------------------
            # 先根据前面已经做好的分类结果进行过滤
            # -------------------------------------------------

            if stance != target_stance:
                continue

            if reason != target_reason:
                continue

            candidate_count += 1

            embedding = item.get(
                "embedding",
                []
            )

            if len(embedding) != EXPECTED_DIMENSION:
                continue

            similarity = cosine_similarity(
                query_embedding,
                embedding
            )

            result = {

                "similarity":
                    similarity,

                "row_id":
                    str(
                        item.get(
                            "row_id",
                            ""
                        )
                    ).strip(),

                "stance":
                    stance,

                "primary_reason_code":
                    reason,

                "tweet_content":
                    html.unescape(
                        str(
                            item.get(
                                "tweet_content",
                                ""
                            )
                        )
                    ),

                "reason_evidence":
                    html.unescape(
                        str(
                            item.get(
                                "reason_evidence",
                                ""
                            )
                        )
                    ),

                "tweet_url":
                    str(
                        item.get(
                            "tweet_url",
                            ""
                        )
                    ).strip()
            }

            heap_item = (
                similarity,
                candidate_count,
                result
            )

            if len(best_results) < TOP_K:

                heapq.heappush(
                    best_results,
                    heap_item
                )

            elif similarity > best_results[0][0]:

                heapq.heapreplace(
                    best_results,
                    heap_item
                )

    best_results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return (
        [
            x[2]
            for x in best_results
        ],
        total_count,
        candidate_count
    )


# =========================================================
# 构造给Qwen看的证据
#
# 关键修正：
# 不再使用 Evidence 1 / Evidence 2 让模型误认成Row ID。
#
# 每段直接以真实Row ID开头：
#
# [Row 4693]
# =========================================================

def build_evidence_text(results):

    blocks = []

    for item in results:

        row_id = item["row_id"]

        block = (
            f"[Row {row_id}]\n"
            f"Similarity: {item['similarity']:.4f}\n"
            f"Tweet: {item['tweet_content']}\n"
        )

        if item["reason_evidence"]:

            block += (
                "Previously extracted reason evidence: "
                f"{item['reason_evidence']}\n"
            )

        if item["tweet_url"]:

            block += (
                f"Source URL: "
                f"{item['tweet_url']}\n"
            )

        blocks.append(
            block
        )

    return "\n\n".join(
        blocks
    )


# =========================================================
# 调用LM Studio Qwen
# =========================================================

def call_qwen(
    system_prompt,
    user_prompt,
    max_output_tokens=1000
):

    body = {

        "model":
            CHAT_MODEL,

        "system_prompt":
            system_prompt,

        "input":
            user_prompt,

        "reasoning":
            "off",

        "temperature":
            0,

        "max_output_tokens":
            max_output_tokens,

        "store":
            False
    }

    request = urllib.request.Request(
        CHAT_API_URL,
        data=json.dumps(
            body,
            ensure_ascii=False
        ).encode("utf-8"),
        headers={
            "Content-Type":
                "application/json"
        },
        method="POST"
    )

    with urllib.request.urlopen(
        request,
        timeout=180
    ) as response:

        result = json.loads(
            response.read().decode(
                "utf-8"
            )
        )

    message_parts = []

    for item in result.get(
        "output",
        []
    ):

        if item.get(
            "type"
        ) == "message":

            content = item.get(
                "content",
                ""
            )

            if content:
                message_parts.append(
                    content
                )

    answer = "\n".join(
        message_parts
    ).strip()

    if not answer:

        raise ValueError(
            "Qwen没有返回正式回答。"
        )

    stats = result.get(
        "stats",
        {}
    )

    return (
        answer,
        stats
    )


# =========================================================
# 生成RAG回答
# =========================================================

def generate_answer(
    question,
    option,
    results
):

    evidence_text = build_evidence_text(
        results
    )

    allowed_row_ids = [
        item["row_id"]
        for item in results
    ]

    allowed_citations = ", ".join(
        f"[Row {row_id}]"
        for row_id in allowed_row_ids
    )

    system_prompt = """
You are an evidence-grounded social-media research assistant.

You are analyzing retrieved X/Twitter posts concerning Donald Trump.

You MUST answer using ONLY the retrieved tweet evidence supplied in the prompt.

Rules:

1. Do not use outside knowledge.
2. Do not invent facts, events, motivations, statistics, or interpretations not supported by the retrieved tweets.
3. Tweets may contain opinions, allegations, sarcasm, jokes, exaggeration, misinformation, or ambiguous language.
4. Never present accusations contained in tweets as established facts. Describe them as views, claims, or allegations expressed by users.
5. Every substantive claim should be connected to the supplied evidence.
6. Citation format MUST use the EXACT original Row ID shown in the evidence.
7. Example of a valid citation:
   [Row 4693]
8. Never use evidence order numbers such as [Row 1], [Row 2], etc. unless 1 or 2 is itself an actual supplied Row ID.
9. Do not invent a Row ID.
10. If an individual retrieved tweet appears sarcastic, ambiguous, contradictory, weak, or potentially misclassified, explicitly say so rather than treating it as strong evidence.
11. If the evidence is insufficient for a conclusion, explicitly say that.
12. Answer in Chinese.
"""

    user_prompt = f"""
研究问题：

{question}

已经选择的数据范围：

Stance: {option['stance']}
Reason Code: {option['code']}
Reason Category: {option['name']}

本次允许使用的真实引用编号只有：

{allowed_citations}

以下内容是从RAG向量索引中检索出的真实推文证据。

================ RETRIEVED EVIDENCE ================

{evidence_text}

================ END OF EVIDENCE ====================

请仅根据上述证据回答研究问题。

请按照以下结构回答：

1. 总体结论

概括这些检索结果反映出的主要观点。
不要把8条检索结果说成能够代表全部特朗普支持者或反对者。

2. 主要主题

归纳2至5个能够由这些推文直接支持的主题。

每个主题后面必须使用真实Row ID引用，例如：

[Row 4693]

3. 代表性证据

选择最有代表性的证据进行解释。

必须引用原始Row ID。

不要使用“第1条证据”“Row 1”这种编号替代真实Row ID。

4. 异常或弱证据

如果某些检索结果存在讽刺、含义模糊、与类别可能不一致、
信息不足等问题，请单独指出，并引用对应真实Row ID。

5. 局限性

说明：
- 本次回答只基于Top {TOP_K}条检索结果；
- 上游stance和reason分类由模型生成，因此可能存在分类误差；
- 推文属于用户表达，不能自动视为事实；
- 该结果不代表所有公众意见。
"""

    answer, stats = call_qwen(
        system_prompt,
        user_prompt,
        max_output_tokens=1600
    )

    return (
        answer,
        stats,
        allowed_row_ids,
        evidence_text
    )


# =========================================================
# 检查Qwen是否引用了错误Row ID
# =========================================================

def find_cited_row_ids(answer):

    matches = re.findall(
        r"\[Row\s+([0-9]+)\]",
        answer,
        flags=re.IGNORECASE
    )

    return matches


def validate_citations(
    answer,
    allowed_row_ids
):

    cited_ids = find_cited_row_ids(
        answer
    )

    allowed_set = set(
        str(x)
        for x in allowed_row_ids
    )

    invalid_ids = sorted(
        {
            row_id
            for row_id in cited_ids
            if row_id not in allowed_set
        },
        key=lambda x: int(x)
    )

    return (
        cited_ids,
        invalid_ids
    )


# =========================================================
# 如果模型还是引用错Row ID，则自动纠正一次
# =========================================================

def repair_citations(
    question,
    option,
    answer,
    results,
    invalid_ids
):

    evidence_text = build_evidence_text(
        results
    )

    allowed_row_ids = [
        item["row_id"]
        for item in results
    ]

    allowed_citations = ", ".join(
        f"[Row {row_id}]"
        for row_id in allowed_row_ids
    )

    system_prompt = """
You are correcting citation identifiers in an evidence-grounded research answer.

You must preserve the analytical meaning of the answer, but every citation must use an exact Row ID from the supplied evidence.

Do not invent Row IDs.

Answer in Chinese.
"""

    user_prompt = f"""
原始研究问题：

{question}

数据范围：

{option['stance']} / {option['code']} / {option['name']}

允许使用的Row ID只有：

{allowed_citations}

发现以下非法Row ID引用：

{invalid_ids}

原回答：

================ ORIGINAL ANSWER ================

{answer}

================ END ORIGINAL ANSWER ============

原始检索证据：

================ EVIDENCE =======================

{evidence_text}

================ END EVIDENCE ===================

请重新输出完整答案。

要求：

1. 保留原回答合理的分析内容。
2. 删除或修正所有非法引用。
3. 所有引用只能使用上述允许的真实Row ID。
4. 不得将Evidence顺序误写成Row ID。
5. 不得添加证据之外的新事实。
"""

    repaired_answer, stats = call_qwen(
        system_prompt,
        user_prompt,
        max_output_tokens=1100
    )

    return (
        repaired_answer,
        stats
    )


# =========================================================
# 打印检索结果
# =========================================================

def print_retrieved_results(
    results
):

    print()

    print("=" * 80)

    print(
        "Retrieved Evidence"
    )

    print("=" * 80)

    for rank, item in enumerate(
        results,
        start=1
    ):

        print()

        print(
            f"Evidence Rank {rank}"
        )

        print(
            f"真实Row ID："
            f"{item['row_id']}"
        )

        print(
            f"Similarity："
            f"{item['similarity']:.4f}"
        )

        print(
            f"Stance："
            f"{item['stance']}"
        )

        print(
            f"Primary Reason："
            f"{item['primary_reason_code']}"
        )

        print(
            "Tweet："
        )

        print(
            item["tweet_content"]
        )

        if item["reason_evidence"]:

            print(
                "Reason Evidence："
            )

            print(
                item["reason_evidence"]
            )

        if item["tweet_url"]:

            print(
                "URL："
            )

            print(
                item["tweet_url"]
            )


# =========================================================
# 保存最后一次RAG结果
# =========================================================

def save_result(
    question,
    option,
    results,
    answer,
    cited_ids,
    invalid_ids,
    stats
):

    lines = []

    lines.append(
        "=" * 80
    )

    lines.append(
        "Trump RAG Question Answering Result"
    )

    lines.append(
        "=" * 80
    )

    lines.append("")

    lines.append(
        f"Question: {question}"
    )

    lines.append(
        f"Stance: {option['stance']}"
    )

    lines.append(
        f"Reason Code: {option['code']}"
    )

    lines.append(
        f"Reason Category: {option['name']}"
    )

    lines.append("")

    lines.append(
        "=" * 80
    )

    lines.append(
        "Retrieved Evidence"
    )

    lines.append(
        "=" * 80
    )

    for rank, item in enumerate(
        results,
        start=1
    ):

        lines.append("")

        lines.append(
            f"Evidence Rank {rank}"
        )

        lines.append(
            f"Row ID: {item['row_id']}"
        )

        lines.append(
            f"Similarity: "
            f"{item['similarity']:.4f}"
        )

        lines.append(
            f"Tweet: "
            f"{item['tweet_content']}"
        )

        if item["tweet_url"]:

            lines.append(
                f"URL: "
                f"{item['tweet_url']}"
            )

    lines.append("")

    lines.append(
        "=" * 80
    )

    lines.append(
        "RAG Final Answer"
    )

    lines.append(
        "=" * 80
    )

    lines.append("")

    lines.append(
        answer
    )

    lines.append("")

    lines.append(
        "=" * 80
    )

    lines.append(
        "Citation Validation"
    )

    lines.append(
        "=" * 80
    )

    lines.append(
        f"Cited Row IDs: {cited_ids}"
    )

    lines.append(
        f"Invalid Row IDs: {invalid_ids}"
    )

    lines.append("")

    lines.append(
        "=" * 80
    )

    lines.append(
        "LM Studio Stats"
    )

    lines.append(
        "=" * 80
    )

    lines.append(
        "reasoning_output_tokens: "
        + str(
            stats.get(
                "reasoning_output_tokens",
                "N/A"
            )
        )
    )

    lines.append(
        "total_output_tokens: "
        + str(
            stats.get(
                "total_output_tokens",
                "N/A"
            )
        )
    )

    lines.append(
        "tokens_per_second: "
        + str(
            stats.get(
                "tokens_per_second",
                "N/A"
            )
        )
    )

    OUTPUT_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 80)

    print(
        "Trump RAG Question Answering System - Final"
    )

    print("=" * 80)

    print()

    print(
        "请选择要查询的原因范围："
    )

    print()

    for key, value in REASON_OPTIONS.items():

        print(
            f"{key}. "
            f"{value['stance']} / "
            f"{value['code']} / "
            f"{value['name']}"
        )

    print()

    choice = input(
        "请输入1-6："
    ).strip()

    if choice not in REASON_OPTIONS:

        print(
            "输入无效，程序结束。"
        )

        return

    option = REASON_OPTIONS[
        choice
    ]

    print()

    question = input(
        "请输入你的问题：\n"
    ).strip()

    if not question:

        print(
            "没有输入问题，程序结束。"
        )

        return

    print()

    print(
        "=" * 80
    )

    print(
        "第一步：生成查询向量"
    )

    print(
        "=" * 80
    )

    query_embedding = embed_query(
        question
    )

    print(
        "查询向量生成成功。"
    )

    print()

    print(
        "=" * 80
    )

    print(
        "第二步：Metadata Filter + Vector Retrieval"
    )

    print(
        "=" * 80
    )

    (
        results,
        total_count,
        candidate_count
    ) = retrieve(
        query_embedding,
        option["stance"],
        option["code"]
    )

    print(
        f"RAG总文档："
        f"{total_count}"
    )

    print(
        f"过滤范围："
        f"{option['stance']} / "
        f"{option['code']}"
    )

    print(
        f"过滤后候选文档："
        f"{candidate_count}"
    )

    print(
        f"最终检索证据："
        f"{len(results)}条"
    )

    if not results:

        print(
            "没有检索到有效证据，程序结束。"
        )

        return

    print_retrieved_results(
        results
    )

    print()

    print(
        "=" * 80
    )

    print(
        "第三步：Qwen Evidence-Grounded Generation"
    )

    print(
        "=" * 80
    )

    answer, stats, allowed_row_ids, _ = generate_answer(
        question,
        option,
        results
    )

    # -----------------------------------------------------
    # 自动检查引用
    # -----------------------------------------------------

    cited_ids, invalid_ids = validate_citations(
        answer,
        allowed_row_ids
    )

    if invalid_ids:

        print()

        print(
            "检测到错误Row ID引用：",
            invalid_ids
        )

        print(
            "正在自动要求Qwen修正引用..."
        )

        (
            answer,
            repair_stats
        ) = repair_citations(
            question,
            option,
            answer,
            results,
            invalid_ids
        )

        # 使用最后一次生成的stats
        stats = repair_stats

        cited_ids, invalid_ids = validate_citations(
            answer,
            allowed_row_ids
        )

    print()

    print("=" * 80)

    print(
        "RAG Final Answer"
    )

    print("=" * 80)

    print()

    print(
        answer
    )

    print()

    print("=" * 80)

    print(
        "Citation Validation"
    )

    print("=" * 80)

    print(
        "允许引用的Row ID："
    )

    print(
        allowed_row_ids
    )

    print()

    print(
        "回答中实际引用的Row ID："
    )

    print(
        cited_ids
    )

    print()

    if invalid_ids:

        print(
            "警告：仍然存在非法Row ID："
        )

        print(
            invalid_ids
        )

    else:

        print(
            "引用检查通过："
            "没有发现非法Row ID。"
        )

    print()

    print("=" * 80)

    print(
        "LM Studio Stats"
    )

    print("=" * 80)

    print(
        "reasoning_output_tokens：",
        stats.get(
            "reasoning_output_tokens",
            "N/A"
        )
    )

    print(
        "total_output_tokens：",
        stats.get(
            "total_output_tokens",
            "N/A"
        )
    )

    print(
        "tokens_per_second：",
        stats.get(
            "tokens_per_second",
            "N/A"
        )
    )

    # -----------------------------------------------------
    # 保存最后一次结果
    # -----------------------------------------------------

    save_result(
        question,
        option,
        results,
        answer,
        cited_ids,
        invalid_ids,
        stats
    )

    print()

    print("=" * 80)

    print(
        "结果已保存"
    )

    print("=" * 80)

    print(
        OUTPUT_FILE
    )

    print()

    if (
        not invalid_ids
        and stats.get(
            "reasoning_output_tokens",
            0
        ) == 0
    ):

        print(
            "RAG主链路运行完成，"
            "Row ID引用检查通过，"
            "reasoning已关闭。"
        )

    elif not invalid_ids:

        print(
            "RAG主链路运行完成，"
            "Row ID引用检查通过。"
        )


if __name__ == "__main__":
    main()