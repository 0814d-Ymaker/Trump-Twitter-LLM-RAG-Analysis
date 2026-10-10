import csv
import json
import time
import urllib.request
from pathlib import Path


# =========================================================
# 基本设置
# =========================================================

BASE_DIR = Path(__file__).parent

SUPPORT_SAMPLE_FILE = (
    BASE_DIR / "reason_seed_support_200.csv"
)

ANTI_SAMPLE_FILE = (
    BASE_DIR / "reason_seed_anti_200.csv"
)


# V2 checkpoint：
# 第一次运行V3时可以继承V2已经成功的8批结果
V2_CHECKPOINT_FILE = (
    BASE_DIR / "reason_taxonomy_discovery_checkpoint_v2.json"
)


# V3自己的断点
V3_CHECKPOINT_FILE = (
    BASE_DIR / "reason_taxonomy_discovery_checkpoint_v3.json"
)


V3_LOG_FILE = (
    BASE_DIR / "reason_taxonomy_discovery_log_v3.txt"
)


SUPPORT_OUTPUT_FILE = (
    BASE_DIR / "reason_taxonomy_support_candidate_v3.txt"
)

ANTI_OUTPUT_FILE = (
    BASE_DIR / "reason_taxonomy_anti_candidate_v3.txt"
)


API_URL = (
    "http://127.0.0.1:1234/api/v1/chat"
)

MODEL_NAME = "qwen3.5-4b"

BATCH_SIZE = 25

MAX_RETRIES = 3


# =========================================================
# 调用 LM Studio 原生 API
# =========================================================

def call_qwen(
    prompt,
    max_output_tokens
):

    request_body = {

        "model": MODEL_NAME,

        "input": prompt,

        "system_prompt":
            "You are a careful social media research assistant. "
            "Base all conclusions only on the supplied tweets "
            "or supplied summaries. "
            "Do not invent political explanations that are not "
            "supported by the supplied material.",

        # 关键：
        # 从API层面真正关闭推理
        "reasoning": "off",

        "temperature": 0,

        "max_output_tokens": max_output_tokens,

        "store": False
    }


    data = json.dumps(
        request_body,
        ensure_ascii=False
    ).encode("utf-8")


    request = urllib.request.Request(

        API_URL,

        data=data,

        headers={
            "Content-Type": "application/json"
        },

        method="POST"
    )


    with urllib.request.urlopen(
        request,
        timeout=600
    ) as response:

        response_text = (
            response
            .read()
            .decode("utf-8")
        )


    result = json.loads(
        response_text
    )


    # =====================================================
    # 提取正式回答
    # =====================================================

    message_parts = []


    for item in result.get(
        "output",
        []
    ):

        if item.get(
            "type"
        ) == "message":

            content = (
                item.get(
                    "content",
                    ""
                )
                or ""
            ).strip()


            if content:

                message_parts.append(
                    content
                )


    content = "\n\n".join(
        message_parts
    ).strip()


    # =====================================================
    # Token统计
    # =====================================================

    stats = result.get(
        "stats",
        {}
    )


    reasoning_tokens = stats.get(
        "reasoning_output_tokens",
        "未知"
    )


    total_output_tokens = stats.get(
        "total_output_tokens",
        "未知"
    )


    return (
        content,
        reasoning_tokens,
        total_output_tokens,
        result
    )


# =========================================================
# 原始25条批次有效性检查
# =========================================================

def validate_original_batch(
    content,
    reasoning_tokens
):

    problems = []


    if len(content) < 300:

        problems.append(
            "正文少于300字符"
        )


    if "候选原因" not in content:

        problems.append(
            "缺少“候选原因”"
        )


    if "原因名称" not in content:

        problems.append(
            "缺少“原因名称”"
        )


    if (
        isinstance(
            reasoning_tokens,
            int
        )
        and
        reasoning_tokens != 0
    ):

        problems.append(
            f"reasoning_output_tokens="
            f"{reasoning_tokens}"
        )


    return problems


# =========================================================
# 中间压缩结果有效性检查
# =========================================================

def validate_compression(
    content,
    reasoning_tokens
):

    problems = []


    if len(content) < 400:

        problems.append(
            "压缩结果少于400字符"
        )


    if "候选原因" not in content:

        problems.append(
            "压缩结果缺少“候选原因”"
        )


    if "原因名称" not in content:

        problems.append(
            "压缩结果缺少“原因名称”"
        )


    if (
        isinstance(
            reasoning_tokens,
            int
        )
        and
        reasoning_tokens != 0
    ):

        problems.append(
            f"reasoning_output_tokens="
            f"{reasoning_tokens}"
        )


    return problems


# =========================================================
# 最终taxonomy有效性检查
# =========================================================

def validate_final(
    content,
    reasoning_tokens
):

    problems = []


    if len(content) < 800:

        problems.append(
            "最终体系少于800字符"
        )


    required_terms = [
        "一级原因",
        "二级原因",
        "定义",
        "编码原则",
        "人工复核"
    ]


    for term in required_terms:

        if term not in content:

            problems.append(
                f"缺少“{term}”"
            )


    if (
        isinstance(
            reasoning_tokens,
            int
        )
        and
        reasoning_tokens != 0
    ):

        problems.append(
            f"reasoning_output_tokens="
            f"{reasoning_tokens}"
        )


    return problems


# =========================================================
# 带有效性检查和重试的调用
# =========================================================

def call_with_retry(
    prompt,
    max_output_tokens,
    validation_type
):

    last_error = None


    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            (
                content,
                reasoning_tokens,
                total_output_tokens,
                raw_result
            ) = call_qwen(
                prompt,
                max_output_tokens
            )


            print(
                f"正式回答长度："
                f"{len(content)} 字符"
            )


            print(
                f"reasoning_output_tokens："
                f"{reasoning_tokens}"
            )


            print(
                f"total_output_tokens："
                f"{total_output_tokens}"
            )


            # =============================================
            # 根据阶段采用不同验证规则
            # =============================================

            if validation_type == "original":

                problems = (
                    validate_original_batch(
                        content,
                        reasoning_tokens
                    )
                )


            elif validation_type == "compression":

                problems = (
                    validate_compression(
                        content,
                        reasoning_tokens
                    )
                )


            elif validation_type == "final":

                problems = (
                    validate_final(
                        content,
                        reasoning_tokens
                    )
                )


            else:

                raise ValueError(
                    "未知 validation_type："
                    f"{validation_type}"
                )


            if not problems:

                return (
                    content,
                    raw_result
                )


            print(
                "输出有效性检查失败："
            )


            for problem in problems:

                print(
                    f"- {problem}"
                )


            last_error = "；".join(
                problems
            )


        except Exception as e:

            last_error = str(e)


            print(
                f"第 {attempt} 次请求异常："
                f"{e}"
            )


        if attempt < MAX_RETRIES:

            print(
                "3秒后重新尝试..."
            )

            time.sleep(3)


    raise RuntimeError(
        "连续3次均未获得有效输出。\n"
        f"最后问题：{last_error}"
    )


# =========================================================
# 读取CSV
# =========================================================

def load_sample(
    file_path
):

    if not file_path.exists():

        raise FileNotFoundError(
            f"找不到文件：\n"
            f"{file_path}"
        )


    with file_path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        rows = list(reader)


    if not rows:

        raise ValueError(
            f"文件为空："
            f"{file_path}"
        )


    return rows


# =========================================================
# 切成25条一批
# =========================================================

def make_batches(
    rows
):

    return [

        rows[
            i:i + BATCH_SIZE
        ]

        for i in range(
            0,
            len(rows),
            BATCH_SIZE
        )
    ]


# =========================================================
# checkpoint
# =========================================================

def create_empty_checkpoint():

    return {

        "support_batches": {},
        "anti_batches": {},

        "support_pairs": {},
        "anti_pairs": {},

        "support_halves": {},
        "anti_halves": {},

        "support_final": None,
        "anti_final": None
    }


def load_checkpoint():

    # =====================================================
    # 优先读取V3
    # =====================================================

    if V3_CHECKPOINT_FILE.exists():

        print(
            "检测到 V3 checkpoint，"
            "继续已有进度..."
        )


        with V3_CHECKPOINT_FILE.open(
            "r",
            encoding="utf-8"
        ) as f:

            checkpoint = json.load(f)


    # =====================================================
    # 没有V3时继承V2
    # =====================================================

    elif V2_CHECKPOINT_FILE.exists():

        print(
            "检测到 V2 checkpoint，"
            "正在继承已经成功的批次结果..."
        )


        with V2_CHECKPOINT_FILE.open(
            "r",
            encoding="utf-8"
        ) as f:

            old = json.load(f)


        checkpoint = (
            create_empty_checkpoint()
        )


        checkpoint[
            "support_batches"
        ] = old.get(
            "support_batches",
            {}
        )


        checkpoint[
            "anti_batches"
        ] = old.get(
            "anti_batches",
            {}
        )


    else:

        checkpoint = (
            create_empty_checkpoint()
        )


    # =====================================================
    # 补齐字段
    # =====================================================

    defaults = (
        create_empty_checkpoint()
    )


    for key, value in defaults.items():

        checkpoint.setdefault(
            key,
            value
        )


    # =====================================================
    # 清除无效final
    # =====================================================

    for key in [
        "support_final",
        "anti_final"
    ]:

        value = checkpoint.get(
            key
        )


        if (
            value
            and
            len(
                str(value).strip()
            ) < 800
        ):

            checkpoint[
                key
            ] = None


    return checkpoint


def save_checkpoint(
    checkpoint
):

    with V3_CHECKPOINT_FILE.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            checkpoint,
            f,
            ensure_ascii=False,
            indent=2
        )


# =========================================================
# 日志
# =========================================================

def append_log(
    title,
    content
):

    with V3_LOG_FILE.open(
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n"
            + "=" * 80
            + "\n"
        )


        f.write(
            title
            + "\n"
        )


        f.write(
            "=" * 80
            + "\n\n"
        )


        f.write(
            content
        )


        f.write(
            "\n"
        )


# =========================================================
# 格式化原始推文
# =========================================================

def format_batch(
    batch
):

    parts = []


    for row in batch:

        sample_id = (
            row.get(
                "Sample ID",
                ""
            )
            or ""
        )


        content = (
            row.get(
                "Tweet Content",
                ""
            )
            or ""
        )


        parts.append(
            f"[{sample_id}]\n"
            f"{content}"
        )


    return "\n\n".join(
        parts
    )


# =========================================================
# 原始25条提示词
# =========================================================

def build_original_prompt(
    batch,
    stance_name
):

    tweets_text = format_batch(
        batch
    )


    if stance_name == "支持":

        task = """
这些推文已经被前一阶段分类为
“支持 Donald Trump”。

不要重新判断立场。

研究问题：

这些用户为什么支持 Trump？
"""

    else:

        task = """
这些推文已经被前一阶段分类为
“反对 Donald Trump”。

不要重新判断立场。

研究问题：

这些用户为什么反对 Trump？
"""


    return f"""
你正在进行 Twitter/X 社交媒体政治立场原因分析。

{task}

规则：

1. 只能依据本批推文本身。
2. 不使用外部政治知识补充作者没有表达的原因。
3. 没有明确原因的推文不要强行归因。
4. 相似原因应合并。
5. 不要把“支持Trump”或“反对Trump”本身当原因。
6. 原因必须是以后可以重复使用的稳定类别。
7. 不要逐条分析全部推文。

请归纳大约4～8个候选原因。

使用下面格式：

【候选原因1】

原因名称：
原因定义：
典型表现：
代表样本ID：

【候选原因2】

原因名称：
原因定义：
典型表现：
代表样本ID：

……

【本批观察】

最常见原因：
证据较弱的原因：
无法明确归因的情况：

直接开始分析，不要打招呼。

==================================================
推文
==================================================

{tweets_text}
""".strip()


# =========================================================
# 原始8批
# =========================================================

def run_original_batches(
    stance_name,
    rows,
    checkpoint,
    batch_key
):

    batches = make_batches(
        rows
    )


    completed = checkpoint[
        batch_key
    ]


    print()
    print("-" * 72)

    print(
        f"{stance_name}：检查原始8批结果"
    )

    print("-" * 72)


    for index, batch in enumerate(
        batches,
        start=1
    ):

        key = str(
            index
        )


        existing = completed.get(
            key,
            ""
        )


        # 有效断点直接跳过
        if len(
            existing.strip()
        ) >= 300:

            print(
                f"{stance_name} 第 "
                f"{index}/8 批："
                "已有有效断点，跳过。"
            )

            continue


        print()
        print(
            f"{stance_name} 第 "
            f"{index}/8 批：开始运行"
        )


        prompt = (
            build_original_prompt(
                batch,
                stance_name
            )
        )


        start = time.time()


        content, _ = (
            call_with_retry(

                prompt,

                max_output_tokens=1800,

                validation_type="original"
            )
        )


        elapsed = (
            time.time()
            - start
        )


        completed[
            key
        ] = content


        checkpoint[
            batch_key
        ] = completed


        save_checkpoint(
            checkpoint
        )


        append_log(
            (
                f"{stance_name} "
                f"Original Batch {index}"
            ),
            content
        )


        print(
            f"完成，耗时："
            f"{elapsed:.2f} 秒"
        )


    return completed


# =========================================================
# 中间压缩提示词
# =========================================================

def build_compression_prompt(
    stance_name,
    summaries,
    stage_name
):

    pieces = []


    for index, summary in enumerate(
        summaries,
        start=1
    ):

        pieces.append(
            f"========== 输入总结 {index} ==========\n"
            f"{summary}"
        )


    combined = "\n\n".join(
        pieces
    )


    if stance_name == "支持":

        question = (
            "用户为什么支持 Donald Trump？"
        )

    else:

        question = (
            "用户为什么反对 Donald Trump？"
        )


    return f"""
你正在进行社交媒体原因分类体系的中间合并。

研究问题：

{question}

当前阶段：

{stage_name}

输入内容已经是上一阶段产生的原因总结，
不是新的原始推文。

你的任务是：

把下面多份总结压缩、去重、合并成一份
更加稳定的候选原因总结。

规则：

1. 不创造输入中没有出现的新原因。
2. 合并相同或高度相似的原因。
3. 保留真正不同的重要原因。
4. 删除重复描述。
5. 不要过度细分。
6. 没有明确依据的原因不要保留。
7. 这还不是最终分类体系，
   重点是压缩和去重。

输出格式：

【候选原因1】

原因名称：
原因定义：
典型表现：
代表样本ID：

【候选原因2】

原因名称：
原因定义：
典型表现：
代表样本ID：

……

最后如篇幅允许，可以补充：

【本批观察】

最常见原因：
可能重叠的原因：
证据较弱的原因：

直接输出结果，不要打招呼。

==================================================
需要合并的总结
==================================================

{combined}
""".strip()


# =========================================================
# 第一层：
# 8 → 4
# =========================================================

def merge_to_four(
    stance_name,
    batches,
    checkpoint,
    pair_key
):

    pair_results = checkpoint[
        pair_key
    ]


    pairs = [
        ("1", "2"),
        ("3", "4"),
        ("5", "6"),
        ("7", "8")
    ]


    print()
    print("-" * 72)

    print(
        f"{stance_name}：第一层合并 8 → 4"
    )

    print("-" * 72)


    for pair_index, (
        first,
        second
    ) in enumerate(
        pairs,
        start=1
    ):

        key = str(
            pair_index
        )


        existing = pair_results.get(
            key,
            ""
        )


        if len(
            existing.strip()
        ) >= 400:

            print(
                f"{stance_name} Pair "
                f"{pair_index}/4："
                "已有有效断点，跳过。"
            )

            continue


        print()
        print(
            f"{stance_name} Pair "
            f"{pair_index}/4：开始"
        )


        prompt = (
            build_compression_prompt(

                stance_name,

                [
                    batches[first],
                    batches[second]
                ],

                (
                    "第一层压缩："
                    f"原始批次 {first} + {second}"
                )
            )
        )


        start = time.time()


        content, _ = (
            call_with_retry(

                prompt,

                # 修复：
                # 原来1100太小
                max_output_tokens=1800,

                validation_type="compression"
            )
        )


        elapsed = (
            time.time()
            - start
        )


        pair_results[
            key
        ] = content


        checkpoint[
            pair_key
        ] = pair_results


        save_checkpoint(
            checkpoint
        )


        append_log(
            (
                f"{stance_name} "
                f"PAIR {pair_index}"
            ),
            content
        )


        print(
            f"完成，耗时："
            f"{elapsed:.2f} 秒"
        )


    return pair_results


# =========================================================
# 第二层：
# 4 → 2
# =========================================================

def merge_to_two(
    stance_name,
    pair_results,
    checkpoint,
    half_key
):

    half_results = checkpoint[
        half_key
    ]


    groups = [
        ("1", "2"),
        ("3", "4")
    ]


    print()
    print("-" * 72)

    print(
        f"{stance_name}：第二层合并 4 → 2"
    )

    print("-" * 72)


    for group_index, (
        first,
        second
    ) in enumerate(
        groups,
        start=1
    ):

        key = str(
            group_index
        )


        existing = half_results.get(
            key,
            ""
        )


        if len(
            existing.strip()
        ) >= 400:

            print(
                f"{stance_name} Half "
                f"{group_index}/2："
                "已有有效断点，跳过。"
            )

            continue


        print()
        print(
            f"{stance_name} Half "
            f"{group_index}/2：开始"
        )


        prompt = (
            build_compression_prompt(

                stance_name,

                [
                    pair_results[first],
                    pair_results[second]
                ],

                (
                    "第二层压缩："
                    f"Pair {first} + Pair {second}"
                )
            )
        )


        start = time.time()


        content, _ = (
            call_with_retry(

                prompt,

                # 第二层允许更多输出
                max_output_tokens=2000,

                validation_type="compression"
            )
        )


        elapsed = (
            time.time()
            - start
        )


        half_results[
            key
        ] = content


        checkpoint[
            half_key
        ] = half_results


        save_checkpoint(
            checkpoint
        )


        append_log(
            (
                f"{stance_name} "
                f"HALF {group_index}"
            ),
            content
        )


        print(
            f"完成，耗时："
            f"{elapsed:.2f} 秒"
        )


    return half_results


# =========================================================
# 最终taxonomy提示词
# =========================================================

def build_final_prompt(
    stance_name,
    half_results
):

    if stance_name == "支持":

        question = (
            "为什么这些用户支持 Donald Trump？"
        )

    else:

        question = (
            "为什么这些用户反对 Donald Trump？"
        )


    combined = (
        "========== 总结 A ==========\n"
        + half_results["1"]
        + "\n\n"
        + "========== 总结 B ==========\n"
        + half_results["2"]
    )


    return f"""
现在请根据下面两个经过分层压缩的原因总结，
建立最终候选原因分类体系。

这些总结最终来自200条随机抽样推文。

研究问题：

{question}

==================================================
分类原则
==================================================

1. 只能依据下面总结中真正存在的原因。
2. 合并近义、重复类别。
3. 不为了凑数量创造新原因。
4. 不要过度细分。
5. 一条推文以后可以同时包含多个原因。
6. 没有明确原因时允许使用“原因不明确”。
7. 分类名称必须清晰、稳定、可复用。

建立三级结构：

一级原因
→ 二级原因
→ 数据确实支持时才设置三级原因

一级原因建议约5～8类，
但不要强行凑数量。

==================================================
必须使用以下格式
==================================================

【一级原因 1：名称】

定义：

包括：

不包括：

二级原因：

1.1 二级原因名称

定义：

三级原因（如有）：
- 1.1.1 ...
- 1.1.2 ...


【一级原因 2：名称】

定义：

包括：

不包括：

二级原因：

……


最后必须包含：

【建议的编码原则】

说明：

1. 一条推文是否允许多个原因；
2. 没有明确原因如何处理；
3. 如何处理讽刺；
4. 如何处理仅攻击政党或政治对手；
5. 如何处理新闻转述；
6. 如何处理模糊指代。


【可能需要人工复核的边界】

列出最容易混淆的类别及区分方法。

==================================================
分层压缩后的两个总结
==================================================

{combined}
""".strip()


# =========================================================
# 最终：
# 2 → 1
# =========================================================

def build_final_taxonomy(
    stance_name,
    half_results,
    checkpoint,
    final_key,
    output_file
):

    existing = checkpoint.get(
        final_key
    )


    if (
        existing
        and
        len(
            existing.strip()
        ) >= 800
    ):

        print(
            f"{stance_name}最终体系："
            "已有有效断点，跳过。"
        )

        final_result = existing


    else:

        print()
        print("-" * 72)

        print(
            f"{stance_name}：最终合并 2 → 1"
        )

        print("-" * 72)


        prompt = build_final_prompt(
            stance_name,
            half_results
        )


        start = time.time()


        final_result, _ = (
            call_with_retry(

                prompt,

                max_output_tokens=2500,

                validation_type="final"
            )
        )


        elapsed = (
            time.time()
            - start
        )


        checkpoint[
            final_key
        ] = final_result


        save_checkpoint(
            checkpoint
        )


        append_log(
            (
                f"{stance_name} "
                "FINAL TAXONOMY"
            ),
            final_result
        )


        print(
            f"{stance_name}最终候选体系完成。"
        )


        print(
            f"耗时："
            f"{elapsed:.2f} 秒"
        )


    with output_file.open(
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            final_result
        )


    print(
        f"结果文件：\n"
        f"{output_file}"
    )


# =========================================================
# 一个立场的完整流程
# =========================================================

def process_stance(
    stance_name,
    rows,
    checkpoint,
    batch_key,
    pair_key,
    half_key,
    final_key,
    output_file
):

    print()
    print("=" * 72)

    print(
        f"开始处理：{stance_name}"
    )

    print("=" * 72)


    original_batches = (
        run_original_batches(

            stance_name,

            rows,

            checkpoint,

            batch_key
        )
    )


    pair_results = (
        merge_to_four(

            stance_name,

            original_batches,

            checkpoint,

            pair_key
        )
    )


    half_results = (
        merge_to_two(

            stance_name,

            pair_results,

            checkpoint,

            half_key
        )
    )


    build_final_taxonomy(

        stance_name,

        half_results,

        checkpoint,

        final_key,

        output_file
    )


# =========================================================
# 主程序
# =========================================================

def main():

    print("=" * 72)

    print(
        "Trump Reason Taxonomy Discovery V3 FINAL"
    )

    print(
        "Native API + Reasoning OFF + Hierarchical Merge"
    )

    print("=" * 72)


    support_rows = load_sample(
        SUPPORT_SAMPLE_FILE
    )


    anti_rows = load_sample(
        ANTI_SAMPLE_FILE
    )


    print()

    print(
        f"支持样本："
        f"{len(support_rows)} 条"
    )


    print(
        f"反对样本："
        f"{len(anti_rows)} 条"
    )


    checkpoint = (
        load_checkpoint()
    )


    save_checkpoint(
        checkpoint
    )


    # =====================================================
    # 支持
    # =====================================================

    process_stance(

        "支持",

        support_rows,

        checkpoint,

        "support_batches",

        "support_pairs",

        "support_halves",

        "support_final",

        SUPPORT_OUTPUT_FILE
    )


    # =====================================================
    # 反对
    # =====================================================

    process_stance(

        "反对",

        anti_rows,

        checkpoint,

        "anti_batches",

        "anti_pairs",

        "anti_halves",

        "anti_final",

        ANTI_OUTPUT_FILE
    )


    print()
    print("=" * 72)

    print(
        "原因体系发现 V3 完成"
    )

    print("=" * 72)

    print()


    print(
        f"支持最终候选体系：\n"
        f"{SUPPORT_OUTPUT_FILE}"
    )

    print()


    print(
        f"反对最终候选体系：\n"
        f"{ANTI_OUTPUT_FILE}"
    )

    print()


    print(
        f"V3 checkpoint：\n"
        f"{V3_CHECKPOINT_FILE}"
    )

    print()


    print(
        f"V3 日志：\n"
        f"{V3_LOG_FILE}"
    )

    print()


    print(
        "下一步："
        "人工审查支持和反对两个最终候选原因体系。"
    )


if __name__ == "__main__":

    main()