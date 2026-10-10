import time
from pathlib import Path

import discover_reason_taxonomy_v3 as base


# =========================================================
# V4 输出文件
# =========================================================

BASE_DIR = Path(__file__).parent

SUPPORT_OUTPUT_FILE = (
    BASE_DIR / "reason_taxonomy_support_candidate_v4.txt"
)

ANTI_OUTPUT_FILE = (
    BASE_DIR / "reason_taxonomy_anti_candidate_v4.txt"
)

V4_LOG_FILE = (
    BASE_DIR / "reason_taxonomy_discovery_log_v4.txt"
)


MAX_RETRIES = 3


# =========================================================
# 日志
# =========================================================

def append_log(title, content):

    with V4_LOG_FILE.open(
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

        f.write(content)

        f.write("\n")


# =========================================================
# 通用模型调用 + 验证
# =========================================================

def call_validated(
    prompt,
    max_output_tokens,
    stage
):

    last_problem = None


    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            (
                content,
                reasoning_tokens,
                total_output_tokens,
                _
            ) = base.call_qwen(
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


            problems = []


            # =============================================
            # reasoning 必须真正关闭
            # =============================================

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


            # =============================================
            # 原因体系主体
            # =============================================

            if stage == "core":

                if len(content) < 800:

                    problems.append(
                        "原因体系主体少于800字符"
                    )


                required = [
                    "一级原因",
                    "二级原因",
                    "定义"
                ]


                for term in required:

                    if term not in content:

                        problems.append(
                            f"缺少“{term}”"
                        )


            # =============================================
            # 编码原则 + 人工复核
            # =============================================

            elif stage == "rules":

                if len(content) < 300:

                    problems.append(
                        "编码规则正文少于300字符"
                    )


                if "编码原则" not in content:

                    problems.append(
                        "缺少“编码原则”"
                    )


                if "人工复核" not in content:

                    problems.append(
                        "缺少“人工复核”"
                    )


            else:

                raise ValueError(
                    f"未知阶段：{stage}"
                )


            # =============================================
            # 是否通过
            # =============================================

            if not problems:

                return content


            print(
                "输出有效性检查失败："
            )


            for problem in problems:

                print(
                    f"- {problem}"
                )


            last_problem = (
                "；".join(problems)
            )


        except Exception as e:

            last_problem = str(e)

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
        "连续3次未得到有效结果。\n"
        f"最后问题：{last_problem}"
    )


# =========================================================
# 构建原因体系主体提示词
# =========================================================

def build_core_prompt(
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
你正在构建 Twitter/X 社交媒体政治立场的
最终原因分类体系。

研究问题：

{question}

下面两个总结已经经过：

200条随机样本
→ 8个原始批次
→ 4份合并总结
→ 2份压缩总结

现在只完成一件事情：

建立最终的“原因分类体系主体”。

==================================================
原则
==================================================

1. 只能使用下面总结真正支持的原因。
2. 合并重复、近义原因。
3. 不创造数据中没有出现的新原因。
4. 不要过度细分。
5. 原因类别必须可以用于后续大规模编码。
6. 一级原因建议5～8类，但不要强行凑数量。
7. 一级原因下建立二级原因。
8. 只有数据明显支持时才建立三级原因。
9. 输出必须简洁，不要写长篇解释。
10. 本次不要写编码原则，也不要写人工复核部分。

==================================================
输出格式
==================================================

【一级原因 1：名称】

定义：
用1～2句话说明。

包括：
- 简洁说明

不包括：
- 简洁说明

二级原因：

1.1 二级原因名称
定义：
简洁说明。

三级原因（如有）：
- 1.1.1 ...
- 1.1.2 ...


1.2 二级原因名称
定义：
简洁说明。


【一级原因 2：名称】

……

请尽量控制篇幅。

每个一级原因最多设置约2～4个主要二级原因。

不要为了完整而制造没有数据依据的类别。

直接输出原因体系，不要打招呼。

==================================================
两个压缩总结
==================================================

{combined}
""".strip()


# =========================================================
# 构建编码原则提示词
# =========================================================

def build_rules_prompt(
    stance_name,
    taxonomy_core
):

    if stance_name == "支持":

        direction = "支持 Trump"

    else:

        direction = "反对 Trump"


    return f"""
下面是一套已经生成的 Twitter/X
“{direction}”原因分类体系。

现在不要修改原因体系本身。

你的任务只完成两个附加部分：

1. 建议的编码原则
2. 可能需要人工复核的边界

==================================================
已经确定的原因体系
==================================================

{taxonomy_core}

==================================================
输出要求
==================================================

必须使用下面格式：

【建议的编码原则】

1. 多原因：
说明一条推文包含多个原因时如何编码。

2. 原因不明确：
说明只有立场、没有明确理由时如何处理。

3. 讽刺：
说明如何根据文本证据处理讽刺。

4. 政党或政治对手：
说明只攻击其他政党/候选人时，
什么时候可以视为与Trump相关原因，
什么时候不能。

5. 新闻转述：
说明转发或描述新闻本身时如何处理。

6. 模糊指代：
说明文本证据不足时如何处理。


【可能需要人工复核的边界】

列出3～6个最容易混淆的原因类别，
并简要说明区分标准。

要求：

- 不增加新的原因类别。
- 不使用外部知识。
- 内容简洁。
- 不重复整套原因体系。
- 直接输出，不要打招呼。
""".strip()


# =========================================================
# 两阶段最终合并
# =========================================================

def build_final_two_stage(
    stance_name,
    half_results,
    checkpoint,
    core_key,
    rules_key,
    final_key,
    output_file
):

    print()
    print("-" * 72)

    print(
        f"{stance_name}："
        "最终阶段 A —— 原因体系主体"
    )

    print("-" * 72)


    # =====================================================
    # Stage A
    # =====================================================

    existing_core = checkpoint.get(
        core_key
    )


    if (
        existing_core
        and
        len(
            existing_core.strip()
        ) >= 800
    ):

        print(
            "原因体系主体已有有效断点，跳过。"
        )

        core = existing_core


    else:

        prompt = build_core_prompt(
            stance_name,
            half_results
        )


        start = time.time()


        core = call_validated(
            prompt,
            max_output_tokens=2200,
            stage="core"
        )


        elapsed = (
            time.time()
            - start
        )


        checkpoint[
            core_key
        ] = core


        base.save_checkpoint(
            checkpoint
        )


        append_log(
            (
                f"{stance_name} "
                "FINAL CORE"
            ),
            core
        )


        print(
            f"原因体系主体完成，耗时："
            f"{elapsed:.2f} 秒"
        )


    # =====================================================
    # Stage B
    # =====================================================

    print()
    print("-" * 72)

    print(
        f"{stance_name}："
        "最终阶段 B —— 编码原则与人工复核"
    )

    print("-" * 72)


    existing_rules = checkpoint.get(
        rules_key
    )


    if (
        existing_rules
        and
        len(
            existing_rules.strip()
        ) >= 300
        and
        "编码原则" in existing_rules
        and
        "人工复核" in existing_rules
    ):

        print(
            "编码原则已有有效断点，跳过。"
        )

        rules = existing_rules


    else:

        prompt = build_rules_prompt(
            stance_name,
            core
        )


        start = time.time()


        rules = call_validated(
            prompt,
            max_output_tokens=1000,
            stage="rules"
        )


        elapsed = (
            time.time()
            - start
        )


        checkpoint[
            rules_key
        ] = rules


        base.save_checkpoint(
            checkpoint
        )


        append_log(
            (
                f"{stance_name} "
                "CODING RULES"
            ),
            rules
        )


        print(
            f"编码原则完成，耗时："
            f"{elapsed:.2f} 秒"
        )


    # =====================================================
    # 合并最终结果
    # =====================================================

    final_result = (
        core.strip()
        + "\n\n"
        + rules.strip()
        + "\n"
    )


    checkpoint[
        final_key
    ] = final_result


    base.save_checkpoint(
        checkpoint
    )


    with output_file.open(
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            final_result
        )


    print()
    print(
        f"{stance_name}最终候选体系生成成功。"
    )

    print(
        f"最终正文长度："
        f"{len(final_result)} 字符"
    )

    print(
        f"结果文件：\n"
        f"{output_file}"
    )


# =========================================================
# 一个立场完整流程
# =========================================================

def process_stance(
    stance_name,
    rows,
    checkpoint,
    batch_key,
    pair_key,
    half_key,
    core_key,
    rules_key,
    final_key,
    output_file
):

    print()
    print("=" * 72)

    print(
        f"开始处理：{stance_name}"
    )

    print("=" * 72)


    # =====================================================
    # 原始8批
    # 已完成的自动跳过
    # =====================================================

    original_batches = (
        base.run_original_batches(

            stance_name,

            rows,

            checkpoint,

            batch_key
        )
    )


    # =====================================================
    # 8 → 4
    # 已完成的自动跳过
    # =====================================================

    pair_results = (
        base.merge_to_four(

            stance_name,

            original_batches,

            checkpoint,

            pair_key
        )
    )


    # =====================================================
    # 4 → 2
    # 已完成的自动跳过
    # =====================================================

    half_results = (
        base.merge_to_two(

            stance_name,

            pair_results,

            checkpoint,

            half_key
        )
    )


    # =====================================================
    # V4 最终阶段
    # =====================================================

    build_final_two_stage(

        stance_name,

        half_results,

        checkpoint,

        core_key,

        rules_key,

        final_key,

        output_file
    )


# =========================================================
# 主程序
# =========================================================

def main():

    print("=" * 72)

    print(
        "Trump Reason Taxonomy Discovery V4"
    )

    print(
        "Two-Stage Final Merge + Reasoning OFF"
    )

    print("=" * 72)


    # =====================================================
    # 读取400条样本
    # =====================================================

    support_rows = base.load_sample(
        base.SUPPORT_SAMPLE_FILE
    )


    anti_rows = base.load_sample(
        base.ANTI_SAMPLE_FILE
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


    # =====================================================
    # 直接读取V3已有checkpoint
    # =====================================================

    checkpoint = (
        base.load_checkpoint()
    )


    # 新增V4字段
    checkpoint.setdefault(
        "support_final_core_v4",
        None
    )

    checkpoint.setdefault(
        "support_final_rules_v4",
        None
    )

    checkpoint.setdefault(
        "support_final_v4",
        None
    )


    checkpoint.setdefault(
        "anti_final_core_v4",
        None
    )

    checkpoint.setdefault(
        "anti_final_rules_v4",
        None
    )

    checkpoint.setdefault(
        "anti_final_v4",
        None
    )


    base.save_checkpoint(
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

        "support_final_core_v4",

        "support_final_rules_v4",

        "support_final_v4",

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

        "anti_final_core_v4",

        "anti_final_rules_v4",

        "anti_final_v4",

        ANTI_OUTPUT_FILE
    )


    print()
    print("=" * 72)

    print(
        "原因体系发现 V4 完成"
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
        "下一步不要直接对5352条数据分类。"
    )

    print(
        "先人工审查支持和反对两个V4候选体系。"
    )


if __name__ == "__main__":

    main()