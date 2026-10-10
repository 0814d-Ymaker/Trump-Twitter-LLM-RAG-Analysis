import csv
import json
import re
import time
import urllib.request
from pathlib import Path


# =========================================================
# 基本设置
# =========================================================

BASE_DIR = Path(__file__).parent

INPUT_FILE = BASE_DIR / "Trump_10000_tweets.csv"

OUTPUT_FILE = BASE_DIR / "Trump_10000_classified_v3.csv"

CHECKPOINT_FILE = BASE_DIR / "Trump_10000_v3_checkpoint.csv"

RAW_FILE = BASE_DIR / "Trump_10000_v3_raw_responses.txt"

ERROR_FILE = BASE_DIR / "Trump_10000_v3_errors.txt"


API_URL = "http://127.0.0.1:1234/v1/chat/completions"

MODEL_NAME = "qwen3.5-4b"


# 正常主批次
MAIN_BATCH_SIZE = 20

# 主批次失败后，自动拆成 5 条
FALLBACK_BATCH_SIZE = 5

# 单条最终补救时，每条最多尝试次数
SINGLE_MAX_RETRIES = 2


VALID_LABELS = {
    "支持",
    "反对",
    "中立或无法判断"
}


# =========================================================
# 标签标准化
# =========================================================

def normalize_label(label):

    if label is None:
        return None

    label = str(label).strip()

    if label == "支持":
        return "支持"

    if label == "反对":
        return "反对"

    if label in {
        "中立",
        "无法判断",
        "中立或无法判断",
        "中立/无法判断",
        "中立／无法判断"
    }:
        return "中立或无法判断"

    return None


# =========================================================
# 自动寻找推文正文列
# =========================================================

def find_content_column(fieldnames):

    candidates = [
        "Tweet Content",
        "Content",
        "Tweet",
        "Text",
        "tweet_content",
        "tweet_text",
        "content",
        "text"
    ]

    normalized = {
        name.strip().lower(): name
        for name in fieldnames
        if name
    }

    for candidate in candidates:

        key = candidate.lower()

        if key in normalized:
            return normalized[key]

    raise ValueError(
        "无法自动找到推文正文列。\n"
        f"CSV 实际列名为：{fieldnames}"
    )


# =========================================================
# 自动寻找真正的 CSV 表头
# =========================================================

def find_header_index(lines):

    content_candidates = {
        "tweet content",
        "content",
        "tweet",
        "text",
        "tweet_content",
        "tweet_text"
    }

    max_scan = min(
        100,
        len(lines)
    )

    for index in range(max_scan):

        try:

            row = next(
                csv.reader(
                    [lines[index]]
                )
            )

        except Exception:
            continue

        normalized = {
            cell.strip().lower()
            for cell in row
            if cell
        }

        if normalized & content_candidates:
            return index

    return None


# =========================================================
# 读取 10,000 条推文
# =========================================================

def load_tweets(file_path):

    with file_path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        lines = f.readlines()


    if not lines:

        raise ValueError(
            "CSV 文件为空。"
        )


    header_index = find_header_index(
        lines
    )


    if header_index is None:

        print()
        print("没有找到真正的 CSV 表头。")
        print("文件前 20 行：")
        print("=" * 70)

        for i, line in enumerate(
            lines[:20],
            start=1
        ):
            print(
                f"{i}: {line.rstrip()}"
            )

        print("=" * 70)

        raise ValueError(
            "无法识别 CSV 表头。"
        )


    print(
        f"检测到真正的 CSV 表头位于第 "
        f"{header_index + 1} 行"
    )


    data_lines = lines[
        header_index:
    ]


    reader = csv.DictReader(
        data_lines
    )


    if not reader.fieldnames:

        raise ValueError(
            "找到表头后仍无法读取字段。"
        )


    original_fieldnames = list(
        reader.fieldnames
    )


    print(
        f"CSV 字段："
        f"{original_fieldnames}"
    )


    content_column = find_content_column(
        original_fieldnames
    )


    print(
        f"检测到推文正文列："
        f"{content_column}"
    )


    tweets = []


    for row_number, row in enumerate(
        reader,
        start=1
    ):

        content = (
            row.get(
                content_column,
                ""
            )
            or ""
        )

        content = content.strip()

        if not content:
            continue


        tweets.append({
            "Row ID": row_number,
            "Content": content,
            "Original Row": row
        })


    return (
        tweets,
        content_column,
        original_fieldnames
    )


# =========================================================
# v3 提示词
# =========================================================

def build_prompt(batch):

    tweet_text = []


    for tweet in batch:

        tweet_text.append(
            f"[Tweet {tweet['Row ID']}]\n"
            f"{tweet['Content']}"
        )


    joined_tweets = "\n\n".join(
        tweet_text
    )


    prompt = f"""
你需要判断下面每一条 Twitter/X 推文中，
发帖者本人对 Donald Trump 的立场。

只能使用当前推文本身的信息，
不要使用外部政治知识，
不要猜测作者身份。

可用标签只有三个：

支持
反对
中立或无法判断


==================================================
核心分类规则
==================================================

【支持】

如果作者明确：

- 赞扬 Trump
- 支持 Trump
- 为 Trump 辩护
- 为 Trump 的行为或立场进行正面辩解
- 明确认同 Trump
- 明确站在 Trump 一方攻击其反对者
- 对 Trump 的领导、能力、政策或行为作明显正面评价

则判断为：

支持


【反对】

如果作者明确：

- 批评 Trump
- 贬低 Trump
- 嘲讽 Trump
- 指责 Trump
- 谴责 Trump
- 将 Trump 与欺骗、犯罪、失败、腐败、
  危害等明显负面评价直接联系
- 对 Trump 的行为、政策或能力作明显负面评价

则判断为：

反对


【中立或无法判断】

如果：

- 只是陈述事实
- 只是报道新闻
- 只是转述别人说的话
- 只是引用某人的观点
- 只提到 Trump，但没有明显评价
- 作者真实立场无法从文本本身确定
- 关键代词或指代对象不清楚
- 同一句话存在多种合理解释，
  且无法判断哪一种才是作者真正意思

则判断为：

中立或无法判断


==================================================
特别注意
==================================================

1. 判断的是“发帖者本人”的态度。

如果作者只是在转述：

“某某人说 Trump 很好”

不能自动判作者支持 Trump。

如果作者只是在转述：

“某某人说 Trump 很坏”

也不能自动判作者反对 Trump。


2. 但是不要过度保守。

如果文本中已经存在明显的赞扬、批评、
讽刺、指责或辩护，
就应该按照文本表达出的总体态度分类。

不要因为一句话不是直接写：

“I support Trump”

或：

“I oppose Trump”

就强行判成中立。


3. 讽刺需要结合文本本身判断。

例如明显使用：

pathetic
criminal
fraud
disgusting
joke

等词直接描述 Trump，
一般可以判断为反对。

明显赞扬 Trump 的能力、成就、
领导力或行为，
一般可以判断为支持。


4. 反问本身不自动代表支持或反对。

必须看反问是否从文本本身能够明确体现作者态度。

如果存在多种合理理解，
选择：

中立或无法判断


5. 批评或支持其他人物，
不自动等于支持或反对 Trump。

必须确认这种评价是否真正体现了作者对 Trump 的态度。


6. 如果作者在为 Trump 辩护，
即使没有直接说“我支持 Trump”，
也可以判断为支持。

例如：

反驳针对 Trump 的指控、
替 Trump 的行为解释、
明确认为批评 Trump 的人是错误的。

如果这种辩护从文本中很明确，
可以判支持。


7. 如果作者直接把 Trump 与明显负面行为联系起来，
并且语气是在批评这种行为，
可以判断为反对。

不要因为它表面上也包含“事实陈述”
就自动判为中立。


8. 如果推文同时存在正面和负面评价：

判断作者的主要、总体态度。

只有当正负态度确实无法区分主次时，
才判：

中立或无法判断


9. 每一条推文必须独立判断。

不要让同一批中其他推文的情绪、
主题或政治立场影响当前推文。


10. 如果真的无法判断，
才使用：

中立或无法判断

不要强行猜测，
但也不要把这个标签作为默认答案。


==================================================
内部判断顺序
==================================================

每条推文在内部依次检查：

1. 这句话真正评价的是谁？
2. 是作者自己的态度，还是在转述别人？
3. 对 Trump 有没有明确正面或负面评价？
4. 有没有明确替 Trump 辩护或攻击 Trump？
5. 有没有明显讽刺？
6. 有没有重要的指代不清或多义性？
7. 综合全文判断作者总体立场。

请在内部完成这些判断，
但不要输出分析过程。


==================================================
输出格式
==================================================

只输出：

TweetID|标签

例如：

1|支持
2|反对
3|中立或无法判断

不要输出理由。
不要输出解释。
不要输出 Markdown。
不要输出代码块。
不要输出任何其他文字。

每一条输入推文必须输出且只能输出一行。


==================================================
待分类推文
==================================================

{joined_tweets}
"""

    return prompt.strip()


# =========================================================
# 调用本地 Qwen
# =========================================================

def call_qwen(batch):

    prompt = build_prompt(
        batch
    )


    request_body = {

        "model": MODEL_NAME,

        "messages": [

            {
                "role": "system",
                "content":
                    "You are a careful tweet stance classifier. "
                    "Classify each tweet independently and "
                    "follow the user's rules exactly."
            },

            {
                "role": "user",
                "content": prompt
            }
        ],

        "temperature": 0
    }


    data = json.dumps(
        request_body
    ).encode(
        "utf-8"
    )


    request = urllib.request.Request(

        API_URL,

        data=data,

        headers={
            "Content-Type":
                "application/json"
        },

        method="POST"
    )


    with urllib.request.urlopen(
        request,
        timeout=300
    ) as response:

        response_text = (
            response
            .read()
            .decode(
                "utf-8"
            )
        )


    result = json.loads(
        response_text
    )


    return (
        result[
            "choices"
        ][0][
            "message"
        ][
            "content"
        ]
    )


# =========================================================
# 鲁棒解析器
# =========================================================

def parse_response(
    response_text,
    expected_ids
):

    results = {}


    pattern = re.compile(

        r"(?:Tweet\s*)?"
        r"(\d+)"
        r"\s*"
        r"[|｜:：]"
        r"\s*"
        r"(支持|反对|中立或无法判断|中立/无法判断|中立／无法判断|中立|无法判断)",

        re.IGNORECASE
    )


    for match in pattern.finditer(
        response_text
    ):

        tweet_id = int(
            match.group(1)
        )


        stance = normalize_label(
            match.group(2)
        )


        if (
            tweet_id in expected_ids
            and stance in VALID_LABELS
        ):

            results[
                tweet_id
            ] = stance


    return results


# =========================================================
# 单条结果的额外解析
#
# 如果模型没有按：
#
# 123|反对
#
# 输出，而只输出：
#
# 反对
#
# 单条模式仍然可以识别。
# =========================================================

def parse_single_response(
    response_text,
    row_id
):

    normal_result = parse_response(
        response_text,
        {row_id}
    )


    if row_id in normal_result:

        return normal_result[
            row_id
        ]


    stripped = (
        response_text
        .strip()
    )


    exact = normalize_label(
        stripped
    )


    if exact in VALID_LABELS:

        return exact


    # 尝试识别：
    # 判断：反对
    # 标签：支持
    secondary_pattern = re.compile(

        r"(?:判断|标签|立场|结果)"
        r"\s*"
        r"[:：]?"
        r"\s*"
        r"(支持|反对|中立或无法判断|中立/无法判断|中立／无法判断|中立|无法判断)"
    )


    matches = list(
        secondary_pattern.finditer(
            response_text
        )
    )


    if matches:

        stance = normalize_label(
            matches[-1].group(1)
        )

        if stance in VALID_LABELS:
            return stance


    return None


# =========================================================
# 把列表切成固定大小的小组
# =========================================================

def chunk_list(
    items,
    size
):

    for i in range(
        0,
        len(items),
        size
    ):

        yield items[
            i:i + size
        ]


# =========================================================
# 显示 Row ID
# =========================================================

def describe_row_ids(batch):

    ids = [
        tweet["Row ID"]
        for tweet in batch
    ]


    if not ids:
        return ""


    contiguous = True


    for i in range(
        1,
        len(ids)
    ):

        if (
            ids[i]
            != ids[i - 1] + 1
        ):

            contiguous = False
            break


    if contiguous:

        return (
            f"Row "
            f"{ids[0]} - {ids[-1]}"
        )


    return (
        "Row IDs: "
        + ", ".join(
            str(x)
            for x in ids
        )
    )


# =========================================================
# 读取 checkpoint
# =========================================================

def load_checkpoint():

    completed = {}


    if not CHECKPOINT_FILE.exists():

        return completed


    with CHECKPOINT_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(
            f
        )


        for row in reader:

            try:

                row_id = int(
                    row[
                        "Row ID"
                    ]
                )

            except Exception:
                continue


            stance = normalize_label(
                row.get(
                    "Stance",
                    ""
                )
            )


            if stance in VALID_LABELS:

                completed[
                    row_id
                ] = stance


    return completed


# =========================================================
# 保存 checkpoint
# =========================================================

def save_checkpoint(
    results
):

    with CHECKPOINT_FILE.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Row ID",
                "Stance"
            ]
        )


        writer.writeheader()


        for row_id in sorted(
            results.keys()
        ):

            writer.writerow({

                "Row ID":
                    row_id,

                "Stance":
                    results[
                        row_id
                    ]
            })


# =========================================================
# 生成最终 CSV
# =========================================================

def save_final_csv(
    tweets,
    results,
    original_fieldnames
):

    output_fields = (
        ["Row ID"]
        + list(
            original_fieldnames
        )
        + ["Trump Stance"]
    )


    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=output_fields
        )


        writer.writeheader()


        for tweet in tweets:

            row_id = tweet[
                "Row ID"
            ]


            output_row = {
                "Row ID":
                    row_id
            }


            output_row.update(
                tweet[
                    "Original Row"
                ]
            )


            output_row[
                "Trump Stance"
            ] = results.get(
                row_id,
                "ERROR"
            )


            writer.writerow(
                output_row
            )


# =========================================================
# 最终统计
# =========================================================

def print_final_statistics(
    results,
    total_tweets
):

    counts = {

        "支持": 0,

        "反对": 0,

        "中立或无法判断": 0
    }


    for stance in results.values():

        if stance in counts:

            counts[
                stance
            ] += 1


    print()
    print("=" * 72)
    print("最终立场统计")
    print("=" * 72)


    for label in [
        "支持",
        "反对",
        "中立或无法判断"
    ]:

        count = counts[
            label
        ]


        percentage = (
            count
            / total_tweets
            * 100
        )


        print(
            f"{label}："
            f"{count} "
            f"({percentage:.2f}%)"
        )


# =========================================================
# 日志写入
# =========================================================

def write_raw_log(
    raw_file,
    stage,
    requested_batch,
    response_text
):

    raw_file.write(
        "-" * 70
        + "\n"
    )

    raw_file.write(
        f"Stage: {stage}\n"
    )

    raw_file.write(
        f"{describe_row_ids(requested_batch)}\n\n"
    )

    raw_file.write(
        response_text
        + "\n\n"
    )

    raw_file.flush()


# =========================================================
# 主批 20 → 小组 5 → 单条 1
# =========================================================

def process_with_fallback(
    batch,
    all_results,
    raw_file,
    error_file
):

    expected_ids = {
        tweet["Row ID"]
        for tweet in batch
    }


    local_results = {}


    # =====================================================
    # 第一阶段：
    # 20 条主批次只尝试一次
    # =====================================================

    try:

        response_text = call_qwen(
            batch
        )


        write_raw_log(
            raw_file,
            "MAIN-20",
            batch,
            response_text
        )


        parsed = parse_response(
            response_text,
            expected_ids
        )


        local_results.update(
            parsed
        )


    except Exception as e:

        print(
            f"主批请求异常：{e}"
        )


        write_raw_log(
            raw_file,
            "MAIN-20 ERROR",
            batch,
            repr(e)
        )


    # 保存当前成功项
    all_results.update(
        local_results
    )

    save_checkpoint(
        all_results
    )


    missing_ids = (
        expected_ids
        - set(
            local_results.keys()
        )
    )


    # =====================================================
    # 如果 20 条已经全部成功
    # 直接结束
    # =====================================================

    if not missing_ids:

        return local_results


    print(
        f"主批完成后："
        f"{len(local_results)}/"
        f"{len(batch)} 成功，"
        f"剩余 {len(missing_ids)} 条。"
    )


    print(
        "开始自动拆成最多 5 条一组补跑..."
    )


    # =====================================================
    # 第二阶段：
    # 把缺失数据拆成 5 条一组
    # =====================================================

    missing_tweets = [

        tweet

        for tweet in batch

        if tweet["Row ID"]
        in missing_ids
    ]


    for subgroup in chunk_list(
        missing_tweets,
        FALLBACK_BATCH_SIZE
    ):

        subgroup_ids = {
            tweet["Row ID"]
            for tweet in subgroup
        }


        try:

            response_text = call_qwen(
                subgroup
            )


            write_raw_log(
                raw_file,
                "FALLBACK-5",
                subgroup,
                response_text
            )


            parsed = parse_response(
                response_text,
                subgroup_ids
            )


            local_results.update(
                parsed
            )


            all_results.update(
                parsed
            )


            save_checkpoint(
                all_results
            )


            print(
                f"5条补跑："
                f"{len(parsed)}/"
                f"{len(subgroup)} 成功 "
                f"({describe_row_ids(subgroup)})"
            )


        except Exception as e:

            print(
                f"5条补跑异常：{e}"
            )


            write_raw_log(
                raw_file,
                "FALLBACK-5 ERROR",
                subgroup,
                repr(e)
            )


    # =====================================================
    # 再检查还有哪些没有成功
    # =====================================================

    missing_ids = (
        expected_ids
        - set(
            local_results.keys()
        )
    )


    if not missing_ids:

        print(
            "5条分组补跑后已经全部成功。"
        )

        return local_results


    print(
        f"5条补跑后仍剩余 "
        f"{len(missing_ids)} 条。"
    )


    print(
        "开始逐条补跑..."
    )


    # =====================================================
    # 第三阶段：
    # 剩余项目逐条跑
    # =====================================================

    remaining_tweets = [

        tweet

        for tweet in batch

        if tweet["Row ID"]
        in missing_ids
    ]


    for tweet in remaining_tweets:

        row_id = tweet[
            "Row ID"
        ]


        success = False


        for attempt in range(
            1,
            SINGLE_MAX_RETRIES + 1
        ):

            try:

                response_text = call_qwen(
                    [tweet]
                )


                write_raw_log(
                    raw_file,
                    (
                        f"FALLBACK-1 "
                        f"attempt {attempt}"
                    ),
                    [tweet],
                    response_text
                )


                stance = parse_single_response(
                    response_text,
                    row_id
                )


                if stance in VALID_LABELS:

                    local_results[
                        row_id
                    ] = stance


                    all_results[
                        row_id
                    ] = stance


                    save_checkpoint(
                        all_results
                    )


                    print(
                        f"单条补跑成功："
                        f"Row {row_id} "
                        f"-> {stance}"
                    )


                    success = True

                    break


                else:

                    print(
                        f"Row {row_id} "
                        f"第 {attempt} 次单条返回"
                        f"仍无法解析。"
                    )


            except Exception as e:

                print(
                    f"Row {row_id} "
                    f"第 {attempt} 次单条请求异常："
                    f"{e}"
                )


                write_raw_log(
                    raw_file,
                    (
                        f"FALLBACK-1 ERROR "
                        f"attempt {attempt}"
                    ),
                    [tweet],
                    repr(e)
                )


            if (
                attempt
                < SINGLE_MAX_RETRIES
            ):

                time.sleep(
                    1
                )


        if not success:

            error_file.write(
                "=" * 80
                + "\n"
            )

            error_file.write(
                f"最终无法分类 Row ID："
                f"{row_id}\n"
            )

            error_file.write(
                f"Content:\n"
                f"{tweet['Content']}\n\n"
            )

            error_file.flush()


    return local_results


# =========================================================
# 主程序
# =========================================================

def main():

    # -----------------------------------------------------
    # 检查输入文件
    # -----------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"找不到输入文件："
            f"{INPUT_FILE}"
        )


    # =====================================================
    # 读取数据
    # =====================================================

    (
        tweets,
        content_column,
        original_fieldnames
    ) = load_tweets(
        INPUT_FILE
    )


    total_tweets = len(
        tweets
    )


    print()
    print("=" * 72)
    print("Trump 10,000 条正式分类")
    print("方案：Batch20 v3 + 20→5→1 自动补跑")
    print("=" * 72)
    print()


    print(
        f"成功读取有效推文："
        f"{total_tweets} 条"
    )


    print(
        f"正文列："
        f"{content_column}"
    )


    print(
        f"主 Batch Size："
        f"{MAIN_BATCH_SIZE}"
    )


    # =====================================================
    # 读取旧 checkpoint
    # =====================================================

    all_results = load_checkpoint()


    valid_row_ids = {
        tweet["Row ID"]
        for tweet in tweets
    }


    all_results = {

        row_id: stance

        for row_id, stance
        in all_results.items()

        if row_id in valid_row_ids
    }


    already_done = len(
        all_results
    )


    print(
        f"断点中已经完成："
        f"{already_done} 条"
    )


    # =====================================================
    # 查找所有未完成项目
    # =====================================================

    pending_tweets = [

        tweet

        for tweet in tweets

        if tweet["Row ID"]
        not in all_results
    ]


    remaining = len(
        pending_tweets
    )


    print(
        f"本次还需要处理："
        f"{remaining} 条"
    )


    # =====================================================
    # 如果已经完成
    # =====================================================

    if remaining == 0:

        print()
        print(
            "所有推文已经分类完成。"
        )


        save_final_csv(
            tweets,
            all_results,
            original_fieldnames
        )


        print(
            f"最终文件："
            f"{OUTPUT_FILE}"
        )


        print_final_statistics(
            all_results,
            total_tweets
        )

        return


    total_batches = (
        remaining
        + MAIN_BATCH_SIZE
        - 1
    ) // MAIN_BATCH_SIZE


    print(
        f"本次主批次数："
        f"{total_batches}"
    )

    print()

    print(
        "补跑逻辑："
    )

    print(
        "20条主批失败"
        " -> 拆成5条"
        " -> 再失败则逐条处理"
    )

    print()

    print(
        "每获得新的成功结果都会更新 checkpoint。"
    )

    print()


    # =====================================================
    # 日志
    # =====================================================

    raw_file = RAW_FILE.open(
        "a",
        encoding="utf-8"
    )


    error_file = ERROR_FILE.open(
        "a",
        encoding="utf-8"
    )


    session_start = time.time()

    session_start_completed = len(
        all_results
    )


    # =====================================================
    # 正式处理
    # =====================================================

    try:

        for batch_index in range(
            total_batches
        ):

            start = (
                batch_index
                * MAIN_BATCH_SIZE
            )


            end = min(
                start
                + MAIN_BATCH_SIZE,

                remaining
            )


            batch = pending_tweets[
                start:end
            ]


            print(
                f"正在处理第 "
                f"{batch_index + 1}/"
                f"{total_batches} 主批"
            )


            print(
                describe_row_ids(
                    batch
                )
            )


            batch_start = time.time()


            batch_results = (
                process_with_fallback(
                    batch,
                    all_results,
                    raw_file,
                    error_file
                )
            )


            batch_time = (
                time.time()
                - batch_start
            )


            completed_total = len(
                all_results
            )


            progress = (
                completed_total
                / total_tweets
                * 100
            )


            print(
                f"本主批最终成功："
                f"{len(batch_results)}/"
                f"{len(batch)} 条"
            )


            print(
                f"本主批总耗时："
                f"{batch_time:.2f} 秒"
            )


            print(
                f"总进度："
                f"{completed_total}/"
                f"{total_tweets} "
                f"({progress:.2f}%)"
            )


            # =================================================
            # 动态剩余时间估算
            # =================================================

            session_new_completed = (
                completed_total
                - session_start_completed
            )


            elapsed = (
                time.time()
                - session_start
            )


            if session_new_completed > 0:

                avg_seconds = (
                    elapsed
                    / session_new_completed
                )


                still_remaining = (
                    total_tweets
                    - completed_total
                )


                estimated_hours = (
                    avg_seconds
                    * still_remaining
                    / 3600
                )


                print(
                    f"当前平均："
                    f"{avg_seconds:.2f} 秒/条"
                )


                print(
                    f"按当前速度估算剩余："
                    f"{estimated_hours:.2f} 小时"
                )


            print()


    finally:

        raw_file.close()

        error_file.close()


    # =====================================================
    # 本轮结束
    # =====================================================

    print()
    print("=" * 72)
    print("本轮处理结束")
    print("=" * 72)


    completed_total = len(
        all_results
    )


    print(
        f"成功分类："
        f"{completed_total}/"
        f"{total_tweets}"
    )


    # =====================================================
    # 全部完成
    # =====================================================

    if completed_total == total_tweets:

        save_final_csv(
            tweets,
            all_results,
            original_fieldnames
        )


        print()
        print(
            "全部推文分类完成。"
        )


        print(
            f"最终结果文件："
            f"{OUTPUT_FILE}"
        )


        print_final_statistics(
            all_results,
            total_tweets
        )


    else:

        remaining_count = (
            total_tweets
            - completed_total
        )


        print()

        print(
            f"仍有 "
            f"{remaining_count} 条"
            f"没有成功分类。"
        )


        print(
            "重新运行本程序，"
            "会自动继续处理这些缺失项。"
        )


        print(
            f"Checkpoint："
            f"{CHECKPOINT_FILE}"
        )


# =========================================================
# 启动
# =========================================================

if __name__ == "__main__":

    main()