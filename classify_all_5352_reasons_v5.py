import csv
import json
import time
import urllib.request
from pathlib import Path
from collections import Counter


# =========================================================
# 基本路径
# =========================================================

BASE_DIR = Path(__file__).parent

SUPPORT_FILE = BASE_DIR / "Trump_support_tweets.csv"
ANTI_FILE = BASE_DIR / "Trump_anti_tweets.csv"


# =========================================================
# 正式输出文件
# =========================================================

OUTPUT_ALL = BASE_DIR / "Trump_5352_reasons_v5.csv"
OUTPUT_SUPPORT = BASE_DIR / "Trump_support_reasons_v5.csv"
OUTPUT_ANTI = BASE_DIR / "Trump_anti_reasons_v5.csv"

CHECKPOINT_FILE = BASE_DIR / "Trump_5352_reasons_v5_checkpoint.json"
LOG_FILE = BASE_DIR / "Trump_5352_reasons_v5_log.txt"
SUMMARY_FILE = BASE_DIR / "Trump_5352_reasons_v5_summary.txt"


# =========================================================
# LM Studio
# =========================================================

API_URL = "http://127.0.0.1:1234/api/v1/chat"
MODEL_NAME = "qwen3.5-4b"

BATCH_SIZE = 10
MAX_RETRIES = 2


# =========================================================
# 正式冻结的原因分类体系
# =========================================================

SUPPORT_TAXONOMY = """
S0 原因不明确 / 其他

作者确实支持Trump，
但正文没有足够信息判断为什么支持；
或者原因无法可靠归入S1-S6。


S1 个人领导力信任与情感认同

因为：
喜欢Trump、
感谢Trump、
信任Trump、
赞扬Trump本人、
认为Trump有能力、
有领导力
而支持。

例如：
"I like Trump"
"Thank you President Trump"
"Trump is the best"
"Love Trump"


S2 反建制、司法质疑与政治对手否定

因为反感或不信任：
民主党、
共和党建制派、
媒体、
检察官、
法院、
法官、
司法机构、
政治对手

而支持Trump。

也包括：
认为Trump遭遇政治迫害、
虚假指控、
司法不公。


S3 移民、边境、治安与执法

因为：
边境控制、
ICE、
驱逐非法移民、
打击犯罪、
毒品、
贩毒集团、
治安、
加强执法

而支持Trump。


S4 经济、社会与国内政策表现

因为：
经济、
就业、
通胀、
税收、
贸易、
关税、
能源、
医疗、
科技、
制造业、
社会福利、
财政政策、
其他国内政策成果

而支持Trump。


S5 保守价值观、国家认同与美国优先

因为：
保守主义、
传统价值、
宗教价值、
America First、
爱国主义、
国家认同、
反社会主义、
反共产主义

而支持Trump。

注意：
S5不能作为普通支持Trump的兜底类别。


S6 外交、战争与国家安全

因为：
外交政策、
战争、
外国政府、
军事力量、
国防、
国家安全、
美国国际实力、
国际关系

而支持Trump。
"""


ANTI_TAXONOMY = """
A0 原因不明确 / 其他

作者确实反对Trump，
但只有：
辱骂、
厌恶、
嘲笑、
讽刺、
外貌攻击

而没有更具体的反对理由。

例如：
"fuck Trump"
"Trump sucks"
"Trump is disgusting"


A1 个人品行、道德与违法犯罪指控

因为Trump本人的：
腐败、
欺诈、
犯罪、
性犯罪、
严重道德问题、
具体个人不当行为

而反对。


A2 法治、司法与民主程序争议

因为：
法院、
司法、
法律责任、
刑事案件、
选举程序、
选举诚信、
民主制度、
权力滥用、
不遵守法院裁决

而反对。


A3 治理能力、行政表现与信息诚信

因为认为Trump：
治理能力不足、
认知能力不足、
行政混乱、
决策荒谬、
撒谎、
误导公众、
传播错误信息

而反对。


A4 经济民生、社会福利与国内政策损害

因为：
物价、
油价、
就业、
收入、
通胀、
经济衰退、
贸易、
福利、
医疗、
其他国内经济政策损害

而反对。


A5 外交、战争与国家安全担忧

因为：
战争、
外交政策、
外国领导人、
俄罗斯、
中国、
中东、
军事行动、
核风险、
国际安全、
国际信誉

而反对。


A6 歧视、移民与群体权利争议

因为：
移民政策、
种族、
族裔、
宗教、
性别、
性取向、
LGBTQ+、
其他群体权利、
歧视性政策或言论

而反对。

注意：
文本中只是出现：
MAGA、
gay、
Israel、
Jewish、
Muslim

并不自动等于A6。

必须真正涉及：
歧视、
移民政策、
群体权利。
"""


SUPPORT_CODES = {
    "S0", "S1", "S2",
    "S3", "S4", "S5", "S6"
}

ANTI_CODES = {
    "A0", "A1", "A2",
    "A3", "A4", "A5", "A6"
}

VALID_CONFIDENCE = {
    "high",
    "medium",
    "low"
}


# =========================================================
# 原因中文名称
# =========================================================

CODE_NAMES = {
    "S0": "原因不明确/其他",
    "S1": "个人领导力信任与情感认同",
    "S2": "反建制、司法质疑与政治对手否定",
    "S3": "移民、边境、治安与执法",
    "S4": "经济、社会与国内政策表现",
    "S5": "保守价值观、国家认同与美国优先",
    "S6": "外交、国家安全与国际实力",

    "A0": "原因不明确/其他",
    "A1": "个人品行、道德与违法犯罪指控",
    "A2": "法治、司法与民主程序争议",
    "A3": "治理能力、行政表现与信息诚信",
    "A4": "经济民生、社会福利与国内政策损害",
    "A5": "外交、战争与国家安全担忧",
    "A6": "歧视、移民与群体权利争议"
}


# =========================================================
# CSV读取
# =========================================================

def load_csv(file_path):

    if not file_path.exists():
        raise FileNotFoundError(
            f"找不到文件：\n{file_path}"
        )

    with file_path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        raise ValueError(
            f"文件为空：{file_path}"
        )

    return rows, fields


# =========================================================
# Row ID标准化
# =========================================================

def normalize_row_id(value):

    return str(
        value if value is not None else ""
    ).strip()


# =========================================================
# 获取Tweet Content
# =========================================================

def get_tweet_content(row):

    possible_fields = [
        "Tweet Content",
        "tweet_content",
        "Content",
        "content",
        "Text",
        "text"
    ]

    for field in possible_fields:

        value = row.get(field, "")

        if value:
            return str(value).strip()

    return ""


# =========================================================
# Checkpoint
# =========================================================

def load_checkpoint():

    if not CHECKPOINT_FILE.exists():

        return {
            "support": {},
            "anti": {}
        }

    try:

        with CHECKPOINT_FILE.open(
            "r",
            encoding="utf-8"
        ) as f:

            checkpoint = json.load(f)

        checkpoint.setdefault(
            "support",
            {}
        )

        checkpoint.setdefault(
            "anti",
            {}
        )

        return checkpoint

    except Exception as e:

        raise RuntimeError(
            f"Checkpoint读取失败：{e}\n"
            "不要删除旧文件，请先把报错发给我。"
        )


def save_checkpoint(checkpoint):

    temp_file = CHECKPOINT_FILE.with_suffix(
        ".tmp"
    )

    with temp_file.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            checkpoint,
            f,
            ensure_ascii=False,
            indent=2
        )

    temp_file.replace(
        CHECKPOINT_FILE
    )


# =========================================================
# 日志
# =========================================================

def append_log(title, content):

    with LOG_FILE.open(
        "a",
        encoding="utf-8"
    ) as f:

        f.write("\n")
        f.write("=" * 80)
        f.write("\n")
        f.write(title)
        f.write("\n")
        f.write("=" * 80)
        f.write("\n\n")
        f.write(content)
        f.write("\n")


# =========================================================
# LM Studio调用
# =========================================================

def call_model(
    prompt,
    max_output_tokens
):

    body = {

        "model": MODEL_NAME,

        "input": prompt,

        "system_prompt": (
            "You are a careful social-media research "
            "coding assistant. "
            "The stance label has already been assigned. "
            "Your task is only to code the main reason "
            "expressed in the supplied tweet. "
            "Judge only the supplied text. "
            "Do not use outside political knowledge. "
            "Do not invent reasons that are not present. "
            "Return only the requested JSON."
        ),

        "reasoning": "off",

        "temperature": 0,

        "max_output_tokens":
            max_output_tokens,

        "store": False
    }

    data = json.dumps(
        body,
        ensure_ascii=False
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
        timeout=600
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

    message_parts = []

    for item in result.get(
        "output",
        []
    ):

        if item.get("type") == "message":

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

    stats = result.get(
        "stats",
        {}
    )

    reasoning_tokens = stats.get(
        "reasoning_output_tokens",
        "未知"
    )

    output_tokens = stats.get(
        "total_output_tokens",
        "未知"
    )

    return (
        content,
        reasoning_tokens,
        output_tokens
    )


# =========================================================
# JSON解析
# =========================================================

def extract_json_array(text):

    cleaned = text.strip()

    if cleaned.startswith("```"):

        lines = cleaned.splitlines()

        if lines:
            lines = lines[1:]

        if (
            lines
            and
            lines[-1].strip().startswith("```")
        ):
            lines = lines[:-1]

        cleaned = "\n".join(
            lines
        ).strip()

    start = cleaned.find("[")
    end = cleaned.rfind("]")

    if (
        start == -1
        or end == -1
        or end <= start
    ):
        raise ValueError(
            "没有找到完整JSON数组。"
        )

    data = json.loads(
        cleaned[start:end + 1]
    )

    if not isinstance(data, list):
        raise ValueError(
            "模型返回内容不是JSON数组。"
        )

    return data


# =========================================================
# 正式Stage 2 Prompt
# =========================================================

def build_reason_prompt(
    rows,
    stance
):

    if stance == "support":

        taxonomy = SUPPORT_TAXONOMY
        zero_code = "S0"

        stance_text = (
            "这些推文已经被上游立场分类器"
            "标记为支持 Donald Trump。"
        )

    else:

        taxonomy = ANTI_TAXONOMY
        zero_code = "A0"

        stance_text = (
            "这些推文已经被上游立场分类器"
            "标记为反对 Donald Trump。"
        )

    blocks = []

    for row in rows:

        row_id = normalize_row_id(
            row.get(
                "Row ID",
                ""
            )
        )

        tweet = get_tweet_content(
            row
        )

        blocks.append(
            f"[ROW_ID={row_id}]\n"
            f"{tweet}"
        )

    tweet_text = "\n\n".join(
        blocks
    )

    return f"""
现在只进行原因分类。

{stance_text}

不要重新判断立场。

即使你认为某些推文的立场标签可能有问题，
本阶段也不要修改标签。

你的任务只有一个：

根据推文本身，
判断作者表达出的最主要支持/反对原因。


==================================================
正式冻结的分类体系
==================================================

{taxonomy}


==================================================
核心规则
==================================================

1.

Primary Reason只选择一个。

选择最主要、
最直接、
文本证据最明确的原因。


2.

Secondary Reason不是必须填写。

只有当推文中存在第二个
独立而明确的理由时才填写。

否则：

secondary_code = ""


3.

禁止仅根据关键词分类。


4.

禁止补充推文中没有出现的政治背景。


5.

如果作者只有支持/反对态度，
但没有明确原因：

使用：

{zero_code}


6.

不要为了避免{zero_code}
而强行猜测原因。


==================================================
支持类别边界
==================================================

"Thank you Trump"
"Love Trump"
"Trump is the best"

→ S1


"Democrats fabricated cases against Trump"

→ S2


"Trump will deport illegal immigrants"

→ S3


"Trump created jobs"
"Trump cut taxes"

→ S4


"America First"
"Trump stopped socialism"

→ S5


"Trump ended the war"
"Trump is strong against foreign enemies"

→ S6


==================================================
反对类别边界
==================================================

"Fuck Trump"

→ A0


"Trump committed fraud"

→ A1


"Trump tried to overturn the election"

→ A2


"Trump lies every day"

→ A3


"Trump caused high prices"

→ A4


"Trump's war is dangerous"

→ A5


"Trump's immigration policy discriminates
against migrants"

→ A6


==================================================
特别容易混淆的类别
==================================================

纯辱骂、纯嘲笑：

→ A0


具体腐败、诈骗、犯罪、
性犯罪、严重个人道德问题：

→ A1


法院、司法、
选举程序、
民主制度：

→ A2


撒谎、
认知能力、
治理能力、
行政混乱：

→ A3


物价、
油价、
就业、
收入、
经济：

→ A4


个人喜欢、
个人赞扬、
领导力：

→ S1


攻击民主党、
媒体、
司法机构、
政治对手：

→ S2


经济和其他国内政策成果：

→ S4


保守价值观、
America First、
反社会主义：

→ S5


外交、
外国政府、
战争、
国家安全：

→ S6


==================================================
输出要求
==================================================

只输出JSON数组。

不要Markdown代码框。

不要输出JSON之外的任何内容。


格式：

[
  {{
    "row_id": "123",
    "primary_code": "S3",
    "secondary_code": "",
    "evidence": "secure the border",
    "confidence": "high",
    "note": "明确支持边境控制政策"
  }}
]


confidence只能是：

high
medium
low


evidence必须尽量复制推文中的原文。

不要编造证据。


==================================================
待分类推文
==================================================

{tweet_text}
""".strip()


# =========================================================
# 验证单条结果
# =========================================================

def validate_result(
    item,
    expected_ids,
    stance
):

    if not isinstance(
        item,
        dict
    ):
        return None

    row_id = normalize_row_id(
        item.get(
            "row_id",
            ""
        )
    )

    if row_id not in expected_ids:
        return None

    if stance == "support":

        valid_codes = SUPPORT_CODES
        zero_code = "S0"

    else:

        valid_codes = ANTI_CODES
        zero_code = "A0"

    primary = str(
        item.get(
            "primary_code",
            ""
        )
        or ""
    ).strip().upper()

    secondary = str(
        item.get(
            "secondary_code",
            ""
        )
        or ""
    ).strip().upper()

    evidence = str(
        item.get(
            "evidence",
            ""
        )
        or ""
    ).strip()

    confidence = str(
        item.get(
            "confidence",
            ""
        )
        or ""
    ).strip().lower()

    note = str(
        item.get(
            "note",
            ""
        )
        or ""
    ).strip()

    if primary not in valid_codes:
        return None

    if (
        secondary
        and
        secondary not in valid_codes
    ):
        return None

    if primary == zero_code:
        secondary = ""

    if secondary == zero_code:
        secondary = ""

    if secondary == primary:
        secondary = ""

    if confidence not in VALID_CONFIDENCE:
        confidence = "low"

    return {
        "row_id": row_id,
        "primary_code": primary,
        "secondary_code": secondary,
        "evidence": evidence,
        "confidence": confidence,
        "note": note
    }


# =========================================================
# 调用模型分类一组
# =========================================================

def classify_group(
    rows,
    stance
):

    expected_ids = {
        normalize_row_id(
            row.get(
                "Row ID",
                ""
            )
        )
        for row in rows
    }

    prompt = build_reason_prompt(
        rows,
        stance
    )

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            if len(rows) >= 8:
                max_tokens = 2200

            elif len(rows) >= 4:
                max_tokens = 1400

            else:
                max_tokens = 650

            (
                content,
                reasoning_tokens,
                output_tokens
            ) = call_model(
                prompt,
                max_tokens
            )

            print(
                f"正式回答长度："
                f"{len(content)}"
            )

            print(
                f"reasoning_output_tokens："
                f"{reasoning_tokens}"
            )

            print(
                f"total_output_tokens："
                f"{output_tokens}"
            )

            if (
                isinstance(
                    reasoning_tokens,
                    int
                )
                and
                reasoning_tokens != 0
            ):
                raise ValueError(
                    "reasoning没有成功关闭。"
                )

            if (
                isinstance(
                    output_tokens,
                    int
                )
                and
                output_tokens >= max_tokens
            ):
                raise ValueError(
                    "模型输出疑似达到上限。"
                )

            append_log(
                (
                    f"{stance.upper()} "
                    f"{sorted(expected_ids)}"
                ),
                content
            )

            data = extract_json_array(
                content
            )

            results = {}

            for item in data:

                validated = validate_result(
                    item,
                    expected_ids,
                    stance
                )

                if validated:

                    results[
                        validated["row_id"]
                    ] = validated

            if results:
                return results

            last_error = (
                "没有得到任何有效结果。"
            )

        except Exception as e:

            last_error = str(e)

            print(
                f"第 {attempt} 次失败："
                f"{e}"
            )

        if attempt < MAX_RETRIES:
            time.sleep(2)

    print(
        f"本组最终失败："
        f"{last_error}"
    )

    return {}


# =========================================================
# 处理一个立场组
# =========================================================

def process_stance(
    rows,
    stance,
    checkpoint
):

    if stance == "support":
        store = checkpoint["support"]
        chinese_name = "支持"

    else:
        store = checkpoint["anti"]
        chinese_name = "反对"

    pending = []

    for row in rows:

        row_id = normalize_row_id(
            row.get(
                "Row ID",
                ""
            )
        )

        if row_id not in store:
            pending.append(row)

    print()
    print("=" * 72)

    print(
        f"正式处理：{chinese_name}"
    )

    print(
        f"总数：{len(rows)}"
    )

    print(
        f"已有断点："
        f"{len(rows) - len(pending)}"
    )

    print(
        f"待处理：{len(pending)}"
    )

    print("=" * 72)

    if not pending:
        return

    for start in range(
        0,
        len(pending),
        BATCH_SIZE
    ):

        batch = pending[
            start:start + BATCH_SIZE
        ]

        first_id = normalize_row_id(
            batch[0]["Row ID"]
        )

        last_id = normalize_row_id(
            batch[-1]["Row ID"]
        )

        print()
        print(
            f"{chinese_name} Batch "
            f"{start // BATCH_SIZE + 1}："
            f"Row {first_id} → Row {last_id}"
        )

        batch_start = time.time()

        results = classify_group(
            batch,
            stance
        )

        for row_id, result in (
            results.items()
        ):

            store[row_id] = result

            save_checkpoint(
                checkpoint
            )

        missing = [

            row

            for row in batch

            if normalize_row_id(
                row["Row ID"]
            )
            not in store
        ]

        # =================================================
        # 5条fallback
        # =================================================

        if missing:

            print(
                f"主批次缺失："
                f"{len(missing)} 条"
            )

            still_missing = []

            for i in range(
                0,
                len(missing),
                5
            ):

                group = missing[
                    i:i + 5
                ]

                group_results = classify_group(
                    group,
                    stance
                )

                for row_id, result in (
                    group_results.items()
                ):

                    store[row_id] = result

                    save_checkpoint(
                        checkpoint
                    )

                for row in group:

                    row_id = normalize_row_id(
                        row["Row ID"]
                    )

                    if row_id not in store:
                        still_missing.append(
                            row
                        )

            # =============================================
            # 单条fallback
            # =============================================

            if still_missing:

                print(
                    f"5条补跑后仍缺失："
                    f"{len(still_missing)} 条"
                )

                print(
                    "开始单条补跑。"
                )

                for row in still_missing:

                    row_id = normalize_row_id(
                        row["Row ID"]
                    )

                    single_results = classify_group(
                        [row],
                        stance
                    )

                    if row_id in single_results:

                        store[row_id] = (
                            single_results[
                                row_id
                            ]
                        )

                        save_checkpoint(
                            checkpoint
                        )

                    else:

                        print(
                            f"警告：Row ID "
                            f"{row_id} "
                            f"最终仍失败。"
                        )

        elapsed = (
            time.time()
            - batch_start
        )

        completed = sum(
            1
            for row in rows
            if normalize_row_id(
                row["Row ID"]
            )
            in store
        )

        print(
            f"{chinese_name}进度："
            f"{completed}/{len(rows)}"
        )

        print(
            f"本批耗时："
            f"{elapsed:.2f} 秒"
        )


# =========================================================
# 添加分类字段
# =========================================================

def build_output_row(
    row,
    stance,
    result
):

    new_row = dict(row)

    if stance == "support":
        stance_chinese = "支持"
    else:
        stance_chinese = "反对"

    new_row[
        "Reason Analysis Stance"
    ] = stance_chinese

    if result:

        new_row[
            "Primary Reason Code"
        ] = result.get(
            "primary_code",
            ""
        )

        new_row[
            "Secondary Reason Code"
        ] = result.get(
            "secondary_code",
            ""
        )

        new_row[
            "Reason Evidence"
        ] = result.get(
            "evidence",
            ""
        )

        new_row[
            "Reason Confidence"
        ] = result.get(
            "confidence",
            ""
        )

        new_row[
            "Reason Model Note"
        ] = result.get(
            "note",
            ""
        )

    else:

        new_row[
            "Primary Reason Code"
        ] = "ERROR"

        new_row[
            "Secondary Reason Code"
        ] = ""

        new_row[
            "Reason Evidence"
        ] = ""

        new_row[
            "Reason Confidence"
        ] = ""

        new_row[
            "Reason Model Note"
        ] = (
            "模型原因分类未成功完成"
        )

    return new_row


# =========================================================
# 保存CSV
# =========================================================

def save_csv(
    file_path,
    rows,
    fieldnames
):

    with file_path.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore"
        )

        writer.writeheader()
        writer.writerows(rows)


# =========================================================
# 创建正式输出
# =========================================================

def save_outputs(
    support_rows,
    anti_rows,
    support_fields,
    anti_fields,
    checkpoint
):

    support_output = []

    for row in support_rows:

        row_id = normalize_row_id(
            row["Row ID"]
        )

        result = checkpoint[
            "support"
        ].get(
            row_id
        )

        support_output.append(
            build_output_row(
                row,
                "support",
                result
            )
        )

    anti_output = []

    for row in anti_rows:

        row_id = normalize_row_id(
            row["Row ID"]
        )

        result = checkpoint[
            "anti"
        ].get(
            row_id
        )

        anti_output.append(
            build_output_row(
                row,
                "anti",
                result
            )
        )

    base_fields = []

    for field in (
        support_fields
        + anti_fields
    ):

        if field not in base_fields:
            base_fields.append(field)

    extra_fields = [
        "Reason Analysis Stance",
        "Primary Reason Code",
        "Secondary Reason Code",
        "Reason Evidence",
        "Reason Confidence",
        "Reason Model Note"
    ]

    fieldnames = (
        base_fields
        + [
            field
            for field in extra_fields
            if field not in base_fields
        ]
    )

    save_csv(
        OUTPUT_SUPPORT,
        support_output,
        fieldnames
    )

    save_csv(
        OUTPUT_ANTI,
        anti_output,
        fieldnames
    )

    save_csv(
        OUTPUT_ALL,
        support_output + anti_output,
        fieldnames
    )


# =========================================================
# 正式统计
# =========================================================

def make_summary(
    support_rows,
    anti_rows,
    checkpoint
):

    support_counter = Counter()
    anti_counter = Counter()

    support_confidence = Counter()
    anti_confidence = Counter()

    support_errors = 0
    anti_errors = 0

    for row in support_rows:

        row_id = normalize_row_id(
            row["Row ID"]
        )

        result = checkpoint[
            "support"
        ].get(
            row_id
        )

        if not result:

            support_errors += 1
            continue

        support_counter[
            result["primary_code"]
        ] += 1

        support_confidence[
            result["confidence"]
        ] += 1

    for row in anti_rows:

        row_id = normalize_row_id(
            row["Row ID"]
        )

        result = checkpoint[
            "anti"
        ].get(
            row_id
        )

        if not result:

            anti_errors += 1
            continue

        anti_counter[
            result["primary_code"]
        ] += 1

        anti_confidence[
            result["confidence"]
        ] += 1

    lines = []

    lines.append(
        "=" * 72
    )

    lines.append(
        "Trump 5352 Tweets - V5 Reason Analysis"
    )

    lines.append(
        "=" * 72
    )

    lines.append("")

    lines.append(
        f"支持总数：{len(support_rows)}"
    )

    lines.append(
        f"反对总数：{len(anti_rows)}"
    )

    lines.append(
        f"合计："
        f"{len(support_rows) + len(anti_rows)}"
    )

    lines.append("")

    lines.append(
        f"支持成功："
        f"{len(support_rows) - support_errors}"
    )

    lines.append(
        f"支持ERROR："
        f"{support_errors}"
    )

    lines.append(
        f"反对成功："
        f"{len(anti_rows) - anti_errors}"
    )

    lines.append(
        f"反对ERROR："
        f"{anti_errors}"
    )

    lines.append("")

    # =====================================================
    # 支持原因
    # =====================================================

    lines.append(
        "=" * 72
    )

    lines.append(
        "支持Trump原因分布"
    )

    lines.append(
        "=" * 72
    )

    for code in [
        "S0", "S1", "S2",
        "S3", "S4", "S5", "S6"
    ]:

        count = support_counter.get(
            code,
            0
        )

        percent = (
            count
            / len(support_rows)
            * 100
        )

        lines.append(
            f"{code} "
            f"{CODE_NAMES[code]}："
            f"{count} "
            f"({percent:.2f}%)"
        )

    lines.append("")

    # =====================================================
    # 反对原因
    # =====================================================

    lines.append(
        "=" * 72
    )

    lines.append(
        "反对Trump原因分布"
    )

    lines.append(
        "=" * 72
    )

    for code in [
        "A0", "A1", "A2",
        "A3", "A4", "A5", "A6"
    ]:

        count = anti_counter.get(
            code,
            0
        )

        percent = (
            count
            / len(anti_rows)
            * 100
        )

        lines.append(
            f"{code} "
            f"{CODE_NAMES[code]}："
            f"{count} "
            f"({percent:.2f}%)"
        )

    # =====================================================
    # Top 3
    # =====================================================

    support_explicit = [
        ("S1", support_counter["S1"]),
        ("S2", support_counter["S2"]),
        ("S3", support_counter["S3"]),
        ("S4", support_counter["S4"]),
        ("S5", support_counter["S5"]),
        ("S6", support_counter["S6"])
    ]

    support_explicit.sort(
        key=lambda x: x[1],
        reverse=True
    )

    anti_explicit = [
        ("A1", anti_counter["A1"]),
        ("A2", anti_counter["A2"]),
        ("A3", anti_counter["A3"]),
        ("A4", anti_counter["A4"]),
        ("A5", anti_counter["A5"]),
        ("A6", anti_counter["A6"])
    ]

    anti_explicit.sort(
        key=lambda x: x[1],
        reverse=True
    )

    lines.append("")
    lines.append(
        "=" * 72
    )

    lines.append(
        "支持Trump Top 3明确原因"
    )

    lines.append(
        "=" * 72
    )

    for rank, (
        code,
        count
    ) in enumerate(
        support_explicit[:3],
        start=1
    ):

        lines.append(
            f"Top {rank}: "
            f"{code} "
            f"{CODE_NAMES[code]} "
            f"({count})"
        )

    lines.append("")
    lines.append(
        "=" * 72
    )

    lines.append(
        "反对Trump Top 3明确原因"
    )

    lines.append(
        "=" * 72
    )

    for rank, (
        code,
        count
    ) in enumerate(
        anti_explicit[:3],
        start=1
    ):

        lines.append(
            f"Top {rank}: "
            f"{code} "
            f"{CODE_NAMES[code]} "
            f"({count})"
        )

    # =====================================================
    # Confidence
    # =====================================================

    lines.append("")
    lines.append(
        "=" * 72
    )

    lines.append(
        "Confidence"
    )

    lines.append(
        "=" * 72
    )

    lines.append(
        "支持："
    )

    for key in [
        "high",
        "medium",
        "low"
    ]:

        lines.append(
            f"{key}: "
            f"{support_confidence.get(key, 0)}"
        )

    lines.append("")

    lines.append(
        "反对："
    )

    for key in [
        "high",
        "medium",
        "low"
    ]:

        lines.append(
            f"{key}: "
            f"{anti_confidence.get(key, 0)}"
        )

    summary = "\n".join(
        lines
    )

    with SUMMARY_FILE.open(
        "w",
        encoding="utf-8"
    ) as f:

        f.write(summary)

    return (
        summary,
        support_errors,
        anti_errors
    )


# =========================================================
# 数据检查
# =========================================================

def verify_input(
    support_rows,
    support_fields,
    anti_rows,
    anti_fields
):

    if "Row ID" not in support_fields:
        raise ValueError(
            "Trump_support_tweets.csv "
            "缺少 Row ID。"
        )

    if "Row ID" not in anti_fields:
        raise ValueError(
            "Trump_anti_tweets.csv "
            "缺少 Row ID。"
        )

    if len(support_rows) != 1902:
        raise ValueError(
            f"支持数据应为1902条，"
            f"实际为{len(support_rows)}条。"
        )

    if len(anti_rows) != 3450:
        raise ValueError(
            f"反对数据应为3450条，"
            f"实际为{len(anti_rows)}条。"
        )

    all_ids = []

    for row in support_rows + anti_rows:

        row_id = normalize_row_id(
            row.get(
                "Row ID",
                ""
            )
        )

        if not row_id:
            raise ValueError(
                "发现空Row ID。"
            )

        all_ids.append(
            row_id
        )

    if (
        len(all_ids)
        != len(set(all_ids))
    ):

        raise ValueError(
            "发现重复Row ID，"
            "停止正式运行。"
        )

    print(
        "输入数据检查通过。"
    )

    print(
        "支持：1902"
    )

    print(
        "反对：3450"
    )

    print(
        "合计：5352"
    )

    print(
        "Row ID：无重复"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 72)

    print(
        "Trump 5352 Tweets"
    )

    print(
        "正式原因分析 V5"
    )

    print(
        "Frozen Taxonomy / Stage 2 Only"
    )

    print("=" * 72)

    support_rows, support_fields = (
        load_csv(
            SUPPORT_FILE
        )
    )

    anti_rows, anti_fields = (
        load_csv(
            ANTI_FILE
        )
    )

    verify_input(
        support_rows,
        support_fields,
        anti_rows,
        anti_fields
    )

    checkpoint = (
        load_checkpoint()
    )

    # =====================================================
    # 正式支持原因分类
    # =====================================================

    process_stance(
        support_rows,
        "support",
        checkpoint
    )

    # =====================================================
    # 正式反对原因分类
    # =====================================================

    process_stance(
        anti_rows,
        "anti",
        checkpoint
    )

    # =====================================================
    # 输出最终CSV
    # =====================================================

    save_outputs(
        support_rows,
        anti_rows,
        support_fields,
        anti_fields,
        checkpoint
    )

    # =====================================================
    # 统计
    # =====================================================

    (
        summary,
        support_errors,
        anti_errors
    ) = make_summary(
        support_rows,
        anti_rows,
        checkpoint
    )

    print()
    print(summary)

    print()
    print("=" * 72)

    if (
        support_errors == 0
        and
        anti_errors == 0
    ):

        print(
            "5352 / 5352 原因分类全部完成"
        )

        print(
            "ERROR：0"
        )

    else:

        print(
            "正式运行尚未100%完成"
        )

        print(
            f"总ERROR："
            f"{support_errors + anti_errors}"
        )

        print(
            "请重新运行同一个脚本，"
            "程序会利用checkpoint继续补跑。"
        )

    print("=" * 72)

    print()

    print(
        "合并结果："
    )

    print(
        OUTPUT_ALL
    )

    print()

    print(
        "支持结果："
    )

    print(
        OUTPUT_SUPPORT
    )

    print()

    print(
        "反对结果："
    )

    print(
        OUTPUT_ANTI
    )

    print()

    print(
        "统计摘要："
    )

    print(
        SUMMARY_FILE
    )

    print()

    print(
        "Checkpoint："
    )

    print(
        CHECKPOINT_FILE
    )

    print()

    if (
        support_errors != 0
        or
        anti_errors != 0
    ):

        raise RuntimeError(
            "仍有未完成数据。"
            "请再次运行脚本继续补跑。"
        )


if __name__ == "__main__":
    main()