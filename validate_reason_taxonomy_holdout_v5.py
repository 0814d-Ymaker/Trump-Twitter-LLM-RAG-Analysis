import csv
import json
import random
import time
import urllib.request
from pathlib import Path
from collections import Counter


# =========================================================
# 基本设置
# =========================================================

BASE_DIR = Path(__file__).parent


# =========================================================
# 全量支持 / 反对数据
# =========================================================

SUPPORT_FULL_FILE = (
    BASE_DIR / "Trump_support_tweets.csv"
)

ANTI_FULL_FILE = (
    BASE_DIR / "Trump_anti_tweets.csv"
)


# =========================================================
# 需要排除的旧数据
# =========================================================

EXCLUDE_FILES = [

    # 400条 taxonomy seed
    BASE_DIR / "reason_seed_support_200.csv",
    BASE_DIR / "reason_seed_anti_200.csv",

    # 第一轮人工审核
    BASE_DIR / "reason_taxonomy_audit_60.csv",

    # V2人工审核
    BASE_DIR / "reason_taxonomy_audit_v2_independent_60.csv",

    # V3 holdout
    BASE_DIR / "reason_taxonomy_holdout_v3_sample_40.csv",
    BASE_DIR / "reason_taxonomy_holdout_v3_40.csv",

    # V4 holdout
    BASE_DIR / "reason_taxonomy_holdout_v4_sample_40.csv",
    BASE_DIR / "reason_taxonomy_holdout_v4_40.csv",
]


# =========================================================
# V5输出文件
# =========================================================

SAMPLE_FILE = (
    BASE_DIR / "reason_taxonomy_holdout_v5_sample_40.csv"
)

OUTPUT_FILE = (
    BASE_DIR / "reason_taxonomy_holdout_v5_40.csv"
)

SUMMARY_FILE = (
    BASE_DIR / "reason_taxonomy_holdout_v5_summary.txt"
)

CHECKPOINT_FILE = (
    BASE_DIR / "reason_taxonomy_holdout_v5_checkpoint.json"
)

LOG_FILE = (
    BASE_DIR / "reason_taxonomy_holdout_v5_log.txt"
)


# =========================================================
# LM Studio设置
# =========================================================

API_URL = (
    "http://127.0.0.1:1234/api/v1/chat"
)

MODEL_NAME = "qwen3.5-4b"

BATCH_SIZE = 10

MAX_RETRIES = 2


# =========================================================
# 抽样设置
# =========================================================

RANDOM_SEED = 20261009

SUPPORT_SAMPLE_N = 20

ANTI_SAMPLE_N = 20


# =========================================================
# 原因分类体系
# =========================================================

SUPPORT_TAXONOMY = """
S0 原因不明确 / 其他

作者确实支持Trump，
但正文没有足够信息判断为什么支持；
或者原因无法可靠归入S1-S6。

注意：
明确的个人喜欢、赞扬、感谢，
不是S0。


S1 个人领导力信任与情感认同

因为：
喜欢Trump、
感谢Trump、
信任Trump、
赞扬Trump本人、
认为Trump很强、
有能力、
有领导力
而支持。


例如：

"I like Trump"

"Thank you President Trump"

"Trump is the best"

"Go Trump"


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

S5不能作为
“普通支持Trump”
的兜底类别。


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

MAGA
gay
Israel
Jewish
Muslim

并不自动等于A6。

必须真正涉及：
歧视、
移民政策
或群体权利问题。
"""


SUPPORT_CODES = {
    "S0", "S1", "S2",
    "S3", "S4", "S5", "S6"
}

ANTI_CODES = {
    "A0", "A1", "A2",
    "A3", "A4", "A5", "A6"
}


VALID_CONSISTENCY = {
    "consistent",
    "conflict",
    "uncertain"
}


VALID_CONFIDENCE = {
    "high",
    "medium",
    "low"
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

        fields = (
            reader.fieldnames
            or []
        )

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
        value
        if value is not None
        else ""
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

        value = row.get(
            field,
            ""
        )

        if value:

            return str(
                value
            ).strip()

    return ""


# =========================================================
# 收集所有需要排除的Row ID
# =========================================================

def collect_excluded_ids():

    excluded = set()

    for file_path in EXCLUDE_FILES:

        if not file_path.exists():

            continue

        rows, fields = load_csv(
            file_path
        )

        if "Row ID" not in fields:

            continue

        for row in rows:

            row_id = normalize_row_id(
                row.get(
                    "Row ID",
                    ""
                )
            )

            if row_id:

                excluded.add(
                    row_id
                )

    return excluded


# =========================================================
# 创建V5新的40条holdout
# =========================================================

def create_new_holdout():

    support_rows, support_fields = (
        load_csv(
            SUPPORT_FULL_FILE
        )
    )

    anti_rows, anti_fields = (
        load_csv(
            ANTI_FULL_FILE
        )
    )

    if "Row ID" not in support_fields:

        raise ValueError(
            "支持数据缺少Row ID。"
        )

    if "Row ID" not in anti_fields:

        raise ValueError(
            "反对数据缺少Row ID。"
        )

    excluded_ids = (
        collect_excluded_ids()
    )

    support_pool = [

        row

        for row in support_rows

        if normalize_row_id(
            row.get(
                "Row ID",
                ""
            )
        )
        not in excluded_ids
    ]

    anti_pool = [

        row

        for row in anti_rows

        if normalize_row_id(
            row.get(
                "Row ID",
                ""
            )
        )
        not in excluded_ids
    ]

    print()

    print(
        f"原支持数据："
        f"{len(support_rows)}"
    )

    print(
        f"原反对数据："
        f"{len(anti_rows)}"
    )

    print(
        f"排除旧Row ID："
        f"{len(excluded_ids)}"
    )

    print(
        f"支持V5候选池："
        f"{len(support_pool)}"
    )

    print(
        f"反对V5候选池："
        f"{len(anti_pool)}"
    )

    if (
        len(support_pool)
        <
        SUPPORT_SAMPLE_N
    ):

        raise ValueError(
            "支持候选不足。"
        )

    if (
        len(anti_pool)
        <
        ANTI_SAMPLE_N
    ):

        raise ValueError(
            "反对候选不足。"
        )

    rng = random.Random(
        RANDOM_SEED
    )

    support_selected = rng.sample(
        support_pool,
        SUPPORT_SAMPLE_N
    )

    anti_selected = rng.sample(
        anti_pool,
        ANTI_SAMPLE_N
    )

    result = []

    for index, row in enumerate(
        support_selected,
        start=1
    ):

        result.append({

            "Sample ID":
                f"V5S{index:03d}",

            "Row ID":
                normalize_row_id(
                    row.get(
                        "Row ID",
                        ""
                    )
                ),

            "Trump Stance":
                "支持",

            "Tweet Content":
                get_tweet_content(
                    row
                )
        })

    for index, row in enumerate(
        anti_selected,
        start=1
    ):

        result.append({

            "Sample ID":
                f"V5A{index:03d}",

            "Row ID":
                normalize_row_id(
                    row.get(
                        "Row ID",
                        ""
                    )
                ),

            "Trump Stance":
                "反对",

            "Tweet Content":
                get_tweet_content(
                    row
                )
        })

    if len(result) != 40:

        raise RuntimeError(
            "V5样本不是40条。"
        )

    row_ids = [

        row["Row ID"]

        for row in result
    ]

    if (
        len(row_ids)
        !=
        len(set(row_ids))
    ):

        raise RuntimeError(
            "V5样本Row ID重复。"
        )

    if (
        set(row_ids)
        &
        excluded_ids
    ):

        raise RuntimeError(
            "V5样本与旧数据发生重叠。"
        )

    return (
        result,
        len(support_pool),
        len(anti_pool),
        len(excluded_ids)
    )


# =========================================================
# 保存 / 复用V5 sample
# =========================================================

def load_or_create_sample():

    if SAMPLE_FILE.exists():

        rows, fields = (
            load_csv(
                SAMPLE_FILE
            )
        )

        if len(rows) == 40:

            print(
                "检测到已有V5 sample，"
                "直接复用。"
            )

            return (
                rows,
                None,
                None,
                None
            )

    (
        rows,
        support_pool_n,
        anti_pool_n,
        excluded_n
    ) = create_new_holdout()

    fields = [
        "Sample ID",
        "Row ID",
        "Trump Stance",
        "Tweet Content"
    ]

    with SAMPLE_FILE.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    return (
        rows,
        support_pool_n,
        anti_pool_n,
        excluded_n
    )


# =========================================================
# checkpoint
# =========================================================

def load_checkpoint():

    if not CHECKPOINT_FILE.exists():

        return {
            "stage1": {},
            "stage2": {}
        }

    try:

        with CHECKPOINT_FILE.open(
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        data.setdefault(
            "stage1",
            {}
        )

        data.setdefault(
            "stage2",
            {}
        )

        return data

    except Exception:

        print(
            "V5 checkpoint读取失败，"
            "重新创建。"
        )

        return {
            "stage1": {},
            "stage2": {}
        }


def save_checkpoint(checkpoint):

    with CHECKPOINT_FILE.open(
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

    with LOG_FILE.open(
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
# LM Studio调用
# =========================================================

def call_model(
    prompt,
    max_output_tokens
):

    body = {

        "model":
            MODEL_NAME,

        "input":
            prompt,

        "system_prompt":
            (
                "You are a careful social-media "
                "research coding assistant. "
                "Judge the author's expressed meaning "
                "from the supplied text only. "
                "Do not use outside political knowledge. "
                "Do not be unnecessarily cautious. "
                "When the author's approval or criticism "
                "is reasonably clear from ordinary language, "
                "make a decision instead of choosing uncertain. "
                "Pay attention to quotation, negation, sarcasm, "
                "and who is expressing each opinion."
            ),

        "reasoning":
            "off",

        "temperature":
            0,

        "max_output_tokens":
            max_output_tokens,

        "store":
            False
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

        if (
            item.get(
                "type"
            )
            == "message"
        ):

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

    reasoning_tokens = (
        stats.get(
            "reasoning_output_tokens",
            "未知"
        )
    )

    output_tokens = (
        stats.get(
            "total_output_tokens",
            "未知"
        )
    )

    return (
        content,
        reasoning_tokens,
        output_tokens
    )


# =========================================================
# 提取JSON数组
# =========================================================

def extract_json_array(text):

    cleaned = (
        text.strip()
    )

    if cleaned.startswith(
        "```"
    ):

        lines = (
            cleaned.splitlines()
        )

        lines = lines[1:]

        if (
            lines
            and
            lines[-1]
            .strip()
            .startswith(
                "```"
            )
        ):

            lines = lines[:-1]

        cleaned = (
            "\n".join(
                lines
            )
            .strip()
        )

    start = cleaned.find(
        "["
    )

    end = cleaned.rfind(
        "]"
    )

    if (
        start == -1
        or
        end == -1
        or
        end <= start
    ):

        raise ValueError(
            "没有完整JSON数组。"
        )

    data = json.loads(
        cleaned[
            start:end + 1
        ]
    )

    if not isinstance(
        data,
        list
    ):

        raise ValueError(
            "返回内容不是JSON数组。"
        )

    return data


# =========================================================
# STAGE 1
# V5重点修改：
# 只判断立场一致性
# uncertain使用门槛提高
# =========================================================

def build_stage1_prompt(
    rows,
    stance
):

    if stance == "support":

        given_stance = (
            "支持 Donald Trump"
        )

        opposite = (
            "反对 Donald Trump"
        )

    else:

        given_stance = (
            "反对 Donald Trump"
        )

        opposite = (
            "支持 Donald Trump"
        )

    blocks = []

    for row in rows:

        blocks.append(

            f"[{row['Sample ID']}]\n"
            f"{row['Tweet Content']}"
        )

    tweet_text = (
        "\n\n".join(
            blocks
        )
    )

    return f"""
你的任务只有一个：

判断推文作者自己的真实态度，
是否与给定标签：

{given_stance}

一致。


不要进行原因分类。

不要输出S1-S6或A1-A6。


==================================================
三个类别
==================================================

consistent

作者自己的整体态度
明显或合理地支持给定标签：

{given_stance}


conflict

作者自己的整体态度
明显或合理地更接近：

{opposite}


uncertain

只有在真正无法判断时使用。


==================================================
V5最重要规则
==================================================

不要因为文本不是：

"I support Trump"

或：

"I hate Trump"

这种直接句式，

就判uncertain。


社交媒体通常不会使用正式声明。

只要作者通过：

赞扬、
感谢、
维护、
辩护、
认同政策、
攻击Trump的政治对手、
表达对Trump获胜的高兴、
表达对Trump政策成功的认可

清楚表现支持，

就可以判断支持。


同样，

只要作者通过：

辱骂Trump、
批评Trump、
指控Trump犯罪、
说Trump撒谎、
说Trump无能、
说Trump造成经济损害、
说Trump发动战争、
说Trump伤害某些群体

清楚表现负面态度，

就可以判断反对。


==================================================
uncertain的严格使用条件
==================================================

只有以下情况才使用uncertain：

1.
纯新闻标题或事实转发，
作者完全没有评价。


2.
只有第三方观点，
无法知道作者是否同意。


3.
真正无法理解的讽刺。


4.
文本严重截断，
关键态度缺失。


5.
同一条推文同时出现
强烈支持和强烈反对，
且无法判断最终方向。


不要仅仅因为：

文本复杂、
有多个句子、
出现别人的名字、
包含政治背景、
不是直接说“I support”

就判uncertain。


==================================================
明确支持的例子
==================================================

"Love Trump"

→ 支持


"Trump wins again"

→ 支持


"Trump loves his country"

→ 支持


"Trump Administration will protect women"

→ 支持


"I voted for Trump because..."

→ 支持


"Trump is strong and cannot be bought"

→ 支持


"Democrats tried to destroy Trump
but they had no evidence"

→ 支持


"Thank you Trump"

→ 支持


==================================================
明确反对的例子
==================================================

"Trump is a fraud"

→ 反对


"Trump destroyed the economy"

→ 反对


"Trump is stupid"

→ 反对


"Trump is a rapist"

→ 反对


"Trump started this war"

→ 反对


"Trump lied"

→ 反对


"Trump only cares about himself"

→ 反对


==================================================
引用和反驳
==================================================

必须判断作者是否认同被引用内容。


例如：

"They say Trump is a pedophile,
but Democrats ignored the real criminals."

这里：

"Trump is a pedophile"

是作者正在引用的说法，
不能单独用来判断作者反对Trump。


如果整体语气是在保护Trump、
攻击指控Trump的人，

则可能是支持。


==================================================
另一个例子
==================================================

文本：

"Everyone calls MAGA Hitler.
That's all they have."

作者是在嘲讽：

把MAGA称为Hitler的人。

不是作者自己说：

MAGA = Hitler。


==================================================
否定词
==================================================

必须正确处理：

no
not
never
isn't
wasn't
didn't


例如：

"No, Trump did NOT destroy America."

这是在维护Trump，

不是在说Trump毁掉美国。


==================================================
讽刺
==================================================

只有真正无法确定讽刺方向时，
才用uncertain。


如果普通读者可以从：

语气、
上下文、
emoji、
前后句

判断作者是在赞扬还是攻击，

就应该做出判断。


==================================================
Confidence
==================================================

high

态度很清楚。


medium

整体方向可以判断，
但存在一定语境复杂性。


low

真正接近uncertain边界。


==================================================
Evidence
==================================================

直接复制最能证明
作者态度的短片段。


==================================================
输出格式
==================================================

只输出JSON数组。

不要Markdown。

不要写额外解释。


格式：

[
  {{
    "sample_id": "V5S001",
    "stance_consistency": "consistent",
    "evidence": "Love Trump",
    "confidence": "high",
    "note": "作者直接表达对Trump的喜爱"
  }}
]


==================================================
待判断推文
==================================================

{tweet_text}
""".strip()


# =========================================================
# Stage1结果验证
# =========================================================

def validate_stage1_item(
    item,
    expected_ids
):

    if not isinstance(
        item,
        dict
    ):

        return None

    sample_id = str(
        item.get(
            "sample_id",
            ""
        )
    ).strip()

    if sample_id not in expected_ids:

        return None

    consistency = str(
        item.get(
            "stance_consistency",
            ""
        )
    ).strip().lower()

    if (
        consistency
        not in VALID_CONSISTENCY
    ):

        return None

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

    if (
        confidence
        not in VALID_CONFIDENCE
    ):

        confidence = "low"

    return {

        "sample_id":
            sample_id,

        "stance_consistency":
            consistency,

        "evidence":
            evidence,

        "confidence":
            confidence,

        "note":
            note
    }


# =========================================================
# Stage1调用
# =========================================================

def classify_stage1_group(
    rows,
    stance
):

    expected_ids = {

        row["Sample ID"]

        for row in rows
    }

    prompt = (
        build_stage1_prompt(
            rows,
            stance
        )
    )

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            if len(rows) >= 8:

                max_tokens = 1800

            elif len(rows) >= 4:

                max_tokens = 1100

            else:

                max_tokens = 500

            (
                content,
                reasoning_tokens,
                output_tokens
            ) = call_model(
                prompt,
                max_tokens
            )

            print(
                f"Stage1回答长度："
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
                    "reasoning未关闭。"
                )

            if (
                isinstance(
                    output_tokens,
                    int
                )
                and
                output_tokens
                >= max_tokens
            ):

                raise ValueError(
                    "Stage1疑似被截断。"
                )

            append_log(
                (
                    f"STAGE1 "
                    f"{stance} "
                    f"{sorted(expected_ids)}"
                ),
                content
            )

            data = extract_json_array(
                content
            )

            results = {}

            for item in data:

                validated = (
                    validate_stage1_item(
                        item,
                        expected_ids
                    )
                )

                if validated:

                    results[
                        validated[
                            "sample_id"
                        ]
                    ] = validated

            if results:

                return results

            last_error = (
                "没有有效Stage1结果。"
            )

        except Exception as e:

            last_error = str(e)

            print(
                f"Stage1第{attempt}次失败："
                f"{e}"
            )

        if attempt < MAX_RETRIES:

            time.sleep(2)

    print(
        f"Stage1最终失败："
        f"{last_error}"
    )

    return {}


# =========================================================
# STAGE 2
# 和V4保持同样思路
# 只判断原因
# =========================================================

def build_stage2_prompt(
    rows,
    stance
):

    if stance == "support":

        stance_text = (
            "这些推文已经确认是在支持Trump。"
        )

        taxonomy = (
            SUPPORT_TAXONOMY
        )

        zero_code = "S0"

    else:

        stance_text = (
            "这些推文已经确认是在反对Trump。"
        )

        taxonomy = (
            ANTI_TAXONOMY
        )

        zero_code = "A0"

    blocks = []

    for row in rows:

        blocks.append(

            f"[{row['Sample ID']}]\n"
            f"{row['Tweet Content']}"
        )

    tweet_text = (
        "\n\n".join(
            blocks
        )
    )

    return f"""
现在只做原因分类。

{stance_text}

第一阶段已经完成立场判断。

你不要重新判断立场。


你的唯一任务：

找出作者最主要的
支持或反对原因。


==================================================
原因分类体系
==================================================

{taxonomy}


==================================================
核心规则
==================================================

1.

Primary Reason只选择
最主要、最直接的一个原因。


2.

Secondary Reason只有在
第二个独立原因非常明确时填写。

否则必须为空。


3.

不要因为出现某个关键词
就自动选择类别。


4.

不要因为作者支持或反对Trump，
就硬猜一个原因。


5.

如果只有态度，
没有明确原因：

使用：

{zero_code}


==================================================
支持组边界例子
==================================================

"Thank you President Trump"

→ S1


"Democrats fabricated cases
to stop Trump"

→ S2


"Trump will deport illegal immigrants"

→ S3


"Trump's tariffs created jobs"

→ S4


"America First"

→ S5


"Trump made America stronger
against foreign enemies"

→ S6


==================================================
反对组边界例子
==================================================

"Fuck Trump"

→ A0


"Trump committed fraud"

→ A1


"Trump ignores court rulings"

→ A2


"Trump keeps lying"

→ A3


"Trump caused high prices"

→ A4


"Trump is too friendly with Putin"

→ A5


"Trump's immigration policy
discriminates against migrants"

→ A6


==================================================
特别容易混淆
==================================================

纯辱骂：

→ A0


具体欺诈、犯罪、
严重道德问题：

→ A1


司法、法院、
选举程序：

→ A2


治理能力、
认知能力、
撒谎、误导：

→ A3


经济、物价、
就业、收入：

→ A4


Trump个人能力、
个人赞扬：

→ S1


攻击民主党、
媒体、
法院、
检察官、
政治对手：

→ S2


国内政策效果：

→ S4


保守价值、
America First：

→ S5


外交、
战争、
外国政府：

→ S6


==================================================
Evidence
==================================================

复制原文中
最能支持分类的短片段。


==================================================
Confidence
==================================================

high
medium
low


==================================================
输出
==================================================

只输出JSON数组。

格式：

[
  {{
    "sample_id": "V5S001",
    "primary_code": "S3",
    "secondary_code": "",
    "evidence": "secure the border",
    "confidence": "high",
    "note": "明确支持边境政策"
  }}
]


==================================================
待分类推文
==================================================

{tweet_text}
""".strip()


# =========================================================
# Stage2结果验证
# =========================================================

def validate_stage2_item(
    item,
    expected_ids,
    valid_codes,
    zero_code
):

    if not isinstance(
        item,
        dict
    ):

        return None

    sample_id = str(
        item.get(
            "sample_id",
            ""
        )
    ).strip()

    if sample_id not in expected_ids:

        return None

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

    if (
        confidence
        not in VALID_CONFIDENCE
    ):

        confidence = "low"

    return {

        "sample_id":
            sample_id,

        "primary_code":
            primary,

        "secondary_code":
            secondary,

        "evidence":
            evidence,

        "confidence":
            confidence,

        "note":
            note
    }


# =========================================================
# Stage2调用
# =========================================================

def classify_stage2_group(
    rows,
    stance
):

    if stance == "support":

        valid_codes = (
            SUPPORT_CODES
        )

        zero_code = "S0"

    else:

        valid_codes = (
            ANTI_CODES
        )

        zero_code = "A0"

    expected_ids = {

        row["Sample ID"]

        for row in rows
    }

    prompt = (
        build_stage2_prompt(
            rows,
            stance
        )
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
                f"Stage2回答长度："
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
                    "reasoning未关闭。"
                )

            if (
                isinstance(
                    output_tokens,
                    int
                )
                and
                output_tokens
                >= max_tokens
            ):

                raise ValueError(
                    "Stage2疑似被截断。"
                )

            append_log(
                (
                    f"STAGE2 "
                    f"{stance} "
                    f"{sorted(expected_ids)}"
                ),
                content
            )

            data = extract_json_array(
                content
            )

            results = {}

            for item in data:

                validated = (
                    validate_stage2_item(
                        item,
                        expected_ids,
                        valid_codes,
                        zero_code
                    )
                )

                if validated:

                    results[
                        validated[
                            "sample_id"
                        ]
                    ] = validated

            if results:

                return results

            last_error = (
                "没有有效Stage2结果。"
            )

        except Exception as e:

            last_error = str(e)

            print(
                f"Stage2第{attempt}次失败："
                f"{e}"
            )

        if attempt < MAX_RETRIES:

            time.sleep(2)

    print(
        f"Stage2最终失败："
        f"{last_error}"
    )

    return {}


# =========================================================
# 通用Fallback
# =========================================================

def process_with_fallback(
    rows,
    stance,
    stage,
    checkpoint
):

    if stage == 1:

        result_store = (
            checkpoint["stage1"]
        )

        classify_function = (
            classify_stage1_group
        )

    else:

        result_store = (
            checkpoint["stage2"]
        )

        classify_function = (
            classify_stage2_group
        )

    pending = [

        row

        for row in rows

        if (
            row["Sample ID"]
            not in result_store
        )
    ]

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

        print()

        print(
            f"Stage {stage} "
            f"{stance}："
            f"{batch[0]['Sample ID']} "
            f"→ "
            f"{batch[-1]['Sample ID']}"
        )

        start_time = (
            time.time()
        )

        results = (
            classify_function(
                batch,
                stance
            )
        )

        for result in (
            results.values()
        ):

            result_store[
                result["sample_id"]
            ] = result

            save_checkpoint(
                checkpoint
            )

        missing = [

            row

            for row in batch

            if (
                row["Sample ID"]
                not in results
            )
        ]

        # =================================================
        # 5条补跑
        # =================================================

        if missing:

            print(
                f"缺失 {len(missing)} 条，"
                "开始5条补跑。"
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

                group_results = (
                    classify_function(
                        group,
                        stance
                    )
                )

                for result in (
                    group_results.values()
                ):

                    result_store[
                        result["sample_id"]
                    ] = result

                    save_checkpoint(
                        checkpoint
                    )

                for row in group:

                    if (
                        row["Sample ID"]
                        not in group_results
                    ):

                        still_missing.append(
                            row
                        )

            # =============================================
            # 单条最终补跑
            # =============================================

            if still_missing:

                print(
                    f"仍缺失 "
                    f"{len(still_missing)} 条，"
                    "开始单条补跑。"
                )

                for row in still_missing:

                    single_result = (
                        classify_function(
                            [row],
                            stance
                        )
                    )

                    sample_id = (
                        row["Sample ID"]
                    )

                    if (
                        sample_id
                        in single_result
                    ):

                        result_store[
                            sample_id
                        ] = (
                            single_result[
                                sample_id
                            ]
                        )

                        save_checkpoint(
                            checkpoint
                        )

                    else:

                        print(
                            f"警告："
                            f"{sample_id} "
                            "最终失败。"
                        )

        elapsed = (
            time.time()
            - start_time
        )

        print(
            f"Stage {stage}本批耗时："
            f"{elapsed:.2f} 秒"
        )


# =========================================================
# 保存最终人工审核CSV
# =========================================================

def save_final_output(
    rows,
    checkpoint
):

    fields = [

        "Audit ID",
        "Sample ID",
        "Row ID",
        "Trump Stance",
        "Tweet Content",

        "Stance Consistency",
        "Stance Evidence",
        "Stance Confidence",
        "Stance Model Note",

        "Primary Reason Code",
        "Secondary Reason Code",
        "Reason Evidence",
        "Reason Confidence",
        "Reason Model Note",

        "Human Stance Consistency Accept",
        "Human Correct Stance Consistency",

        "Human Reason Accept",
        "Human Correct Primary Code",
        "Human Correct Secondary Code",

        "Human Note"
    ]

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        for index, row in enumerate(
            rows,
            start=1
        ):

            sample_id = (
                row["Sample ID"]
            )

            stage1 = (
                checkpoint[
                    "stage1"
                ].get(
                    sample_id,
                    {}
                )
            )

            stage2 = (
                checkpoint[
                    "stage2"
                ].get(
                    sample_id,
                    {}
                )
            )

            consistency = (
                stage1.get(
                    "stance_consistency",
                    "ERROR"
                )
            )

            # =================================================
            # conflict
            # =================================================

            if consistency == "conflict":

                primary = ""
                secondary = ""

                reason_evidence = ""
                reason_confidence = ""
                reason_note = ""

            # =================================================
            # uncertain
            # =================================================

            elif consistency == "uncertain":

                if (
                    row["Trump Stance"]
                    == "支持"
                ):

                    primary = "S0"

                else:

                    primary = "A0"

                secondary = ""

                reason_evidence = ""

                reason_confidence = ""

                reason_note = (
                    "Stage1立场不确定，"
                    "不进行具体原因分类。"
                )

            # =================================================
            # consistent
            # =================================================

            else:

                primary = (
                    stage2.get(
                        "primary_code",
                        "ERROR"
                    )
                )

                secondary = (
                    stage2.get(
                        "secondary_code",
                        ""
                    )
                )

                reason_evidence = (
                    stage2.get(
                        "evidence",
                        ""
                    )
                )

                reason_confidence = (
                    stage2.get(
                        "confidence",
                        ""
                    )
                )

                reason_note = (
                    stage2.get(
                        "note",
                        ""
                    )
                )

            writer.writerow({

                "Audit ID":
                    f"V5H{index:03d}",

                "Sample ID":
                    sample_id,

                "Row ID":
                    row["Row ID"],

                "Trump Stance":
                    row["Trump Stance"],

                "Tweet Content":
                    row["Tweet Content"],

                "Stance Consistency":
                    consistency,

                "Stance Evidence":
                    stage1.get(
                        "evidence",
                        ""
                    ),

                "Stance Confidence":
                    stage1.get(
                        "confidence",
                        ""
                    ),

                "Stance Model Note":
                    stage1.get(
                        "note",
                        ""
                    ),

                "Primary Reason Code":
                    primary,

                "Secondary Reason Code":
                    secondary,

                "Reason Evidence":
                    reason_evidence,

                "Reason Confidence":
                    reason_confidence,

                "Reason Model Note":
                    reason_note,

                "Human Stance Consistency Accept":
                    "",

                "Human Correct Stance Consistency":
                    "",

                "Human Reason Accept":
                    "",

                "Human Correct Primary Code":
                    "",

                "Human Correct Secondary Code":
                    "",

                "Human Note":
                    ""
            })


# =========================================================
# Summary
# =========================================================

def create_summary(
    rows,
    checkpoint,
    support_pool_n,
    anti_pool_n,
    excluded_n
):

    consistency_counter = (
        Counter()
    )

    reason_counter = (
        Counter()
    )

    stage1_conf_counter = (
        Counter()
    )

    stage2_conf_counter = (
        Counter()
    )

    stage1_errors = 0

    stage2_errors = 0

    stage2_expected = 0

    for row in rows:

        sample_id = (
            row["Sample ID"]
        )

        stage1 = (
            checkpoint[
                "stage1"
            ].get(
                sample_id
            )
        )

        if not stage1:

            stage1_errors += 1
            continue

        consistency = (
            stage1[
                "stance_consistency"
            ]
        )

        consistency_counter[
            consistency
        ] += 1

        stage1_conf_counter[
            stage1.get(
                "confidence",
                ""
            )
        ] += 1

        if consistency == "consistent":

            stage2_expected += 1

            stage2 = (
                checkpoint[
                    "stage2"
                ].get(
                    sample_id
                )
            )

            if not stage2:

                stage2_errors += 1
                continue

            reason_counter[
                stage2[
                    "primary_code"
                ]
            ] += 1

            stage2_conf_counter[
                stage2.get(
                    "confidence",
                    ""
                )
            ] += 1

    lines = []

    lines.append(
        "=" * 72
    )

    lines.append(
        "Reason Taxonomy Holdout V5 Two-Stage Validation"
    )

    lines.append(
        "=" * 72
    )

    lines.append("")

    lines.append(
        f"随机种子："
        f"{RANDOM_SEED}"
    )

    if excluded_n is not None:

        lines.append(
            f"排除旧Row ID："
            f"{excluded_n}"
        )

    if support_pool_n is not None:

        lines.append(
            f"支持候选池："
            f"{support_pool_n}"
        )

    if anti_pool_n is not None:

        lines.append(
            f"反对候选池："
            f"{anti_pool_n}"
        )

    lines.append(
        f"最终holdout："
        f"{len(rows)}"
    )

    lines.append("")

    lines.append(
        "Stage 1：立场一致性"
    )

    lines.append(
        f"成功："
        f"{len(rows) - stage1_errors}"
    )

    lines.append(
        f"ERROR："
        f"{stage1_errors}"
    )

    for key in [
        "consistent",
        "conflict",
        "uncertain"
    ]:

        lines.append(
            f"{key}: "
            f"{consistency_counter.get(key, 0)}"
        )

    lines.append("")

    lines.append(
        "Stage 1 Confidence："
    )

    for key in [
        "high",
        "medium",
        "low"
    ]:

        lines.append(
            f"{key}: "
            f"{stage1_conf_counter.get(key, 0)}"
        )

    lines.append("")

    lines.append(
        "Stage 2：原因分类"
    )

    lines.append(
        f"需要分类："
        f"{stage2_expected}"
    )

    lines.append(
        f"成功："
        f"{stage2_expected - stage2_errors}"
    )

    lines.append(
        f"ERROR："
        f"{stage2_errors}"
    )

    lines.append("")

    lines.append(
        "Primary Reason："
    )

    for key, value in sorted(
        reason_counter.items()
    ):

        lines.append(
            f"{key}: {value}"
        )

    lines.append("")

    lines.append(
        "Stage 2 Confidence："
    )

    for key in [
        "high",
        "medium",
        "low"
    ]:

        lines.append(
            f"{key}: "
            f"{stage2_conf_counter.get(key, 0)}"
        )

    summary = (
        "\n".join(
            lines
        )
    )

    with SUMMARY_FILE.open(
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            summary
        )

    return summary


# =========================================================
# 主程序
# =========================================================

def main():

    print("=" * 72)

    print(
        "Reason Taxonomy Holdout V5"
    )

    print(
        "Two-Stage Validation"
    )

    print(
        "Stage1判断阈值优化版"
    )

    print("=" * 72)

    # =====================================================
    # 创建新的holdout
    # =====================================================

    (
        rows,
        support_pool_n,
        anti_pool_n,
        excluded_n
    ) = load_or_create_sample()

    print()

    print(
        f"V5 Holdout："
        f"{len(rows)} 条"
    )

    print(
        "支持：20"
    )

    print(
        "反对：20"
    )

    # =====================================================
    # checkpoint
    # =====================================================

    checkpoint = (
        load_checkpoint()
    )

    support_rows = [

        row

        for row in rows

        if (
            row["Trump Stance"]
            == "支持"
        )
    ]

    anti_rows = [

        row

        for row in rows

        if (
            row["Trump Stance"]
            == "反对"
        )
    ]

    # =====================================================
    # STAGE 1
    # =====================================================

    print()

    print("=" * 72)

    print(
        "STAGE 1"
    )

    print(
        "只判断立场一致性"
    )

    print(
        "V5：减少过度uncertain"
    )

    print("=" * 72)

    process_with_fallback(
        support_rows,
        "support",
        1,
        checkpoint
    )

    process_with_fallback(
        anti_rows,
        "anti",
        1,
        checkpoint
    )

    # =====================================================
    # 检查Stage1
    # =====================================================

    stage1_success = sum(

        1

        for row in rows

        if (
            row["Sample ID"]
            in checkpoint[
                "stage1"
            ]
        )
    )

    print()

    print(
        f"Stage1成功："
        f"{stage1_success}/40"
    )

    if stage1_success != 40:

        raise RuntimeError(
            "Stage1没有完成40/40，"
            "暂停Stage2。"
        )

    # =====================================================
    # 只选consistent进入Stage2
    # =====================================================

    support_consistent = []

    anti_consistent = []

    for row in rows:

        sample_id = (
            row["Sample ID"]
        )

        stage1 = (
            checkpoint[
                "stage1"
            ][
                sample_id
            ]
        )

        if (
            stage1[
                "stance_consistency"
            ]
            != "consistent"
        ):

            continue

        if (
            row["Trump Stance"]
            == "支持"
        ):

            support_consistent.append(
                row
            )

        else:

            anti_consistent.append(
                row
            )

    # =====================================================
    # STAGE 2
    # =====================================================

    print()

    print("=" * 72)

    print(
        "STAGE 2"
    )

    print(
        "只对consistent样本判断原因"
    )

    print("=" * 72)

    print()

    print(
        f"支持需要原因分类："
        f"{len(support_consistent)}"
    )

    print(
        f"反对需要原因分类："
        f"{len(anti_consistent)}"
    )

    process_with_fallback(
        support_consistent,
        "support",
        2,
        checkpoint
    )

    process_with_fallback(
        anti_consistent,
        "anti",
        2,
        checkpoint
    )

    # =====================================================
    # 保存
    # =====================================================

    save_final_output(
        rows,
        checkpoint
    )

    summary = (
        create_summary(
            rows,
            checkpoint,
            support_pool_n,
            anti_pool_n,
            excluded_n
        )
    )

    print()

    print(summary)

    print()

    print("=" * 72)

    print(
        "V5两阶段验证完成"
    )

    print("=" * 72)

    print()

    print(
        f"原始40条：\n"
        f"{SAMPLE_FILE}"
    )

    print()

    print(
        f"人工审核文件：\n"
        f"{OUTPUT_FILE}"
    )

    print()

    print(
        f"统计摘要：\n"
        f"{SUMMARY_FILE}"
    )

    print()

    print(
        f"Checkpoint：\n"
        f"{CHECKPOINT_FILE}"
    )

    print()

    print(
        "下一步："
        "最后人工审核这40条。"
    )

    print(
        "重点检查Stage1是否解决"
        "V4过度uncertain的问题。"
    )


if __name__ == "__main__":

    main()