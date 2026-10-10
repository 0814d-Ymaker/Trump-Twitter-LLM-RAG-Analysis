import csv
from pathlib import Path
from collections import Counter


BASE_DIR = Path(__file__).parent

# 原来的正式运行结果
INPUT_ALL = BASE_DIR / "Trump_5352_reasons_v5.csv"

# 新的最终版文件
OUTPUT_FINAL = BASE_DIR / "Trump_5352_reasons_v5_final.csv"
SUMMARY_FINAL = BASE_DIR / "Trump_5352_reasons_v5_final_summary.txt"


# =========================================================
# 已人工确认的3条上游立场冲突
# =========================================================

CONFLICT_ROWS = {
    "735": {
        "evidence": "Donald Trump should be indicted.",
        "note": (
            "Upstream stance conflict: original stance classifier "
            "assigned support, but the tweet explicitly calls for "
            "Trump to be indicted. Reason model repeatedly returned "
            "an anti-Trump category (A2)."
        )
    },

    "6249": {
        "evidence": (
            "Trump is nothing more than the biggest liar "
            "in any room he's in!"
        ),
        "note": (
            "Upstream stance conflict: original stance classifier "
            "assigned support, but the tweet explicitly criticizes "
            "Trump as a liar. Reason model repeatedly returned "
            "an anti-Trump category (A3)."
        )
    },

    "9289": {
        "evidence": (
            "Trump should finally be impeached. "
            "The evidence is substantial."
        ),
        "note": (
            "Upstream stance conflict: original stance classifier "
            "assigned support, but the tweet explicitly calls for "
            "Trump to be impeached. Repeated reason-model runs "
            "consistently treated it as anti-Trump, although the "
            "specific anti-reason code varied between A1 and A3."
        )
    }
}


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
# 读取CSV
# =========================================================

with INPUT_ALL.open(
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:

    reader = csv.DictReader(f)

    fieldnames = reader.fieldnames
    rows = list(reader)


if len(rows) != 5352:
    raise ValueError(
        f"预期5352条，实际读取{len(rows)}条。"
    )


# =========================================================
# 修改3条冲突记录
# =========================================================

found_conflicts = []

for row in rows:

    row_id = str(
        row.get("Row ID", "")
    ).strip()

    if row_id in CONFLICT_ROWS:

        info = CONFLICT_ROWS[row_id]

        row[
            "Reason Analysis Stance"
        ] = "上游立场冲突"

        row[
            "Primary Reason Code"
        ] = "UPSTREAM_STANCE_CONFLICT"

        row[
            "Secondary Reason Code"
        ] = ""

        row[
            "Reason Evidence"
        ] = info["evidence"]

        row[
            "Reason Confidence"
        ] = "high"

        row[
            "Reason Model Note"
        ] = info["note"]

        found_conflicts.append(
            row_id
        )


# =========================================================
# 检查3条是否都找到
# =========================================================

expected = set(
    CONFLICT_ROWS.keys()
)

found = set(
    found_conflicts
)

if found != expected:

    raise ValueError(
        f"冲突Row ID检查失败。\n"
        f"应该找到：{sorted(expected)}\n"
        f"实际找到：{sorted(found)}"
    )


# =========================================================
# 保存最终CSV
# =========================================================

with OUTPUT_FINAL.open(
    "w",
    encoding="utf-8-sig",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(rows)


# =========================================================
# 最终统计
# =========================================================

support_counter = Counter()
anti_counter = Counter()

support_confidence = Counter()
anti_confidence = Counter()

conflict_count = 0
error_count = 0


for row in rows:

    stance = str(
        row.get(
            "Reason Analysis Stance",
            ""
        )
    ).strip()

    code = str(
        row.get(
            "Primary Reason Code",
            ""
        )
    ).strip()

    confidence = str(
        row.get(
            "Reason Confidence",
            ""
        )
    ).strip().lower()

    if code == "UPSTREAM_STANCE_CONFLICT":

        conflict_count += 1
        continue

    if code == "ERROR":

        error_count += 1
        continue

    if code.startswith("S"):

        support_counter[
            code
        ] += 1

        if confidence:
            support_confidence[
                confidence
            ] += 1

    elif code.startswith("A"):

        anti_counter[
            code
        ] += 1

        if confidence:
            anti_confidence[
                confidence
            ] += 1


support_valid = sum(
    support_counter.values()
)

anti_valid = sum(
    anti_counter.values()
)


# =========================================================
# Top 3
# =========================================================

support_explicit = [
    (code, support_counter[code])
    for code in [
        "S1", "S2", "S3",
        "S4", "S5", "S6"
    ]
]

support_explicit.sort(
    key=lambda x: x[1],
    reverse=True
)


anti_explicit = [
    (code, anti_counter[code])
    for code in [
        "A1", "A2", "A3",
        "A4", "A5", "A6"
    ]
]

anti_explicit.sort(
    key=lambda x: x[1],
    reverse=True
)


# =========================================================
# 创建最终Summary
# =========================================================

lines = []

lines.append(
    "=" * 72
)

lines.append(
    "Trump 5352 Tweets - Final V5 Reason Analysis"
)

lines.append(
    "=" * 72
)

lines.append("")

lines.append(
    "原始原因分析输入：5352"
)

lines.append(
    "原始Support组：1902"
)

lines.append(
    "原始Anti组：3450"
)

lines.append("")

lines.append(
    f"上游立场冲突：{conflict_count}"
)

lines.append(
    f"最终ERROR：{error_count}"
)

lines.append("")

lines.append(
    f"有效Support原因记录：{support_valid}"
)

lines.append(
    f"有效Anti原因记录：{anti_valid}"
)

lines.append("")


# =========================================================
# 支持原因
# =========================================================

lines.append(
    "=" * 72
)

lines.append(
    "支持Trump原因分布"
)

lines.append(
    f"分母：{support_valid}条有效支持原因记录"
)

lines.append(
    "=" * 72
)


for code in [
    "S0", "S1", "S2",
    "S3", "S4", "S5", "S6"
]:

    count = support_counter[
        code
    ]

    percent = (
        count
        / support_valid
        * 100
    )

    lines.append(
        f"{code} "
        f"{CODE_NAMES[code]}："
        f"{count} "
        f"({percent:.2f}%)"
    )


# =========================================================
# 反对原因
# =========================================================

lines.append("")
lines.append(
    "=" * 72
)

lines.append(
    "反对Trump原因分布"
)

lines.append(
    f"分母：{anti_valid}条有效反对原因记录"
)

lines.append(
    "=" * 72
)


for code in [
    "A0", "A1", "A2",
    "A3", "A4", "A5", "A6"
]:

    count = anti_counter[
        code
    ]

    percent = (
        count
        / anti_valid
        * 100
    )

    lines.append(
        f"{code} "
        f"{CODE_NAMES[code]}："
        f"{count} "
        f"({percent:.2f}%)"
    )


# =========================================================
# Top 3
# =========================================================

lines.append("")
lines.append(
    "=" * 72
)

lines.append(
    "支持Trump Top 3明确原因"
)

lines.append(
    "S0不参与Top 3排名"
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
    "A0不参与Top 3排名"
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


# =========================================================
# Conflict记录
# =========================================================

lines.append("")
lines.append(
    "=" * 72
)

lines.append(
    "Upstream Stance Conflicts"
)

lines.append(
    "=" * 72
)

for row_id in [
    "735",
    "6249",
    "9289"
]:

    lines.append(
        f"Row ID {row_id}: "
        f"UPSTREAM_STANCE_CONFLICT"
    )


lines.append("")

lines.append(
    "说明：这3条原始上游stance分类为Support，"
)

lines.append(
    "但文本内容及原因模型输出均明确表现为Anti-Trump。"
)

lines.append(
    "因此不强制编码为S0-S6，而从支持原因频率统计中排除。"
)


# =========================================================
# 保存Summary
# =========================================================

summary = "\n".join(
    lines
)

with SUMMARY_FINAL.open(
    "w",
    encoding="utf-8"
) as f:

    f.write(
        summary
    )


# =========================================================
# 控制台结果
# =========================================================

print(
    "=" * 72
)

print(
    "Final V5 Reason Results"
)

print(
    "=" * 72
)

print()

print(
    f"总记录：{len(rows)}"
)

print(
    f"上游立场冲突：{conflict_count}"
)

print(
    f"最终ERROR：{error_count}"
)

print()

print(
    f"有效Support原因：{support_valid}"
)

print(
    f"有效Anti原因：{anti_valid}"
)

print()

print(
    "冲突Row ID："
)

print(
    sorted(
        found_conflicts,
        key=int
    )
)

print()

print(
    "最终结果文件："
)

print(
    OUTPUT_FINAL
)

print()

print(
    "最终统计文件："
)

print(
    SUMMARY_FINAL
)

print()

if error_count == 0:

    print(
        "收尾完成：不存在未解释ERROR。"
    )

else:

    print(
        "警告：仍然存在ERROR，需要继续检查。"
    )