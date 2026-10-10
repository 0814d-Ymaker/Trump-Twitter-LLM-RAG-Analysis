import csv
from pathlib import Path


# =========================================================
# 文件路径
# =========================================================

BASE_DIR = Path(__file__).parent

INPUT_FILE = BASE_DIR / "Trump_10000_classified_v3.csv"

SUPPORT_FILE = BASE_DIR / "Trump_support_tweets.csv"

ANTI_FILE = BASE_DIR / "Trump_anti_tweets.csv"

NEUTRAL_FILE = BASE_DIR / "Trump_neutral_tweets.csv"

STATS_FILE = BASE_DIR / "Trump_stance_statistics.txt"


# =========================================================
# 三种合法标签
# =========================================================

SUPPORT_LABEL = "支持"
ANTI_LABEL = "反对"
NEUTRAL_LABEL = "中立或无法判断"


# =========================================================
# 主程序
# =========================================================

def main():

    # -----------------------------------------------------
    # 检查输入文件是否存在
    # -----------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"找不到最终分类文件：\n{INPUT_FILE}"
        )


    # -----------------------------------------------------
    # 读取最终分类结果
    # -----------------------------------------------------

    with INPUT_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        if not reader.fieldnames:

            raise ValueError(
                "CSV 文件没有检测到表头。"
            )


        fieldnames = list(
            reader.fieldnames
        )


        print(
            "检测到字段："
        )

        print(
            fieldnames
        )

        print()


        # -------------------------------------------------
        # 检查 Trump Stance 列
        # -------------------------------------------------

        if "Trump Stance" not in fieldnames:

            raise ValueError(
                "CSV 中没有找到 'Trump Stance' 列。\n"
                f"当前字段为：{fieldnames}"
            )


        rows = list(reader)


    total_count = len(rows)


    print(
        f"成功读取：{total_count} 条推文"
    )

    print()


    # =====================================================
    # 按立场拆分
    # =====================================================

    support_rows = []

    anti_rows = []

    neutral_rows = []

    unknown_rows = []


    for row in rows:

        stance = (
            row.get(
                "Trump Stance",
                ""
            )
            or ""
        ).strip()


        if stance == SUPPORT_LABEL:

            support_rows.append(
                row
            )


        elif stance == ANTI_LABEL:

            anti_rows.append(
                row
            )


        elif stance == NEUTRAL_LABEL:

            neutral_rows.append(
                row
            )


        else:

            unknown_rows.append(
                row
            )


    # =====================================================
    # 保存函数
    # =====================================================

    def save_csv(
        file_path,
        data_rows
    ):

        with file_path.open(
            "w",
            encoding="utf-8-sig",
            newline=""
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()

            writer.writerows(
                data_rows
            )


    # =====================================================
    # 保存三个 CSV
    # =====================================================

    save_csv(
        SUPPORT_FILE,
        support_rows
    )

    save_csv(
        ANTI_FILE,
        anti_rows
    )

    save_csv(
        NEUTRAL_FILE,
        neutral_rows
    )


    # =====================================================
    # 数量统计
    # =====================================================

    support_count = len(
        support_rows
    )

    anti_count = len(
        anti_rows
    )

    neutral_count = len(
        neutral_rows
    )

    unknown_count = len(
        unknown_rows
    )


    support_percent = (
        support_count
        / total_count
        * 100
    )

    anti_percent = (
        anti_count
        / total_count
        * 100
    )

    neutral_percent = (
        neutral_count
        / total_count
        * 100
    )


    # =====================================================
    # 输出控制台结果
    # =====================================================

    print("=" * 70)

    print(
        "Trump 10,000 条推文立场拆分结果"
    )

    print("=" * 70)

    print()

    print(
        f"总推文数："
        f"{total_count}"
    )

    print()

    print(
        f"支持："
        f"{support_count} 条 "
        f"({support_percent:.2f}%)"
    )

    print(
        f"反对："
        f"{anti_count} 条 "
        f"({anti_percent:.2f}%)"
    )

    print(
        f"中立或无法判断："
        f"{neutral_count} 条 "
        f"({neutral_percent:.2f}%)"
    )

    print()


    # -----------------------------------------------------
    # 理论上这里应该为 0
    # -----------------------------------------------------

    if unknown_count > 0:

        print(
            f"警告：发现 "
            f"{unknown_count} 条未知标签。"
        )

        print()

        print(
            "这些数据没有被写入三个正式分类文件。"
        )

        print()


    # =====================================================
    # 核对数量
    # =====================================================

    valid_total = (
        support_count
        + anti_count
        + neutral_count
    )


    print(
        f"三类合计："
        f"{valid_total}"
    )


    if valid_total == total_count:

        print(
            "数量核对成功："
            "三类数据总和等于原始数据总数。"
        )

    else:

        print(
            "数量核对失败！"
        )

        print(
            f"原始总数：{total_count}"
        )

        print(
            f"三类总数：{valid_total}"
        )

        print(
            f"未知标签：{unknown_count}"
        )


    # =====================================================
    # 保存统计 TXT
    # =====================================================

    with STATS_FILE.open(
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "Trump 10,000 Tweets "
            "Stance Statistics\n"
        )

        f.write(
            "=" * 60
            + "\n"
        )

        f.write(
            f"Total: {total_count}\n"
        )

        f.write(
            f"Support: "
            f"{support_count} "
            f"({support_percent:.2f}%)\n"
        )

        f.write(
            f"Anti: "
            f"{anti_count} "
            f"({anti_percent:.2f}%)\n"
        )

        f.write(
            f"Neutral/Uncertain: "
            f"{neutral_count} "
            f"({neutral_percent:.2f}%)\n"
        )

        f.write(
            f"Unknown labels: "
            f"{unknown_count}\n"
        )


    # =====================================================
    # 输出文件位置
    # =====================================================

    print()

    print("=" * 70)

    print(
        "生成文件："
    )

    print("=" * 70)

    print(
        f"支持推文：\n"
        f"{SUPPORT_FILE}"
    )

    print()

    print(
        f"反对推文：\n"
        f"{ANTI_FILE}"
    )

    print()

    print(
        f"中立推文：\n"
        f"{NEUTRAL_FILE}"
    )

    print()

    print(
        f"统计文件：\n"
        f"{STATS_FILE}"
    )

    print()

    print(
        "第一阶段数据准备完成。"
    )


# =========================================================
# 启动程序
# =========================================================

if __name__ == "__main__":

    main()