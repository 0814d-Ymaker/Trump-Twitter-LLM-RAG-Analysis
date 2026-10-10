import csv
import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from collections import Counter


# =========================================================
# 基本设置
# =========================================================

BASE_DIR = Path(__file__).parent

INPUT_FILE = BASE_DIR / "Trump_5352_reasons_v5_final.csv"

INDEX_FILE = BASE_DIR / "Trump_rag_index_v1.jsonl"

SUMMARY_FILE = BASE_DIR / "Trump_rag_index_v1_summary.txt"


API_URL = "http://127.0.0.1:1234/v1/embeddings"

EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"

BATCH_SIZE = 16

MAX_RETRIES = 3

EXPECTED_DIMENSION = 768


# =========================================================
# 自动寻找CSV列名
# =========================================================

def find_column(fieldnames, candidates, required=True):

    for candidate in candidates:
        if candidate in fieldnames:
            return candidate

    if required:
        raise ValueError(
            f"找不到需要的列。\n"
            f"尝试寻找：{candidates}\n"
            f"实际CSV列：{fieldnames}"
        )

    return None


# =========================================================
# 调用LM Studio Embedding API
# =========================================================

def request_embeddings(texts):

    # Nomic官方推荐：
    # 文档使用 search_document:
    prefixed_texts = [
        "search_document: " + text
        for text in texts
    ]

    body = {
        "model": EMBEDDING_MODEL,
        "input": prefixed_texts
    }

    data = json.dumps(body).encode("utf-8")

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:

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
                timeout=120
            ) as response:

                result = json.loads(
                    response.read().decode("utf-8")
                )

            embedding_data = result.get("data", [])

            embedding_data = sorted(
                embedding_data,
                key=lambda x: x.get("index", 0)
            )

            embeddings = [
                item["embedding"]
                for item in embedding_data
            ]

            if len(embeddings) != len(texts):
                raise ValueError(
                    f"Embedding数量不一致："
                    f"输入{len(texts)}条，"
                    f"返回{len(embeddings)}条"
                )

            for embedding in embeddings:

                if len(embedding) != EXPECTED_DIMENSION:
                    raise ValueError(
                        f"向量维度异常："
                        f"预期{EXPECTED_DIMENSION}，"
                        f"实际{len(embedding)}"
                    )

            return embeddings

        except Exception as e:

            last_error = e

            print(
                f"Embedding请求失败 "
                f"({attempt}/{MAX_RETRIES})：{e}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(2 * attempt)

    raise RuntimeError(
        f"Embedding请求最终失败：{last_error}"
    )


# =========================================================
# 如果批量失败，尝试逐条补救
# =========================================================

def embed_batch_with_fallback(texts):

    try:
        return request_embeddings(texts)

    except Exception as batch_error:

        print()
        print(
            "批量Embedding失败，开始逐条补跑。"
        )

        print(
            "批量错误：",
            batch_error
        )

        results = []

        for i, text in enumerate(texts, start=1):

            print(
                f"  单条补跑 {i}/{len(texts)}"
            )

            embedding = request_embeddings(
                [text]
            )[0]

            results.append(
                embedding
            )

        return results


# =========================================================
# 读取已经完成的Row ID
# 实现断点续跑
# =========================================================

def load_existing_ids():

    existing_ids = set()

    if not INDEX_FILE.exists():
        return existing_ids

    with INDEX_FILE.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1
        ):

            line = line.strip()

            if not line:
                continue

            try:

                item = json.loads(line)

                row_id = str(
                    item.get("row_id", "")
                ).strip()

                if row_id:
                    existing_ids.add(
                        row_id
                    )

            except json.JSONDecodeError:

                print(
                    f"警告：索引文件第{line_number}行"
                    f"无法解析，将忽略。"
                )

    return existing_ids


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 72)

    print(
        "Trump RAG Index Builder V1"
    )

    print("=" * 72)

    print()

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"找不到输入文件：\n{INPUT_FILE}"
        )

    # -----------------------------------------------------
    # 读取CSV
    # -----------------------------------------------------

    with INPUT_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        fieldnames = reader.fieldnames

        rows = list(reader)

    print(
        f"读取最终原因分析数据：{len(rows)}条"
    )

    # -----------------------------------------------------
    # 自动识别需要的列
    # -----------------------------------------------------

    row_id_col = find_column(
        fieldnames,
        [
            "Row ID",
            "row_id",
            "RowID"
        ]
    )

    text_col = find_column(
        fieldnames,
        [
            "Tweet Content",
            "tweet_content",
            "Tweet content",
            "Content"
        ]
    )

    reason_col = find_column(
        fieldnames,
        [
            "Primary Reason Code",
            "primary_code",
            "Primary Reason"
        ]
    )

    secondary_col = find_column(
        fieldnames,
        [
            "Secondary Reason Code",
            "secondary_code"
        ],
        required=False
    )

    evidence_col = find_column(
        fieldnames,
        [
            "Reason Evidence",
            "evidence"
        ],
        required=False
    )

    confidence_col = find_column(
        fieldnames,
        [
            "Reason Confidence",
            "confidence"
        ],
        required=False
    )

    date_col = find_column(
        fieldnames,
        [
            "Tweet Date",
            "tweet_date"
        ],
        required=False
    )

    username_col = find_column(
        fieldnames,
        [
            "User Name",
            "user_name",
            "Username"
        ],
        required=False
    )

    url_col = find_column(
        fieldnames,
        [
            "Tweet URL",
            "tweet_url",
            "URL"
        ],
        required=False
    )

    # -----------------------------------------------------
    # 只保留有效原因编码
    # S0-S6 / A0-A6
    # 排除3条冲突和ERROR
    # -----------------------------------------------------

    valid_codes = {
        "S0", "S1", "S2",
        "S3", "S4", "S5", "S6",

        "A0", "A1", "A2",
        "A3", "A4", "A5", "A6"
    }

    valid_rows = []

    skipped_rows = []

    for row in rows:

        row_id = str(
            row.get(
                row_id_col,
                ""
            )
        ).strip()

        reason_code = str(
            row.get(
                reason_col,
                ""
            )
        ).strip()

        tweet_text = str(
            row.get(
                text_col,
                ""
            )
        ).strip()

        if (
            reason_code in valid_codes
            and tweet_text
        ):

            valid_rows.append(
                row
            )

        else:

            skipped_rows.append(
                (
                    row_id,
                    reason_code
                )
            )

    print(
        f"有效RAG文档：{len(valid_rows)}"
    )

    print(
        f"排除记录：{len(skipped_rows)}"
    )

    print()

    if len(valid_rows) != 5349:

        raise ValueError(
            f"预期5349条有效记录，"
            f"实际得到{len(valid_rows)}条。\n"
            f"请不要继续运行，先检查数据。"
        )

    # -----------------------------------------------------
    # 读取断点
    # -----------------------------------------------------

    existing_ids = load_existing_ids()

    print(
        f"已有Embedding记录：{len(existing_ids)}"
    )

    pending_rows = [
        row
        for row in valid_rows
        if str(
            row.get(
                row_id_col,
                ""
            )
        ).strip()
        not in existing_ids
    ]

    print(
        f"本次待处理：{len(pending_rows)}"
    )

    print()

    if not pending_rows:

        print(
            "全部5349条已经完成Embedding。"
        )

    else:

        total_batches = (
            len(pending_rows)
            + BATCH_SIZE
            - 1
        ) // BATCH_SIZE

        # -------------------------------------------------
        # 正式批量Embedding
        # -------------------------------------------------

        for batch_number, start in enumerate(
            range(
                0,
                len(pending_rows),
                BATCH_SIZE
            ),
            start=1
        ):

            batch_rows = pending_rows[
                start:
                start + BATCH_SIZE
            ]

            texts = [
                str(
                    row.get(
                        text_col,
                        ""
                    )
                ).strip()
                for row in batch_rows
            ]

            row_ids = [
                str(
                    row.get(
                        row_id_col,
                        ""
                    )
                ).strip()
                for row in batch_rows
            ]

            print(
                f"Batch "
                f"{batch_number}/{total_batches}"
                f"：{len(batch_rows)}条"
            )

            print(
                "Row ID：",
                row_ids[0],
                "→",
                row_ids[-1]
            )

            embeddings = embed_batch_with_fallback(
                texts
            )

            # ---------------------------------------------
            # 每批完成后立即写入
            # 即使中途停止，也不会丢掉已经完成的数据
            # ---------------------------------------------

            with INDEX_FILE.open(
                "a",
                encoding="utf-8"
            ) as f:

                for row, embedding in zip(
                    batch_rows,
                    embeddings
                ):

                    row_id = str(
                        row.get(
                            row_id_col,
                            ""
                        )
                    ).strip()

                    reason_code = str(
                        row.get(
                            reason_col,
                            ""
                        )
                    ).strip()

                    if reason_code.startswith("S"):
                        stance = "support"

                    elif reason_code.startswith("A"):
                        stance = "anti"

                    else:
                        stance = "unknown"

                    record = {
                        "row_id": row_id,

                        "stance": stance,

                        "primary_reason_code":
                            reason_code,

                        "secondary_reason_code":
                            (
                                str(
                                    row.get(
                                        secondary_col,
                                        ""
                                    )
                                ).strip()
                                if secondary_col
                                else ""
                            ),

                        "tweet_content":
                            str(
                                row.get(
                                    text_col,
                                    ""
                                )
                            ).strip(),

                        "tweet_date":
                            (
                                str(
                                    row.get(
                                        date_col,
                                        ""
                                    )
                                ).strip()
                                if date_col
                                else ""
                            ),

                        "user_name":
                            (
                                str(
                                    row.get(
                                        username_col,
                                        ""
                                    )
                                ).strip()
                                if username_col
                                else ""
                            ),

                        "tweet_url":
                            (
                                str(
                                    row.get(
                                        url_col,
                                        ""
                                    )
                                ).strip()
                                if url_col
                                else ""
                            ),

                        "reason_evidence":
                            (
                                str(
                                    row.get(
                                        evidence_col,
                                        ""
                                    )
                                ).strip()
                                if evidence_col
                                else ""
                            ),

                        "reason_confidence":
                            (
                                str(
                                    row.get(
                                        confidence_col,
                                        ""
                                    )
                                ).strip()
                                if confidence_col
                                else ""
                            ),

                        "embedding":
                            embedding
                    }

                    f.write(
                        json.dumps(
                            record,
                            ensure_ascii=False
                        )
                        + "\n"
                    )

            print(
                "本批保存完成。"
            )

            print()

    # -----------------------------------------------------
    # 最终检查索引
    # -----------------------------------------------------

    final_ids = set()

    code_counter = Counter()

    dimensions = set()

    valid_line_count = 0

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

            row_id = str(
                item.get(
                    "row_id",
                    ""
                )
            ).strip()

            code = str(
                item.get(
                    "primary_reason_code",
                    ""
                )
            ).strip()

            embedding = item.get(
                "embedding",
                []
            )

            if row_id:

                final_ids.add(
                    row_id
                )

                valid_line_count += 1

                code_counter[
                    code
                ] += 1

                dimensions.add(
                    len(embedding)
                )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    lines = []

    lines.append(
        "=" * 72
    )

    lines.append(
        "Trump RAG Index V1 Summary"
    )

    lines.append(
        "=" * 72
    )

    lines.append("")

    lines.append(
        f"输入记录：{len(rows)}"
    )

    lines.append(
        f"进入RAG索引：{len(final_ids)}"
    )

    lines.append(
        f"排除记录：{len(rows) - len(final_ids)}"
    )

    lines.append(
        f"Embedding模型：{EMBEDDING_MODEL}"
    )

    lines.append(
        f"向量维度：{sorted(dimensions)}"
    )

    lines.append("")

    lines.append(
        "原因代码分布："
    )

    for code in sorted(
        code_counter.keys()
    ):

        lines.append(
            f"{code}: {code_counter[code]}"
        )

    lines.append("")

    lines.append(
        f"索引文件：{INDEX_FILE}"
    )

    summary = "\n".join(
        lines
    )

    with SUMMARY_FILE.open(
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            summary
        )

    print("=" * 72)

    print(
        "RAG Index Build Result"
    )

    print("=" * 72)

    print(
        f"索引唯一Row ID：{len(final_ids)}"
    )

    print(
        f"有效JSONL记录：{valid_line_count}"
    )

    print(
        f"向量维度：{sorted(dimensions)}"
    )

    print()

    print(
        "索引文件："
    )

    print(
        INDEX_FILE
    )

    print()

    print(
        "统计文件："
    )

    print(
        SUMMARY_FILE
    )

    print()

    if (
        len(final_ids) == 5349
        and dimensions == {768}
    ):

        print(
            "RAG向量索引构建完成。"
        )

    else:

        print(
            "警告：索引尚未完整，请再次运行程序继续断点续跑。"
        )


if __name__ == "__main__":
    main()