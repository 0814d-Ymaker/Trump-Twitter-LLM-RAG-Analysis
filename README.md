# Trump Twitter LLM & RAG Analysis

## Project Overview

This project analyzes 10,000 X/Twitter posts related to Donald Trump using locally deployed language models.

The original tweet dataset was collected in my previous web scraping task. This project focuses on the next stage: stance classification, reason analysis, and building a RAG system that can retrieve source tweets and generate answers based on the retrieved evidence.

The overall workflow is:

```text
10,000 tweets
    ↓
Stance classification
    ↓
Support / Anti / Neutral
    ↓
Reason analysis
    ↓
Top supporting and opposing reasons
    ↓
Embedding + vector retrieval
    ↓
RAG question answering
```

---

## Models and Environment

The models were deployed locally using LM Studio.

Main models:

- `Qwen3.5-4B` for stance classification, reason analysis, and RAG answer generation
- `text-embedding-nomic-embed-text-v1.5` for text embeddings

The embedding dimension is 768.

Main tools:

- Python
- PyCharm
- LM Studio

---

## Stance Classification

The 10,000 tweets were classified into three groups:

| Stance | Count | Percentage |
|---|---:|---:|
| Support | 1,902 | 19.02% |
| Anti | 3,450 | 34.50% |
| Neutral / Unable to Determine | 4,648 | 46.48% |

The final classification script is:

```text
classify_10000_batch20_v3.py
```

Before running the full dataset, I tested several prompting and batch-processing methods and compared their results.

The final script also supports checkpointing, so an interrupted run can continue without restarting from the first tweet.

---

## Reason Analysis

After stance classification, the Support and Anti groups were analyzed separately.

The main Support reason categories include:

- leadership and personal trust
- anti-establishment attitudes and rejection of political opponents
- immigration, border and public safety
- economic and domestic policy
- conservative values and national identity
- foreign policy and national security

The main Anti reason categories include:

- personal conduct and morality
- legal and democratic-process concerns
- governance capability and information credibility
- economic and social policy
- foreign policy and national security
- immigration and group-rights concerns

The main reason-analysis scripts are:

```text
prepare_reason_analysis.py
discover_reason_taxonomy_v3.py
discover_reason_taxonomy_v4.py
validate_reason_taxonomy_holdout_v5.py
classify_all_5352_reasons_v5.py
finalize_reason_results_v5.py
```

`discover_reason_taxonomy_v4.py` uses functions from `discover_reason_taxonomy_v3.py`, so both files are included.

A separate holdout validation step was used before applying the final taxonomy to the full dataset.

---

## Final Reason Results

During the final reason-analysis run, three tweets originally classified as Support were found to be clear stance conflicts.

Instead of forcing them into a Support reason category, they were marked as:

```text
UPSTREAM_STANCE_CONFLICT
```

Final valid records:

- Support: 1,899
- Anti: 3,450
- Upstream stance conflicts: 3
- Unexplained errors: 0

### Top 3 reasons for supporting Trump

| Rank | Code | Reason | Count |
|---|---|---|---:|
| 1 | S1 | Personal leadership trust and emotional identification | 478 |
| 2 | S2 | Anti-establishment attitudes, judicial skepticism, and rejection of political opponents | 382 |
| 3 | S4 | Economic, social, and domestic policy performance | 199 |

### Top 3 reasons for opposing Trump

| Rank | Code | Reason | Count |
|---|---|---|---:|
| 1 | A3 | Governance capability, administrative performance, and information credibility | 644 |
| 2 | A1 | Personal conduct, morality, and allegations of illegal behavior | 589 |
| 3 | A2 | Rule of law, judicial issues, and democratic-process concerns | 286 |

`S0` and `A0` represent unclear or other reasons, so they are not included in the Top 3 ranking.

---

## RAG

After reason analysis, I built a RAG index using 5,349 valid Support and Anti tweets.

Each tweet was converted into a 768-dimensional embedding using `nomic-embed-text-v1.5`.

At first, I tested vector retrieval directly across all 5,349 tweets. One issue was that semantically similar tweets from the opposite stance could still appear in the results.

For example, a query asking why people support Trump could still retrieve Anti tweets discussing Trump's personality or leadership.

To reduce this problem, I changed the retrieval process to:

```text
Metadata filtering
    ↓
Vector similarity retrieval
    ↓
Qwen answer generation
```

For example, a question about leadership-based support is first restricted to:

```text
stance = support
primary_reason_code = S1
```

Vector retrieval is then performed only within that filtered group.

This made the retrieved evidence more relevant to the actual question.

The final RAG scripts are:

```text
build_rag_index.py
rag_qa.py
```

The final RAG system also checks whether the Row IDs cited by Qwen actually exist in the retrieved evidence.

---

## Example RAG Query

Example question:

```text
Why do people support Trump because of his leadership and personality?
```

Retrieval scope:

```text
Support / S1
```

Candidate documents:

```text
478
```

Retrieved documents:

```text
Top 8
```

The final answer was successfully generated using retrieved tweet evidence and valid original Row ID references.

A saved example is included in:

```text
Trump_rag_last_answer.txt
```

---

## Main Files

```text
Trump_Twitter_LLM_RAG_Project/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── classify_10000_batch20_v3.py
├── prepare_reason_analysis.py
├── discover_reason_taxonomy_v3.py
├── discover_reason_taxonomy_v4.py
├── validate_reason_taxonomy_holdout_v5.py
├── classify_all_5352_reasons_v5.py
├── finalize_reason_results_v5.py
├── build_rag_index.py
├── rag_qa.py
│
├── Trump_stance_statistics.txt
├── Trump_5352_reasons_v5_final_summary.txt
├── Trump_rag_index_v1_summary.txt
└── Trump_rag_last_answer.txt
```

This repository contains the main analysis scripts, final summary files, and a representative RAG output.

Intermediate experiments, checkpoints, logs, raw model responses, and large datasets are kept locally.

---

## Data Availability

The full 10,000-tweet dataset and the complete 5,349-document vector index are not included in this repository.

They are excluded because they are large intermediate data files and are not necessary for presenting the main analysis workflow and final results.

---

## Notes and Limitations

This project uses a relatively small locally deployed language model, so classification errors are still possible.

Some tweets contain sarcasm, jokes, incomplete context, or ambiguous language, which can affect both stance and reason classification.

The RAG system also depends on the quality of the upstream stance and reason labels.

For this reason, retrieved tweets should be treated as user-generated source evidence rather than verified facts.

The final results should therefore be interpreted as patterns detected in the analyzed dataset rather than as definitive measurements of public opinion.