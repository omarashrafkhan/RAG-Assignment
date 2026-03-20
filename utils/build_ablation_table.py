import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def load_eval(path: Path) -> Dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "summary" not in data:
        raise RuntimeError(f"Invalid evaluation format in {path}")
    return data


def run_label(summary: Dict) -> str:
    strategy = str(summary.get("strategy", "unknown"))
    bm25_only = bool(summary.get("bm25_only", False))
    bm25_top_k = int(summary.get("bm25_top_k", 30))
    semantic_top_k = int(summary.get("semantic_top_k", 30))
    used_reranker = bool(summary.get("used_reranker", False))

    if bm25_only:
        retrieval = "bm25_only"
    elif bm25_top_k <= 0 and semantic_top_k > 0:
        retrieval = "semantic_only"
    elif semantic_top_k <= 0 and bm25_top_k > 0:
        retrieval = "bm25_only"
    else:
        retrieval = "hybrid"

    rerank = "rerank" if used_reranker else "no_rerank"
    return f"{strategy}__{retrieval}__{rerank}"


def build_rows(files: List[Path]) -> List[Dict]:
    rows: List[Dict] = []
    for f in files:
        data = load_eval(f)
        s = data.get("summary", {})

        row = {
            "run_name": run_label(s),
            "strategy": str(s.get("strategy", "")),
            "retrieval_mode": (
                "BM25-only"
                if bool(s.get("bm25_only", False))
                else (
                    "Semantic-only"
                    if int(s.get("bm25_top_k", 30)) <= 0
                    and int(s.get("semantic_top_k", 30)) > 0
                    else (
                        "BM25-only"
                        if int(s.get("semantic_top_k", 30)) <= 0
                        and int(s.get("bm25_top_k", 30)) > 0
                        else "Hybrid (BM25+Semantic+RRF)"
                    )
                )
            ),
            "reranking": "Yes" if bool(s.get("used_reranker", False)) else "No",
            "n_queries": int(s.get("n_queries", 0)),
            "avg_faithfulness": float(s.get("avg_faithfulness", 0.0)),
            "avg_relevancy": float(s.get("avg_relevancy", 0.0)),
            "generation_model": str(s.get("generation_model", "")),
            "judge_model": str(s.get("judge_model", "")),
            "source_file": f.as_posix(),
            "note": "partial-test" if int(s.get("n_queries", 0)) < 10 else "full-test",
        }
        rows.append(row)

    rows.sort(key=lambda r: (r["strategy"], r["retrieval_mode"], r["reranking"]))
    return rows


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "strategy",
        "retrieval_mode",
        "reranking",
        "n_queries",
        "avg_faithfulness",
        "avg_relevancy",
        "generation_model",
        "judge_model",
        "source_file",
        "note",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_markdown(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "| Run | Chunking | Retrieval | Re-ranking | #Queries | Faithfulness | Relevancy | Note |\n"
        "|---|---|---|---|---:|---:|---:|---|\n"
    )
    lines = [header]
    for r in rows:
        lines.append(
            "| {run_name} | {strategy} | {retrieval_mode} | {reranking} | {n_queries} | {avg_faithfulness:.4f} | {avg_relevancy:.4f} | {note} |\n".format(
                **r
            )
        )
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ablation table (CSV + Markdown) from evaluation result JSON files."
    )
    parser.add_argument(
        "--input-glob",
        type=str,
        default="rag_artifacts/eval/eval*.json",
        help="Glob pattern for evaluation result files.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="rag_artifacts/eval/ablation_table.csv",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default="rag_artifacts/eval/ablation_table.md",
    )
    args = parser.parse_args()

    files = sorted(Path(".").glob(args.input_glob))
    if not files:
        raise RuntimeError(f"No files found for pattern: {args.input_glob}")

    rows = build_rows(files)
    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    write_csv(out_csv, rows)
    write_markdown(out_md, rows)

    print(f"Wrote {len(rows)} rows")
    print(f"CSV: {out_csv}")
    print(f"MD: {out_md}")


if __name__ == "__main__":
    main()
