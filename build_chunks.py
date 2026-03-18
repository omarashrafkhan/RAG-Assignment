import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


def clean_text(text: str) -> str:
    text = text.replace("\ufeff", " ")
    text = text.replace("\u200c", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text, flags=re.UNICODE))


def split_words(text: str) -> List[str]:
    return re.findall(r"\S+", text, flags=re.UNICODE)


def split_paragraphs(text: str) -> List[str]:
    raw = re.split(r"\n\s*\n", text)
    parts = [clean_text(p) for p in raw if clean_text(p)]
    if parts:
        return parts
    return [clean_text(text)] if clean_text(text) else []


def split_sentences_urdu(text: str) -> List[str]:
    text = clean_text(text)
    if not text:
        return []

    # Keep Urdu and standard punctuation as sentence boundaries.
    parts = re.split(r"(?<=[۔.!؟?])\s+", text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts


def parse_document(file_path: Path) -> Dict[str, str]:
    raw = file_path.read_text(encoding="utf-8", errors="ignore")
    lines = [line.strip() for line in raw.splitlines() if line.strip()]

    title = file_path.stem
    category = "unknown"
    body = raw

    # Expected format from convert.py: title, category, then text.
    if len(lines) >= 3 and len(lines[0]) < 200 and len(lines[1]) < 120:
        title = lines[0]
        category = lines[1]
        body = "\n".join(lines[2:])

    return {
        "doc_id": file_path.stem,
        "source_file": file_path.name,
        "title": title,
        "category": category,
        "text": body,
    }


def chunk_fixed_words(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    words = split_words(clean_text(text))
    if not words:
        return []

    chunks: List[str] = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        segment = words[i : i + chunk_size]
        if not segment:
            continue
        chunks.append(" ".join(segment))
        if i + chunk_size >= len(words):
            break
    return chunks


def chunk_paragraph_recursive(
    text: str, max_words: int = 320, overlap_words: int = 40
) -> List[str]:
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return []

    chunks: List[str] = []
    current_words: List[str] = []

    for para in paragraphs:
        para_words = split_words(para)

        if len(para_words) > max_words:
            if current_words:
                chunks.append(" ".join(current_words))
                current_words = (
                    current_words[-overlap_words:] if overlap_words > 0 else []
                )

            # Fallback to fixed splitting for very long paragraphs.
            chunks.extend(
                chunk_fixed_words(para, chunk_size=max_words, overlap=overlap_words)
            )
            continue

        if len(current_words) + len(para_words) <= max_words:
            current_words.extend(para_words)
        else:
            if current_words:
                chunks.append(" ".join(current_words))
            current_words = current_words[-overlap_words:] if overlap_words > 0 else []
            current_words.extend(para_words)

    if current_words:
        chunks.append(" ".join(current_words))

    return [c for c in chunks if c.strip()]


def chunk_sentence_window(
    text: str,
    sentences_per_chunk: int = 5,
    sentence_overlap: int = 2,
    max_words: int = 260,
) -> List[str]:
    sentences = split_sentences_urdu(text)
    if not sentences:
        return []

    chunks: List[str] = []
    step = max(1, sentences_per_chunk - sentence_overlap)

    for i in range(0, len(sentences), step):
        block = sentences[i : i + sentences_per_chunk]
        if not block:
            continue
        candidate = " ".join(block)
        if word_count(candidate) <= max_words:
            chunks.append(candidate)
        else:
            chunks.extend(
                chunk_fixed_words(candidate, chunk_size=max_words, overlap=40)
            )
        if i + sentences_per_chunk >= len(sentences):
            break

    return chunks


def build_chunks_for_strategy(
    documents: List[Dict[str, str]], strategy: str
) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []

    for doc in documents:
        text = doc["text"]
        if strategy == "fixed":
            chunks = chunk_fixed_words(text, chunk_size=300, overlap=50)
        elif strategy == "recursive":
            chunks = chunk_paragraph_recursive(text, max_words=320, overlap_words=40)
        elif strategy == "sentence":
            chunks = chunk_sentence_window(
                text, sentences_per_chunk=5, sentence_overlap=2, max_words=260
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        for idx, chunk_text in enumerate(chunks, start=1):
            output.append(
                {
                    "chunk_id": f"{doc['doc_id']}_{strategy}_{idx:04d}",
                    "doc_id": doc["doc_id"],
                    "source_file": doc["source_file"],
                    "title": doc["title"],
                    "category": doc["category"],
                    "strategy": strategy,
                    "chunk_index": idx,
                    "chunk_word_count": word_count(chunk_text),
                    "text": chunk_text,
                }
            )

    return output


def write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(rows: List[Dict[str, object]], strategy: str) -> Dict[str, object]:
    if not rows:
        return {
            "strategy": strategy,
            "total_chunks": 0,
            "avg_chunk_words": 0,
            "min_chunk_words": 0,
            "max_chunk_words": 0,
        }

    sizes = [int(r["chunk_word_count"]) for r in rows]
    return {
        "strategy": strategy,
        "total_chunks": len(rows),
        "avg_chunk_words": round(sum(sizes) / len(sizes), 2),
        "min_chunk_words": min(sizes),
        "max_chunk_words": max(sizes),
    }


def load_documents(corpus_dir: Path) -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []
    for file_path in sorted(corpus_dir.glob("*.txt")):
        docs.append(parse_document(file_path))
    return docs


def run(corpus_dir: Path, out_dir: Path) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = load_documents(corpus_dir)
    if not docs:
        raise RuntimeError(f"No .txt files found in: {corpus_dir}")

    summaries: List[Dict[str, object]] = []

    for strategy in ["fixed", "recursive", "sentence"]:
        rows = build_chunks_for_strategy(docs, strategy)
        out_file = out_dir / f"chunks_{strategy}.jsonl"
        write_jsonl(out_file, rows)
        summaries.append(summarize(rows, strategy))
        print(f"[{strategy}] chunks: {len(rows)} -> {out_file}")

    summary_file = out_dir / "chunking_summary.json"
    summary_file.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Summary -> {summary_file}")

    docs_file = out_dir / "docs_manifest.json"
    docs_meta = [
        {
            "doc_id": d["doc_id"],
            "source_file": d["source_file"],
            "title": d["title"],
            "category": d["category"],
            "doc_word_count": word_count(d["text"]),
        }
        for d in docs
    ]
    docs_file.write_text(
        json.dumps(docs_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Manifest -> {docs_file}")

    return summary_file, docs_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build 3 chunking variants for Urdu RAG corpus."
    )
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default="urdu_health_corpus",
        help="Path to folder containing .txt documents",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="rag_artifacts/chunks",
        help="Output directory for chunk files",
    )
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    out_dir = Path(args.out_dir)
    run(corpus_dir, out_dir)


if __name__ == "__main__":
    main()
