import json, os, sys, time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

QWEN_API_KEY  = os.environ.get("QWEN_API_KEY", "NotRequired")
QWEN_ENDPOINT = os.environ.get("QWEN_ENDPOINT",
                    "http://localhost:11434/v1/chat/completions")
QWEN_MODEL    = os.environ.get("QWEN_MODEL", "qwen2.5:72b")

N_RUNS     = 3 
N_TEST_DOCS = 5
BIORED_TRAIN = Path("BioRED/Train.BioC.JSON")
OUT_DIR      = Path("alchemist_output")
REPORT_PATH  = OUT_DIR / "uncertainty_report.txt"

BIORED_ENTITY_TYPES = [
    "GeneOrGeneProduct", "DiseaseOrPhenotypicFeature", "ChemicalEntity",
    "SequenceVariant", "CellLine", "OrganismTaxon",
]

def _call(messages: list, temperature: float = 0.1) -> str | None:
    import requests
    try:
        r = requests.post(QWEN_ENDPOINT, headers={
            "Authorization": f"Bearer {QWEN_API_KEY}",
            "Content-Type": "application/json",
        }, json={
            "model": QWEN_MODEL,
            "messages": messages,
            "max_tokens": 2000,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }, timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  [API ERROR] {e}")
        return None


SYSTEM_NER = """\
You are a biomedical NER system. Extract named entities and return JSON only.
Types: GeneOrGeneProduct, DiseaseOrPhenotypicFeature, ChemicalEntity,
SequenceVariant, CellLine, OrganismTaxon.
"""


def extract_entities_one_run(doc_id: str, text: str, temperature: float = 0.1) -> list:
    raw = _call([
        {"role": "system", "content": SYSTEM_NER},
        {"role": "user",   "content": f"DOCUMENT ID: {doc_id}\nABSTRACT:\n{text[:2000]}\nJSON only."}
    ], temperature=temperature)
    if not raw:
        return []
    try:
        import re
        raw = re.sub(r"^```[a-z]*\n?", "", raw.strip(), flags=re.MULTILINE)
        raw = re.sub(r"\n?```$",        "", raw.strip(), flags=re.MULTILINE)
        data = json.loads(raw)
        return [(e.get("text","").lower().strip(), e.get("entity_type",""))
                for e in data.get("entities", [])]
    except Exception:
        return []


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def load_docs(path: Path, n: int) -> list:
    with open(path) as f:
        raw = json.load(f)
    collection = raw if isinstance(raw, list) else raw.get("documents", [])
    docs = []
    for doc in collection[:n * 3]:   
        full_text = ""
        for p in doc.get("passages", []):
            full_text += " " + p.get("text", "")
        if len(full_text.strip()) > 100:
            docs.append({"id": doc.get("id",""), "text": full_text.strip()})
        if len(docs) >= n:
            break
    return docs


def run_uncertainty_experiment():
    OUT_DIR.mkdir(exist_ok=True)

    if not BIORED_TRAIN.exists():
        print(f"ERROR: {BIORED_TRAIN} not found")
        return

    print(f"Loading {N_TEST_DOCS} test documents...")
    docs = load_docs(BIORED_TRAIN, N_TEST_DOCS)
    print(f"  {len(docs)} documents loaded for consistency test.")

    print(f"\nRunning {N_RUNS} inference passes per document...")
    print(f"Temperature: 0.1 (near-deterministic)")
    print(f"Model: {QWEN_MODEL}\n")

    all_results = {}   

    for doc in docs:
        print(f"  Doc {doc['id']}:")
        runs = []
        for run_i in range(N_RUNS):
            entities = extract_entities_one_run(doc["id"], doc["text"])
            entity_set = set(entities)
            runs.append(entity_set)
            print(f"    Run {run_i+1}: {len(entity_set)} entities")
            time.sleep(1)  

        all_results[doc["id"]] = runs

    print("CONSISTENCY RESULTS (Jaccard similarity between runs)")

    doc_similarities = []
    lines = []
    lines.append("UNCERTAINTY & CONSISTENCY ANALYSIS")
    lines.append(f"Model:       {QWEN_MODEL}")
    lines.append(f"Temperature: 0.1")
    lines.append(f"Runs/doc:    {N_RUNS}")
    lines.append(f"Test docs:   {N_TEST_DOCS}")
    lines.append("Method: Run same document 3 times, measure Jaccard similarity")
    lines.append("between entity sets. Jaccard=1.0 = perfectly consistent.")

    for doc_id, runs in all_results.items():
        pairs = [(i, j) for i in range(N_RUNS) for j in range(i+1, N_RUNS)]
        sims = [jaccard(runs[i], runs[j]) for i,j in pairs]
        mean_sim = sum(sims) / len(sims)
        doc_similarities.append(mean_sim)

        lines.append(f"Doc {doc_id}:")
        for (i, j), sim in zip(pairs, sims):
            lines.append(f"  Run {i+1} vs Run {j+1}: Jaccard = {sim:.3f}")

        stable = runs[0]
        for r in runs[1:]:
            stable = stable & r
        unstable = set()
        for r in runs:
            unstable |= r
        unstable -= stable

        lines.append(f"  Stable entities (all runs agree):   {len(stable)}")
        lines.append(f"  Unstable entities (varies by run):  {len(unstable)}")
        lines.append(f"  Mean Jaccard similarity:            {mean_sim:.3f}")
        lines.append("")

        print(f"  {doc_id}: mean Jaccard = {mean_sim:.3f}  "
              f"({len(stable)} stable, {len(unstable)} unstable entities)")

    overall_mean = sum(doc_similarities) / len(doc_similarities) if doc_similarities else 0
    overall_min  = min(doc_similarities) if doc_similarities else 0
    overall_max  = max(doc_similarities) if doc_similarities else 0

    summary = [
        "SUMMARY",
        f"Mean Jaccard similarity across all docs: {overall_mean:.3f}",
        f"Min: {overall_min:.3f}  Max: {overall_max:.3f}",
        "",
        "INTERPRETATION:",
    ]

    if overall_mean >= 0.85:
        summary.append("  HIGH consistency (≥0.85): The LLM produces stable, reproducible")
        summary.append("  outputs at temperature=0.1. Results are reliable.")
    elif overall_mean >= 0.70:
        summary.append("  MODERATE consistency (0.70-0.85): Most predictions are stable.")
        summary.append("  Some variation in borderline cases (abbreviated entity names,")
        summary.append("  ambiguous phenotypes). Acceptable for schema discovery.")
    else:
        summary.append("  LOW consistency (<0.70): Significant variation between runs.")
        summary.append("  Consider lowering temperature further or adding more examples.")


    for line in summary:
        print(line)
    lines.extend(summary)

    report_text = "\n".join(lines)
    REPORT_PATH.write_text(report_text)
    print(f"\nReport saved → {REPORT_PATH}")

    return {
        "mean_jaccard": overall_mean,
        "min_jaccard":  overall_min,
        "max_jaccard":  overall_max,
        "doc_results":  {k: [list(s) for s in v] for k, v in all_results.items()},
    }


if __name__ == "__main__":
    run_uncertainty_experiment()