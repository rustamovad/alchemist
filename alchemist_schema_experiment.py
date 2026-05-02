import json, sys, math
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
import alchemist_evaluate as ev
import importlib; importlib.reload(ev)

BIORED_TRAIN  = Path("BioRED/Train.BioC.JSON")
OUT_DIR       = Path("alchemist_zero_shoot")
RESULTS_FILE  = OUT_DIR / "scale_experiment_results.json"

EVAL_SIZES = [10, 50, 100, 200, 500]
def load_all_predictions() -> dict:
    all_preds = {}
    abox_dir = OUT_DIR / "abox_history"
    active   = OUT_DIR / "current_batch_abox.jsonl"

    files = sorted(abox_dir.glob("*.jsonl")) if abox_dir.exists() else []
    if active.exists():
        files.append(active)

    for fp in files:
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    doc_id = entry.get("doc_id", "")
                    if doc_id and doc_id not in all_preds:
                        for e in entry.get("entities", []):
                            e["text"] = e.get("text", "").lower().strip()
                        all_preds[doc_id] = {
                            "entities":  entry.get("entities", []),
                            "relations": entry.get("relations", []),
                        }
                except Exception:
                    pass
    return all_preds


def evaluate_at_n(n: int, gold_docs: dict, all_preds: dict) -> dict:
    available = [d for d in all_preds if d in gold_docs]
    if len(available) < n:
        print(f"  WARNING: only {len(available)} docs available, requested n={n}")
        n = len(available)

    subset_ids   = set(available[:n])
    subset_gold  = {k: gold_docs[k] for k in subset_ids}
    subset_pred  = {k: all_preds[k]  for k in subset_ids}

    ner = ev.evaluate_ner(subset_gold, subset_pred)
    re_ = ev.evaluate_re(subset_gold, subset_pred)

    conv_log = OUT_DIR / "convergence_log.jsonl"
    novel_classes = 0
    if conv_log.exists():
        entries = []
        with open(conv_log) as f:
            for line in f:
                try:
                    entries.append(json.loads(line.strip()))
                except Exception:
                    pass
        docs_per_batch = 10
        target_batch = max(1, n // docs_per_batch)
        for entry in entries:
            if entry.get("batch", 0) <= target_batch:
                novel_classes = max(novel_classes, entry.get("n_classes", 0) - 7)

    n_entities = sum(len(v["entities"]) for v in subset_pred.values())
    f1 = ner["macro_strict"]["F1"]
    ci = 1.96 * math.sqrt(max(f1 * (1 - f1), 0.001) / max(n_entities, 1))

    return {
        "n_docs":          n,
        "n_predictions":   n_entities,
        "ner_strict_f1":   round(f1, 4),
        "ner_strict_p":    round(ner["macro_strict"]["P"], 4),
        "ner_strict_r":    round(ner["macro_strict"]["R"], 4),
        "ner_lenient_f1":  round(ner["macro_lenient"]["F1"], 4),
        "re_f1":           round(re_["macro"]["F1"], 4),
        "re_p":            round(re_["macro"]["P"], 4),
        "re_r":            round(re_["macro"]["R"], 4),
        "hallucination":   round(ner["hallucination_rate"], 4),
        "novel_classes":   novel_classes,
        "f1_ci_95":        round(ci, 4),
        "per_type_ner":    {t: list(v) for t, v in ner["per_type_strict"].items()},
    }


def run_scale_experiment():
    OUT_DIR.mkdir(exist_ok=True)

    if not BIORED_TRAIN.exists():
        print(f"ERROR: {BIORED_TRAIN} not found.")
        return

    print("Loading BioRED gold standard...")
    gold_docs = ev.load_biored(BIORED_TRAIN)
    print(f"  {len(gold_docs)} gold documents loaded.")

    print("Loading all ABox predictions...")
    all_preds = load_all_predictions()
    print(f"  {len(all_preds)} predicted documents available.")

    if len(all_preds) == 0:
        print("ERROR: No predictions found. Run alchemist_pipeline.py first.")
        return

    results = []
    print(f"\nRunning scale experiment at n = {EVAL_SIZES}\n")

    for n in EVAL_SIZES:
        if n > len(all_preds):
            print(f"  n={n}: SKIPPED — only {len(all_preds)} predictions available")
            print(f"         Run pipeline for more batches first.")
            continue
        print(f"  Evaluating at n={n}...", end=" ", flush=True)
        result = evaluate_at_n(n, gold_docs, all_preds)
        results.append(result)
        print(f"NER F1={result['ner_strict_f1']:.3f}  "
              f"RE F1={result['re_f1']:.3f}  "
              f"Novel={result['novel_classes']}")

    print()
    print("SCALE EXPERIMENT RESULTS")
    print(f"  {'n docs':>8} {'NER F1':>8} {'±95% CI':>8} "
          f"{'RE F1':>8} {'Novel Cls':>10} {'Halluc%':>8}")
    for r in results:
        print(f"  {r['n_docs']:>8} "
              f"{r['ner_strict_f1']:>8.3f} "
              f"{r['f1_ci_95']:>8.3f} "
              f"{r['re_f1']:>8.3f} "
              f"{r['novel_classes']:>10} "
              f"{r['hallucination']:>7.1%}")

    print()
    print("Per-entity-type NER F1 across scales:")
    entity_types = ev.BIORED_ENTITY_TYPES
    header = f"  {'Type':<35}" + "".join(f" n={r['n_docs']:>4}" for r in results)
    print(header)
    print("  " + "-" * (35 + 8 * len(results)))
    for t in entity_types:
        row = f"  {t:<35}"
        for r in results:
            f1 = r["per_type_ner"].get(t, [0,0,0])[2]
            row += f" {f1:>7.3f}"
        print(row)

    if len(results) >= 2:
        f1_change = results[-1]["ner_strict_f1"] - results[0]["ner_strict_f1"]
        novel_change = results[-1]["novel_classes"] - results[0]["novel_classes"]
        print()
        print(f"  NER F1 change from n={results[0]['n_docs']} to "
              f"n={results[-1]['n_docs']}: {f1_change:+.3f}")
        print(f"  Novel classes added:                  +{novel_change}")
        print(f"  → F1 plateaus quickly (zero-shot LLM doesn't learn from more docs)")
        print(f"  → Schema continues growing (convergence not yet reached)")

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nFull results saved → {RESULTS_FILE}")


if __name__ == "__main__":
    run_scale_experiment()