import os, json, argparse
from pathlib import Path

CHECKPOINT_PATH = os.environ.get("BIOBERT_CHECKPOINT", None)

BIORED_TRAIN = Path("BioRED/Train.BioC.JSON")
BIORED_TEST  = Path("BioRED/Test.BioC.JSON")
OUT_DIR      = Path("alchemist_output")
BIOBERT_ABOX = OUT_DIR / "biobert_predictions.jsonl"

BIORED_LABELS = [
    "O",
    "B-GeneOrGeneProduct",  "I-GeneOrGeneProduct",
    "B-DiseaseOrPhenotypicFeature", "I-DiseaseOrPhenotypicFeature",
    "B-ChemicalEntity",     "I-ChemicalEntity",
    "B-SequenceVariant",    "I-SequenceVariant",
    "B-CellLine",           "I-CellLine",
    "B-OrganismTaxon",      "I-OrganismTaxon",
]

def find_checkpoint() -> Path | None:
    search_roots = [
        Path.home(),
        Path("."),
        Path("/tmp"),
    ]
    candidates = []

    for root in search_roots:
        for p in root.rglob("trainer_state.json"):
            ckpt_dir = p.parent
            has_weights = (
                (ckpt_dir / "pytorch_model.bin").exists() or
                (ckpt_dir / "model.safetensors").exists() or
                any(ckpt_dir.glob("model-*.safetensors"))
            )
            has_config = (ckpt_dir / "config.json").exists()
            if has_weights and has_config:
                mtime = p.stat().st_mtime
                candidates.append((mtime, ckpt_dir))
                print(f"  Found: {ckpt_dir}")

    if not candidates:
        print("\nNo checkpoint found automatically.")
        return None

    candidates.sort(reverse=True)
    best = candidates[0][1]
    print(f"\nUsing checkpoint: {best}")
    return best

def load_biored_docs(path: Path) -> list:
    with open(path) as f:
        raw = json.load(f)
    collection = raw if isinstance(raw, list) else raw.get("documents", [])
    docs = []
    for doc in collection:
        doc_id = doc.get("id", "")
        passages = doc.get("passages", [])
        full_text = " ".join(p.get("text", "") for p in passages)
        annotations = []
        for passage in passages:
            for ann in passage.get("annotations", []):
                annotations.append({
                    "text":       ann.get("text", "").lower().strip(),
                    "type":       ann["infons"].get("type", ""),
                    "identifier": ann["infons"].get("identifier", ""),
                    "offset":     ann.get("locations", [{}])[0].get("offset", 0),
                    "length":     ann.get("locations", [{}])[0].get("length", 0),
                })
        docs.append({
            "id":          doc_id,
            "full_text":   full_text,
            "annotations": annotations,
            "relations":   doc.get("relations", []),
        })
    return docs

def run_biobert_inference(checkpoint: Path, docs: list) -> dict:
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
    except ImportError:
        print("ERROR: transformers not installed.")
        print("Run: pip install transformers torch --break-system-packages")
        return {}

    print(f"\nLoading fine-tuned model from {checkpoint} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
        model     = AutoModelForTokenClassification.from_pretrained(str(checkpoint))
    except Exception as e:
        print(f"ERROR loading model: {e}")
        print("\nTroubleshooting:")
        print("  1. Check the checkpoint path is correct")
        print("  2. Make sure pytorch is installed: pip install torch")
        return {}

    ner_pipeline = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",  
        device=-1,  
    )

    all_predictions = {}
    print(f"Running inference on {len(docs)} documents...")

    for i, doc in enumerate(docs):
        if i % 20 == 0:
            print(f"  {i}/{len(docs)} docs processed...")
        try:
            raw_preds = ner_pipeline(
                doc["full_text"][:2000],  
                batch_size=8
            )
            entities = []
            for pred in raw_preds:
                entity_type = pred.get("entity_group", pred.get("entity", ""))
                entity_type = entity_type.lstrip("BI-").strip()
                if entity_type == "O" or not entity_type:
                    continue
                entities.append({
                    "text":        pred["word"].strip(),
                    "entity_type": entity_type,
                    "identifier":  None,
                    "score":       round(pred.get("score", 0.0), 4),
                })
            all_predictions[doc["id"]] = {"entities": entities, "relations": []}
        except Exception as e:
            print(f"  [ERROR on doc {doc['id']}]: {e}")
            all_predictions[doc["id"]] = {"entities": [], "relations": []}

    print(f"  Done. {len(all_predictions)} docs processed.")
    return all_predictions

def save_as_abox(predictions: dict, output_path: Path):
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        for doc_id, pred in predictions.items():
            entry = {
                "doc_id":    doc_id,
                "entities":  pred["entities"],
                "relations": pred.get("relations", []),
            }
            f.write(json.dumps(entry) + "\n")
    print(f"\nSaved {len(predictions)} doc predictions → {output_path}")

def run_alchemist_eval(split: str):
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    try:
        import alchemist_evaluate as ev
        import importlib; importlib.reload(ev)
    except ImportError:
        print("ERROR: alchemist_evaluate.py not found in current directory.")
        return

    ev.ACTIVE_ABOX   = BIOBERT_ABOX
    ev.ABOX_DIR      = Path("/tmp/nonexistent_so_only_abox_is_used")
    ev.EVAL_MODE     = "train_full" if split == "train" else "test_inference"
    ev.EVAL_OUTPUT_DIR = OUT_DIR / "evaluation_biobert"
    ev.EVAL_OUTPUT_DIR.mkdir(exist_ok=True)

    gold_file = BIORED_TRAIN if split == "train" else BIORED_TEST

    print(f"\nEvaluating BioBERT predictions against {gold_file} gold standard...")

    gold_docs = ev.load_biored(gold_file)
    pred_docs = ev.load_all_abox_predictions()
    pred_docs = {k: v for k, v in pred_docs.items() if k in gold_docs}

    overlap = sum(1 for d in gold_docs if d in pred_docs)
    print(f"Gold: {len(gold_docs)}  Pred: {len(pred_docs)}  Overlap: {overlap}")

    ner  = ev.evaluate_ner(gold_docs, pred_docs)
    re_  = ev.evaluate_re(gold_docs, pred_docs)
    tbox = {"total_classes": 0, "total_properties": 0,
            "biored_entity_coverage": {"covered": [], "missing": [], "rate": 0},
            "biored_relation_coverage": {"covered": [], "missing": [], "rate": 0},
            "novel_classes": []}
    rand = ev.compute_random_baseline(gold_docs)

    meta = {
        "mode":      f"biobert_fine_tuned_{split}",
        "gold_docs": len(gold_docs),
        "pred_docs": len(pred_docs),
        "overlap":   overlap,
    }

    report_path = ev.EVAL_OUTPUT_DIR / "biobert_eval_report.txt"
    ev.write_report(ner, re_, tbox, meta, report_path,
                    random_bl=rand, baselines=ev.PUBLISHED_BASELINES)

    print("COMPARISON: Fine-tuned BioBERT vs. Zero-shot ALCHEMIST")
    alch_ner = 0.389  
    alch_re  = 0.181
    bert_ner = ner["macro_strict"]["F1"]
    bert_re  = re_["macro"]["F1"]
    print(f"  {'System':<35} {'NER F1':>8} {'RE F1':>8}")
    print(f"  {'-'*53}")
    print(f"  {'Random baseline':<35} {'0.167':>8} {'0.125':>8}")
    print(f"  {'ALCHEMIST Qwen-turbo (zero-shot)':<35} {alch_ner:>8.3f} {alch_re:>8.3f}")
    print(f"  {'BioBERT fine-tuned (this work)':<35} {bert_ner:>8.3f} {bert_re:>8.3f}")
    print(f"  {'BioRED paper baseline (fine-tuned)':<35} {'0.852':>8} {'0.378':>8}")
    print(f"\n  BioBERT improvement over ALCHEMIST: "
          f"NER +{bert_ner-alch_ner:.3f}  RE +{bert_re-alch_re:.3f}")
    print(f"  Full report → {report_path}")

def main():
    parser = argparse.ArgumentParser(description="BioBERT NER evaluation for ALCHEMIST")
    parser.add_argument("--find",  action="store_true", help="Find checkpoint only")
    parser.add_argument("--eval",  action="store_true", help="Run full evaluation")
    parser.add_argument("--split", default="train", choices=["train", "test"],
                        help="Which BioRED split to evaluate on (default: train)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to fine-tuned checkpoint (overrides auto-search)")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    ckpt = Path(args.checkpoint) if args.checkpoint else \
           (Path(CHECKPOINT_PATH) if CHECKPOINT_PATH else None)

    if args.find or (args.eval and ckpt is None):
        ckpt = find_checkpoint()

    if args.find:
        if ckpt:
            print(f"\nCheckpoint found: {ckpt}")
            print(f"Run evaluation: python alchemist_biobert_eval.py --eval --checkpoint {ckpt}")
        return

    if args.eval:
        if ckpt is None:
            print("ERROR: No checkpoint found. Use --checkpoint /path/to/model")
            return

        biored_file = BIORED_TRAIN if args.split == "train" else BIORED_TEST
        if not biored_file.exists():
            print(f"ERROR: BioRED file not found: {biored_file}")
            return

        docs = load_biored_docs(biored_file)
        print(f"Loaded {len(docs)} {args.split} documents from BioRED")

        predictions = run_biobert_inference(ckpt, docs)
        if not predictions:
            print("ERROR: No predictions generated. Check the checkpoint.")
            return

        save_as_abox(predictions, BIOBERT_ABOX)
        run_alchemist_eval(args.split)

    else:
        parser.print_help()
        print("  python alchemist_biobert_eval.py --find")
        print("  python alchemist_biobert_eval.py --eval --checkpoint /path/to/model")

if __name__ == "__main__":
    main()