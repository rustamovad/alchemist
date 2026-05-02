import os, json, re, time, random, shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import requests
from tqdm import tqdm
from pydantic import BaseModel, Field, ValidationError


QWEN_API_KEY  = os.environ.get("QWEN_API_KEY",  "NotRequired")
QWEN_ENDPOINT = os.environ.get("QWEN_ENDPOINT",  "http://localhost:11434/v1/chat/completions")
QWEN_MODEL    = os.environ.get("QWEN_MODEL",     "qwen2.5:32b")

BIORED_TRAIN_FILE = Path(os.environ.get("BIORED_TRAIN", "BioRED/Train.BioC.JSON"))
BIORED_TEST_FILE  = Path(os.environ.get("BIORED_TEST",  "BioRED/Test.BioC.JSON"))

BATCH_SIZE  = 10
MAX_BATCHES = None

OUT_DIR            = Path("alchemist_zero_shoot")
TTL_SCHEMA_FILE    = OUT_DIR / "active_tbox.ttl"
NEW_SCHEMA_FILE    = OUT_DIR / "candidate_tbox.ttl"
TTL_HISTORY_DIR    = OUT_DIR / "ttl_history"
DATA_HISTORY_DIR   = OUT_DIR / "abox_history"
CURRENT_BATCH_DATA = OUT_DIR / "current_batch_abox.jsonl"
SUGGESTIONS_FILE   = OUT_DIR / "schema_suggestions.jsonl"
CONVERGENCE_LOG    = OUT_DIR / "convergence_log.jsonl"
STATE_FILE         = OUT_DIR / "pipeline_state.json"

DOMAIN_DESCRIPTION = (
    "biomedical research publications (PubMed abstracts) covering genes, proteins, "
    "diseases, chemical compounds, sequence variants, cell lines, and organisms"
)

ENTITY_TASK_DESCRIPTION = (
    "Identify and extract all named biomedical entities from the text. "
    "Classify each entity into exactly one of the following types: "
    "GeneOrGeneProduct, DiseaseOrPhenotypicFeature, ChemicalEntity, "
    "SequenceVariant, CellLine, OrganismTaxon. "
    "Extract every distinct surface form that appears in the text, including "
    "abbreviations and full expansions. Copy entity text verbatim from the source. "
    "Normalize relation phrases to short canonical verbs. "
    "Extract all meaningful relationships from the text."
)

RELATION_TASK_DESCRIPTION = (
    "Identify all pairwise relationships between the provided entities based solely "
    "on what is explicitly stated in the text. "
    "Classify each relationship into exactly one of the following types: "
    "Association, Positive_Correlation, Negative_Correlation, Bind, "
    "Conversion, Cotreatment, Comparison, Drug_Interaction. "
    "Use only entities from the provided list as subject and object. "
    "Normalize relation phrases to short canonical verbs (1-3 words). "
    "Remove auxiliary verbs and modifiers. "
    "Extract all meaningful relationships from the text."
)

SCHEMA_TASK_DESCRIPTION = (
    "Analyze the biomedical schema suggestions collected from literature mining and "
    "propose additions to an existing OWL ontology in the domain. "
    "Add only genuinely novel concepts not already represented. "
    "Maintain valid Turtle (TTL) syntax throughout. "
    "Output the complete updated ontology, not just the new additions."
)

BIORED_ENTITY_TYPES = [
    "GeneOrGeneProduct",
    "DiseaseOrPhenotypicFeature",
    "ChemicalEntity",
    "SequenceVariant",
    "CellLine",
    "OrganismTaxon",
]

BIORED_RELATION_TYPES = [
    "Association",
    "Positive_Correlation",
    "Negative_Correlation",
    "Bind",
    "Conversion",
    "Cotreatment",
    "Comparison",
    "Drug_Interaction",
]


class ExtractedEntity(BaseModel):
    text:        str
    entity_type: str
    identifier:  Optional[str] = None

class ExtractedRelation(BaseModel):
    subject:          str
    relation_phrase:  str
    object:           str
    confidence:       float = Field(default=1.0, ge=0.0, le=1.0)

class Suggestion(BaseModel):
    type:    str
    context: str

class NEROutput(BaseModel):
    doc_id:      str
    entities:    List[ExtractedEntity] = Field(default_factory=list)
    suggestions: List[Suggestion]      = Field(default_factory=list)

class REOutput(BaseModel):
    doc_id:    str
    relations: List[ExtractedRelation] = Field(default_factory=list)
    suggestions: List[Suggestion]      = Field(default_factory=list)

class ExtractionOutput(BaseModel):
    doc_id:      str
    entities:    List[ExtractedEntity]   = Field(default_factory=list)
    relations:   List[ExtractedRelation] = Field(default_factory=list)
    suggestions: List[Suggestion]        = Field(default_factory=list)


def load_biored(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    docs = []
    collection = raw if isinstance(raw, list) else raw.get("documents", [])
    for doc in collection:
        doc_id        = doc.get("id", "")
        title_text    = ""
        abstract_text = ""
        annotations   = []
        relations     = doc.get("relations", [])
        for passage in doc.get("passages", []):
            t      = passage.get("text", "")
            offset = passage.get("offset", 0)
            if offset == 0:
                title_text = t
            else:
                abstract_text = t
            for ann in passage.get("annotations", []):
                annotations.append({
                    "text": ann.get("text", ""),
                    "type": ann["infons"].get("type", ""),
                    "id":   ann["infons"].get("identifier", ""),
                })
        docs.append({
            "id":          doc_id,
            "title":       title_text,
            "abstract":    abstract_text,
            "full_text":   f"{title_text}\n\n{abstract_text}",
            "annotations": annotations,
            "relations":   relations,
        })
    return docs


def load_state() -> Dict[str, Any]:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"processed_ids": [], "batch_count": 0}


def save_state(state: Dict[str, Any]):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def get_next_batch(docs: List[Dict], state: Dict) -> List[Dict]:
    done      = set(state["processed_ids"])
    remaining = [d for d in docs if d["id"] not in done]
    n         = min(BATCH_SIZE, len(remaining))
    return random.sample(remaining, n) if n else []


def _qwen_call(messages: list, max_tokens: int = 2048,
               json_mode: bool = False) -> Optional[str]:
    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload: Dict[str, Any] = {
        "model":       QWEN_MODEL,
        "messages":    messages,
        "max_tokens":  max_tokens,
        "temperature": 0.1,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    for attempt in range(3):
        try:
            r = requests.post(QWEN_ENDPOINT, headers=headers,
                              json=payload, timeout=600)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
            else:
                print(f"  [API ERROR] {e}")
                return None


def _strip_fences(text: str) -> str:
    text = re.sub(r"^```[a-z]*\n?", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\n?```$",        "", text.strip(), flags=re.MULTILINE)
    return text.strip()


MINIMAL_TTL = """\
@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix :     <http://alchemist.bio/ontology#> .

: a owl:Ontology ;
    rdfs:label "ALCHEMIST BioRED Ontology" .

:BiomedicalEntity a owl:Class ;
    rdfs:label "Biomedical Entity" ;
    rdfs:comment "Abstract superclass for all biomedical named entities." .

:GeneOrGeneProduct a owl:Class ;
    rdfs:subClassOf :BiomedicalEntity ;
    rdfs:comment "A gene, protein, RNA, or any gene product." .

:DiseaseOrPhenotypicFeature a owl:Class ;
    rdfs:subClassOf :BiomedicalEntity ;
    rdfs:comment "A disease, syndrome, symptom, or abnormal phenotype." .

:ChemicalEntity a owl:Class ;
    rdfs:subClassOf :BiomedicalEntity ;
    rdfs:comment "A drug, compound, metabolite, or other chemical substance." .

:SequenceVariant a owl:Class ;
    rdfs:subClassOf :BiomedicalEntity ;
    rdfs:comment "A genetic mutation, SNP, or other sequence variation." .

:CellLine a owl:Class ;
    rdfs:subClassOf :BiomedicalEntity ;
    rdfs:comment "A cultured cell line used in biomedical research." .

:OrganismTaxon a owl:Class ;
    rdfs:subClassOf :BiomedicalEntity ;
    rdfs:comment "A species or organism referenced in biomedical text." .

:hasIdentifier a owl:DatatypeProperty ;
    rdfs:domain :BiomedicalEntity ;
    rdfs:range  xsd:string ;
    rdfs:comment "NCBI Gene ID, MeSH ID, dbSNP ID, etc." .

:hasSurfaceForm a owl:DatatypeProperty ;
    rdfs:domain :BiomedicalEntity ;
    rdfs:range  xsd:string ;
    rdfs:comment "Textual mention in the source document." .

:Association a owl:ObjectProperty ;
    rdfs:domain :BiomedicalEntity ;
    rdfs:range  :BiomedicalEntity ;
    rdfs:comment "Generic biomedical association." .

:Positive_Correlation a owl:ObjectProperty ;
    rdfs:subPropertyOf :Association ;
    rdfs:comment "Entity A positively correlates with entity B." .

:Negative_Correlation a owl:ObjectProperty ;
    rdfs:subPropertyOf :Association ;
    rdfs:comment "Entity A negatively correlates with entity B." .

:Bind a owl:ObjectProperty ;
    rdfs:subPropertyOf :Association ;
    rdfs:comment "Direct physical binding between entities." .

:Cotreatment a owl:ObjectProperty ;
    rdfs:subPropertyOf :Association ;
    rdfs:comment "Co-administration of chemical entities." .

:Drug_Interaction a owl:ObjectProperty ;
    rdfs:subPropertyOf :Association ;
    rdfs:comment "Pharmacological drug-drug interaction." .

:Conversion a owl:ObjectProperty ;
    rdfs:subPropertyOf :Association ;
    rdfs:comment "One entity is converted into another." .

:Comparison a owl:ObjectProperty ;
    rdfs:subPropertyOf :Association ;
    rdfs:comment "Comparative statement between two entities." .
"""


def load_or_create_tbox() -> str:
    if not TTL_SCHEMA_FILE.exists():
        TTL_SCHEMA_FILE.write_text(MINIMAL_TTL, encoding="utf-8")
    return TTL_SCHEMA_FILE.read_text(encoding="utf-8")


def count_classes_in_ttl(ttl: str) -> int:
    return len(re.findall(r'owl:Class', ttl))


def count_properties_in_ttl(ttl: str) -> int:
    return len(re.findall(r'owl:(?:Object|Datatype)Property', ttl))


def bootstrap_initial_schema(sample_docs: List[Dict]) -> str:
    print("\n[Bootstrap] Generating initial TBox...")
    sample_text = "\n\n".join(
        f"PMID {d['id']}: {d['full_text'][:800]}" for d in sample_docs[:8]
    )
    prompt = f"""Domain context: {DOMAIN_DESCRIPTION}

Task: Build a complete OWL ontology in Turtle (TTL) format covering all entity and relation types present in the domain.
Include these entity types as owl:Class: {", ".join(BIORED_ENTITY_TYPES)}
Include these relation types as owl:ObjectProperty: {", ".join(BIORED_RELATION_TYPES)}
Add any additional classes or properties discovered in the text.
Use prefix `:` for `http://alchemist.bio/ontology#` and declare standard prefixes.
Add rdfs:comment for every class and property.
Output ONLY valid Turtle syntax, no markdown fences.

Text:
{sample_text}

Turtle TTL:"""

    messages = [
        {"role": "system", "content": "You are a biomedical ontology engineer. Output only valid Turtle TTL."},
        {"role": "user",   "content": prompt},
    ]
    result = _qwen_call(messages, max_tokens=4096)
    if not result or len(result.strip()) < 200:
        return MINIMAL_TTL
    return _strip_fences(result)


def _ner_pass(doc: Dict) -> Optional[NEROutput]:
    schema = NEROutput.model_json_schema()
    prompt = f"""Domain context: {DOMAIN_DESCRIPTION}

Task: {ENTITY_TASK_DESCRIPTION}

Valid entity types: {", ".join(BIORED_ENTITY_TYPES)}

For each entity output a JSON object with exactly these keys:
  "text"        — the verbatim surface form copied from the text
  "entity_type" — one of the valid entity types listed above
  "identifier"  — NCBI/MeSH identifier if inferrable, otherwise null

If you encounter a biologically meaningful concept that does not fit any entity type, add it to "suggestions" with fields "type" (proposed class name) and "context" (verbatim snippet).

Output ONLY a valid JSON object matching this schema. No prose, no explanation.
Schema: {json.dumps(schema, indent=2)}

Document ID: {doc["id"]}
Text:
{doc["full_text"][:3000]}

JSON object:"""

    messages = [
        {"role": "system", "content": "You are a biomedical information extraction system. Output valid JSON only."},
        {"role": "user",   "content": prompt},
    ]
    raw = _qwen_call(messages, max_tokens=3000, json_mode=True)
    if not raw:
        return None
    try:
        data = json.loads(_strip_fences(raw))
        data["doc_id"] = doc["id"]
        return NEROutput.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"  [NER parse error {doc['id']}]: {e}")
        return None


def _re_pass(doc: Dict, entities: List[ExtractedEntity]) -> Optional[REOutput]:
    if not entities:
        return REOutput(doc_id=doc["id"], relations=[])

    seen         = set()
    entity_lines = []
    for e in entities:
        key = (e.text.lower(), e.entity_type)
        if key not in seen:
            seen.add(key)
            entity_lines.append(f'  "{e.text}"  [{e.entity_type}]')
    entity_list_str = "\n".join(entity_lines[:60])

    schema = REOutput.model_json_schema()
    prompt = f"""Domain context: {DOMAIN_DESCRIPTION}

Task: {RELATION_TASK_DESCRIPTION}

Valid relation types: {", ".join(BIORED_RELATION_TYPES)}

Entities extracted from the text (use ONLY these exact strings for subject and object):
{entity_list_str}

For each relationship output a JSON object with exactly these keys:
  "subject"         — the entity doing or receiving something (must match entity list exactly)
  "relation_phrase" — a short canonical verb phrase (1-3 words) describing the relationship
  "object"          — the other entity (must match entity list exactly)
  "confidence"      — a score from 0 to 1 indicating how confident you are in this extraction

If you encounter a meaningful relationship that does not fit any valid relation type, add it to "suggestions" with fields "type" (proposed relation name) and "context" (verbatim snippet).

Output ONLY a valid JSON object matching this schema. No prose, no explanation.
Schema: {json.dumps(schema, indent=2)}

Document ID: {doc["id"]}
Text:
{doc["full_text"][:3000]}

JSON object:"""

    messages = [
        {"role": "system", "content": "You are a biomedical relation extraction system. Output valid JSON only."},
        {"role": "user",   "content": prompt},
    ]
    raw = _qwen_call(messages, max_tokens=2000, json_mode=True)
    if not raw:
        return None
    try:
        data = json.loads(_strip_fences(raw))
        data["doc_id"] = doc["id"]
        return REOutput.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"  [RE parse error {doc['id']}]: {e}")
        return None


def extract_from_doc(doc: Dict, current_ttl: str) -> Optional[ExtractionOutput]:
    ner_result = _ner_pass(doc)
    if ner_result is None:
        return None
    re_result = _re_pass(doc, ner_result.entities)
    
    all_suggestions = ner_result.suggestions
    if re_result and hasattr(re_result, 'suggestions') and re_result.suggestions:
        all_suggestions.extend(re_result.suggestions)
        
    return ExtractionOutput(
        doc_id      = doc["id"],
        entities    = ner_result.entities,
        relations   = re_result.relations if re_result else [],
        suggestions = all_suggestions,
    )


def improve_tbox(current_ttl: str, suggestions: List[Dict]) -> str:
    if not suggestions:
        return current_ttl
    prompt = f"""Domain context: {DOMAIN_DESCRIPTION}

Task: {SCHEMA_TASK_DESCRIPTION}

Current ontology (Turtle TTL):
{current_ttl[:4000]}

Schema suggestions from the latest batch of documents:
{json.dumps(suggestions[:20], indent=2)}

Output the complete updated Turtle TTL. No markdown fences, no explanations.

Updated TTL:"""

    messages = [
        {"role": "system", "content": "You are a biomedical ontology engineer. Output only valid Turtle TTL."},
        {"role": "user",   "content": prompt},
    ]
    result = _qwen_call(messages, max_tokens=4096)
    if not result or len(result.strip()) < 200:
        return current_ttl
    return _strip_fences(result)


def archive_and_promote(batch_num: int):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    TTL_HISTORY_DIR.mkdir(exist_ok=True)
    DATA_HISTORY_DIR.mkdir(exist_ok=True)
    if TTL_SCHEMA_FILE.exists():
        shutil.copy(TTL_SCHEMA_FILE,
                    TTL_HISTORY_DIR / f"tbox_b{batch_num:03d}_{ts}.ttl")
    if NEW_SCHEMA_FILE.exists():
        shutil.copy(NEW_SCHEMA_FILE, TTL_SCHEMA_FILE)
        print(f"  TBox promoted (batch {batch_num})")
    if CURRENT_BATCH_DATA.exists():
        shutil.move(str(CURRENT_BATCH_DATA),
                    str(DATA_HISTORY_DIR / f"abox_b{batch_num:03d}_{ts}.jsonl"))


def log_convergence(batch_num: int, docs_done: int, ttl: str):
    entry = {
        "batch":        batch_num,
        "docs_done":    docs_done,
        "n_classes":    count_classes_in_ttl(ttl),
        "n_properties": count_properties_in_ttl(ttl),
        "timestamp":    datetime.now().isoformat(),
    }
    with open(CONVERGENCE_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def run_pipeline():
    OUT_DIR.mkdir(exist_ok=True)
    TTL_HISTORY_DIR.mkdir(exist_ok=True)
    DATA_HISTORY_DIR.mkdir(exist_ok=True)

    if not BIORED_TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"BioRED training file not found: {BIORED_TRAIN_FILE}\n"
            "Download from https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/"
        )

    print(f"Loading BioRED from {BIORED_TRAIN_FILE}...")
    train_docs = load_biored(BIORED_TRAIN_FILE)
    print(f"  {len(train_docs)} documents loaded.")

    tbox = load_or_create_tbox()
    if count_classes_in_ttl(tbox) <= 2:
        sample = random.sample(train_docs, min(10, len(train_docs)))
        tbox   = bootstrap_initial_schema(sample)
        TTL_SCHEMA_FILE.write_text(tbox, encoding="utf-8")
        print(f"  Bootstrap TBox: {count_classes_in_ttl(tbox)} classes, "
              f"{count_properties_in_ttl(tbox)} properties.")

    state     = load_state()
    batch_num = state["batch_count"]

    while True:
        if MAX_BATCHES and batch_num >= MAX_BATCHES:
            print(f"\nReached MAX_BATCHES={MAX_BATCHES}.")
            break

        batch = get_next_batch(train_docs, state)
        if not batch:
            print("\nAll training documents processed.")
            break

        batch_num += 1
        print(f"\nBATCH {batch_num}  ({len(batch)} docs | {len(state['processed_ids'])} done)")

        all_suggestions = []
        newly_done      = []
        tbox            = load_or_create_tbox()

        abox_f = open(CURRENT_BATCH_DATA, "a", encoding="utf-8")
        sugg_f = open(SUGGESTIONS_FILE,   "a", encoding="utf-8")

        for doc in tqdm(batch, desc=f"  Batch {batch_num}"):
            result = extract_from_doc(doc, tbox)
            if result is None:
                continue

            abox_f.write(json.dumps({
                "doc_id":    result.doc_id,
                "entities":  [e.model_dump() for e in result.entities],
                "relations": [r.model_dump() for r in result.relations],
            }) + "\n")

            for s in result.suggestions:
                sd = s.model_dump()
                sd["source_doc"] = doc["id"]
                sugg_f.write(json.dumps(sd) + "\n")
                all_suggestions.append(sd)

            newly_done.append(doc["id"])

        abox_f.close()
        sugg_f.close()

        new_tbox = improve_tbox(tbox, all_suggestions)
        NEW_SCHEMA_FILE.write_text(new_tbox, encoding="utf-8")

        archive_and_promote(batch_num)

        tbox  = load_or_create_tbox()
        entry = log_convergence(
            batch_num,
            len(state["processed_ids"]) + len(newly_done),
            tbox,
        )
        print(f"  Schema: {entry['n_classes']} classes, {entry['n_properties']} properties | "
              f"{len(all_suggestions)} suggestions")

        state["processed_ids"].extend(newly_done)
        state["batch_count"] = batch_num
        save_state(state)

    print("\nPipeline complete.")
    print(f"  Active TBox : {TTL_SCHEMA_FILE}")
    print(f"  Convergence : {CONVERGENCE_LOG}")
    print(f"  ABox        : {DATA_HISTORY_DIR}")


if __name__ == "__main__":
    run_pipeline()