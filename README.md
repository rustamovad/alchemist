# ALCHEMIST

ALCHEMIST is  a Python tool that uses a zero-shot LLM to read medical abstracts (from the BioRED dataset) and automatically build a knowledge map of how genes, diseases, and drugs connect.

Instead of manually training a model, we built a strict two-pass prompting pipeline using Pydantic to keep the AI from hallucinating.

## File breakdown

Here is what the scripts actually do if you want to poke around:
 
### The Core Pipeline
* `pipeline.py` & `evaluate.py`
  These are the main engines of the project. Run these to see the two-pass extraction in action. They ingest the raw text from the BioRED dataset, force the LLM to extract the entities, and output the final, valid OWL schema.

### Experiments & Validation
* `alchemist_schema_experiment.py`
  This script generates the data for the "Schema Growth Curve" discussed in my paper. It tests the system on increasing batch sizes (from 10 up to 400 documents) to track when the schema stops finding new biological rules.
* `alchemist_uncertainity.py`
  This is the consistency checker. It runs the exact same prompt three times on the same text and uses Jaccard similarity to ensure the AI's extractions are stable and reliable.

### The Supervised Baseline
* `alchemist_train_bert.py` & `alchemist_biobert_eval.py`
  These scripts fine-tune and evaluate a standard BioBERT model. I trained this locally to provide a traditional, supervised machine learning baseline to compare against my zero-shot ALCHEMIST tool.

## Requirements
To run this code, you will need Python installed along with the following libraries:
* `pydantic`
* `rdflib` (for building the OWL ontology)
* An active API key for Qwen-turbo# ALCHEMIST
