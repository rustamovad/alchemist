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

ALCHEMIST is a computer program that uses Artificial Intelligence (AI) to read medical articles. It automatically finds important things like genes, diseases, and drugs, and figures out how they are connected to each other.

## What the files do

Here is a simple breakdown of the files in this project:

### The Main Workers
* **`pipeline.py`**,  **`evaluate.py`**: These are the main programs. They take medical texts, send them to the AI, ask the AI to pull out the important words and their relationships, and save the results.


### The Testers
* **`alchemist_uncertainity.py`**: The double-checker. It asks the AI the exact same question three times to make sure the AI doesn't change its mind and gives consistent answers.
* **`alchemist_schema_experiment.py`**: The scale tester. It checks how the system behaves when you feed it a little bit of data versus a lot of data.

### The Comparisons (Old School vs. New School)
* **`alchemist_train_bert.py`** & **`alchemist_biobert_eval.py`**: These files train and test an older, more traditional type of AI (called BioBERT). We use this to compare against our new, smarter AI to see which one does a better job.
