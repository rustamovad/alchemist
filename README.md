# ALCHEMIST

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