# ALQAC Case Evidence Retrieval

This repository contains the **case evidence retrieval** module developed for the ALQAC task. Given a case identifier and a short case description, the system iteratively queries the Case Content API to retrieve relevant and non-duplicate case-fact chunks.

The retrieval pipeline combines:

- lexical and semantic seed-query generation;
- source-aware probing for the plaintiff, defendant, witness, and procurator;
- adaptive query ranking;
- chunk-level query expansion;
- periodic Evidence Map and Source Coverage updates;
- duplicate detection and query mutation;
- dynamic retrieval budgets and stopping conditions.

The retrieved case facts can be used by downstream modules for legal-provision retrieval and judicial decision prediction.

## Installation and Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Using a virtual environment is recommended:

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 2. Configure environment variables

Open the `.env` file and replace the placeholder values with your local paths and ALQAC API token:

```env
INPUT_PATH="/path/to/input_file"
RUN_DIR="/path/to/output_directory"
ALQAC_TOKEN="your_alqac_token"
```

- `INPUT_PATH`: path to the input case file.
- `RUN_DIR`: directory used to store retrieval outputs, logs, caches, and checkpoints.
- `ALQAC_TOKEN`: authentication token for the ALQAC Case Content API.

### 3. Run the retrieval pipeline

```bash
python run.py
```

The program reads the configuration from `.env`, processes the input cases, and writes the generated evidence files to `RUN_DIR`.

## Notes

- Ensure that `INPUT_PATH` exists before running the program.
- Ensure that `RUN_DIR` is writable.
- A valid `ALQAC_TOKEN` is required to access the Case Content API.
- Retrieval may take time because API requests are rate-limited and each case is processed iteratively.

