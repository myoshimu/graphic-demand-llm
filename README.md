# Which Design Sells Better? -- anonymized code for review 

The sales data, product images, and the retailer's identity are confidential and are **not** included.
The code therefore cannot be run end to end without access to the retailer's data warehouse, but every analysis in the paper
is produced by the scripts here.

## Layout
- `src/ingest/` -- building the SKU, sales, and outcome tables from the data warehouse (BigQuery).
- `src/baselines/` -- baselines B0--B4 (LightGBM on metadata; CLIP ViT-L/14 embeddings with PCA; nearest neighbours).
- `src/panel/` -- LLM calls (Gemini and Claude on Vertex AI) with caching and a cost ledger; personas; Study 1 prompts; Study 2 pairwise prompts (`pairwise.py`).
- `src/eval/` -- metrics and cluster bootstrap.
- `scripts/` -- entry points, run through the `Makefile` (e.g. `make baselines`, `make pairwise`, `make pairwise-robustness`, `make figures`).
- `config/` -- paths, model settings (`models.yaml`), persona distribution (`personas.yaml`), baseline grids (`baselines.yaml`).
  Two files that list the retailer's products and collaborations are replaced by empty templates.

## Key entry points
- Study 1 (absolute ratings): `scripts/run_panel_stage1.py`, `scripts/eval_panel_stage1.py`.
- Study 2 (pairwise judgments): `scripts/run_pairwise.py`, `scripts/eval_pairwise.py`; 2026 confirmation `scripts/run_confirm.py`, `scripts/eval_confirm.py`.
- Robustness (prompt, model, persona, other model family): `scripts/run_pairwise_robustness.py`, `scripts/eval_pairwise_robustness.py`.
- Paper figures and tables: `scripts/make_figures.py`.

Comments in the code are partly in Japanese. Identifiers of the retailer, its products, and people have been masked.
