# CLAUDE.md §6: すべての分析スクリプトは scripts/ から引数なしで再実行可能にし、
#                Makefile に登録する。各フェーズは `make phaseN` で再現できること。

PYTHON ?= python
SCRIPTS := scripts

.DEFAULT_GOAL := help
.PHONY: help human-eval setup check-policy fetch-images-smoke fetch-images build-tables \
        graphic-id ec-listed population data-quality persona-aggregates phase0 embeddings baselines panel-pilot panel evaluate contamination attributes \
        phase1 prereg rct phase2 benchmark figures phase3 test lint clean-cache

help: ## 利用可能なターゲットを表示
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------
# セットアップ
# ---------------------------------------------------------------
setup: ## 依存関係のインストール（パッケージマネージャ未確定: questions.md E3）
	@echo "TODO(E3): uv / poetry のいずれかを確定してから実装する"
	@exit 1

check-policy: ## IP 画像の外部送信・掲載ポリシーの表示（作業前に必ず確認）
	@$(PYTHON) -c "import yaml; p=yaml.safe_load(open('config/ip_api_policy.yaml')); a=p['approval']; \
	print('allow_external_api_for_ip =', p['allow_external_api_for_ip'], '（retailer 許可', a.get('approved_on'), '）'); \
	print('allow_publish_ip_images   =', p['allow_publish_ip_images'], \
	      '：IP 画像を論文・図表・ベンチマークに載せない（CLAUDE.md §2.1）' if not p['allow_publish_ip_images'] else '')"

# ---------------------------------------------------------------
# フェーズ0：データ整備
# ---------------------------------------------------------------
fetch-images-smoke: ## 画像取得の動作確認（20 件のみ。§7-4）
	$(PYTHON) $(SCRIPTS)/fetch_images_smoke.py

fetch-images: ## 商品画像を全件取得（1 req/sec 以下、キャッシュ有効）
	$(PYTHON) $(SCRIPTS)/fetch_images.py

build-tables: ## §3.3 の正規化テーブルを生成（pydantic でスキーマ検証）
	$(PYTHON) $(SCRIPTS)/build_tables.py

graphic-id: ## 画像ハッシュの距離分布を出し、graphic_id を導出（§3.3 / §7-6）
	$(PYTHON) $(SCRIPTS)/graphic_id.py

ec-listed: ## 閲覧ログから EC 掲載フラグを導出（A19。BQ を約 670 GB 走査するので注意）
	$(PYTHON) $(SCRIPTS)/ec_listed.py

population: ## 分析母集団 data/processed/population.parquet を確定（A31。BQ は使わない）
	$(PYTHON) $(SCRIPTS)/population.py

data-quality: ## reports/phase0_data_quality.md を生成
	$(PYTHON) $(SCRIPTS)/phase0_data_quality.py

phase0: ec-listed fetch-images build-tables population graphic-id data-quality test ## フェーズ0 一式

# ---------------------------------------------------------------
# フェーズ1：後ろ向き検証
# ---------------------------------------------------------------
embeddings: ## 母集団画像の open_clip 埋め込み（B2/B4 の入力。CPU で約 1 時間、計算済みは再計算しない）
	$(PYTHON) $(SCRIPTS)/embed_images.py

baselines: embeddings ## B0-B4 ベースラインの学習・評価 → reports/phase1_baselines.md（§5.1）
	$(PYTHON) $(SCRIPTS)/run_baselines.py

persona-aggregates: ## ペルソナ設計用の顧客分布を BQ で集計 → reports/persona_distribution.md（個票は取得しない）
	$(PYTHON) $(SCRIPTS)/persona_aggregates.py

panel-pilot: ## 合成パネルのパイロット（100 SKU × 2 条件 × 50 人 × 2 モデル、上限 $100）→ reports/panel_pilot.md
	$(PYTHON) $(SCRIPTS)/run_panel_pilot.py

panel-stage1: ## 段階 1：テスト期間に Pro のパネル（1 商品ずつの評価、上限 $260）→ 評価 reports/panel_stage1.md
	$(PYTHON) $(SCRIPTS)/run_panel_stage1.py
	$(PYTHON) $(SCRIPTS)/eval_panel_stage1.py

pairwise: ## 2 案比較（同じ IP 内のデザイン優劣、2025 年、上限 $100）→ reports/pairwise.md
	$(PYTHON) $(SCRIPTS)/run_pairwise.py
	$(PYTHON) $(SCRIPTS)/eval_pairwise.py

pairwise-confirm: ## 2 案比較の確認（2026 年発売、prereg/confirm2026_plan.md）→ reports/pairwise_confirm2026.md
	$(PYTHON) $(SCRIPTS)/run_confirm.py
	$(PYTHON) $(SCRIPTS)/eval_confirm.py

pairwise-robustness: ## 2 案比較の頑健性（Flash・言い換え・消費者ペルソナ・Claude）→ reports/pairwise_robustness.md
	$(PYTHON) $(SCRIPTS)/run_pairwise_robustness.py
	$(PYTHON) $(SCRIPTS)/eval_pairwise_robustness.py
	$(PYTHON) $(SCRIPTS)/analyze_pairwise_reasons.py


human-eval: ## 関係者による参考比較（30 組、7 名）の採点 → reports/human_eval.md
	$(PYTHON) $(SCRIPTS)/eval_human.py


panel: ## 合成消費者パネルの実行（§5.2）
	$(PYTHON) $(SCRIPTS)/run_panel.py

attributes: ## VLM による解釈可能属性の付与（§5.5）
	$(PYTHON) $(SCRIPTS)/run_attributes.py

evaluate: ## パネル vs ベースラインの評価（§5.3）
	$(PYTHON) $(SCRIPTS)/run_eval.py

contamination: ## 学習データ汚染チェック → reports/contamination_check.md（§5.3、必須）
	$(PYTHON) $(SCRIPTS)/run_contamination_check.py

phase1: baselines panel evaluate contamination attributes ## フェーズ1 一式

# ---------------------------------------------------------------
# フェーズ2：前向き検証・RCT
# ---------------------------------------------------------------
prereg: ## 予測値の凍結と SHA256 生成（§5.6。生成後は変更不可）
	$(PYTHON) $(SCRIPTS)/freeze_predictions.py

rct: ## RCT 割当の生成と分析（§5.7）
	$(PYTHON) $(SCRIPTS)/run_rct.py

phase2: prereg rct ## フェーズ2 一式

# ---------------------------------------------------------------
# フェーズ3：公開・執筆
# ---------------------------------------------------------------
benchmark: ## 公開ベンチマークの生成（is_original == True のみ、§5 フェーズ3）
	$(PYTHON) $(SCRIPTS)/build_benchmark.py

figures: ## paper/figures, paper/tables を生成
	$(PYTHON) $(SCRIPTS)/make_figures.py
	$(PYTHON) $(SCRIPTS)/make_hero.py

phase3: benchmark figures ## フェーズ3 一式

# ---------------------------------------------------------------
# 品質
# ---------------------------------------------------------------
test: ## テーブル整合性テスト等（tests/）
	$(PYTHON) -m pytest tests/ -q

lint: ## 型チェックと lint
	$(PYTHON) -m ruff check src tests scripts
	$(PYTHON) -m mypy src

clean-cache: ## LLM キャッシュ以外の中間生成物を削除（画像キャッシュは残す）
	rm -rf data/processed/*.parquet
	@echo "画像キャッシュと LLM キャッシュは削除していない（再取得コストが高いため）"
