# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Exploratory data-science study (not an application) on iron contamination in the three Bayer-process
crystallizers ("reatores de PIA") of a Bayer plant. The lab measures iron in each reactor
(`Resultado de Ferro (ppm)`, ~6 samples/day); when it exceeds a fixed limit the batch can be
condemned and the reactor stopped. The question is whether that signal can **predict vessel failure
(`Troca do Reator`, leaks, agitator/nozzle breaks) in advance**, and do it with fewer false alarms
than the incumbent fixed-threshold rule.

Two files carry the work: `utils.py` holds every function, `Crystallizer #1#2#3.ipynb` (~237 cells)
holds the narrative — load, configure, call, look at the output. There is no package and no tests.
Code, comments and variable names are in **Portuguese** — keep that convention.

## Objectives (`aux_files/objetivos.md`) and where they stand

| Etapa | Goal | Status in the notebook |
|---|---|---|
| 1 — Unsupervised | Deep EDA + clustering to find natural operating patterns | Largely done: descriptive stats, Kruskal-Wallis + η², Q-Q, Mann-Whitney per feature, cross-correlation with lags, 5 clustering algorithms. Gaps below. |
| 2 — Supervised | Train on the failure history, cross-validate, **beat the incumbent univariate fixed-threshold model**, cutting false positives | Models are built (LogReg/RF/XGB/SVC + SMOTE/FS ablations, IsolationForest, EWMA, CUSUM), but the incumbent rule is **never measured**, so "beating it" is not yet demonstrated. |
| 3 — Handover | Ship the best algorithm in a Bayer-friendly language + make the team autonomous | Not started. Notebook is exploratory; nothing is exported (`output/` is empty). |

## The incumbent baseline (`aux_files/Histórico de indicação de ferro nos reatores.pptx`)

The Bayer team's own study — this is what the project has to beat, and its data is worth reading
before proposing anything:

- Context: HAZOP discussion triggered by a leak in Reactor #2's glass lining; the study asks
  whether the current safeguard (iron in the lab sample) *works*, and an HCl detector at the
  reactor bottom is on the table as an alternative.
- Method: **7 days** of iron before each event, **fixed limit of 5 ppm** (`CAM-QUAL-000035`), on
  10 selected events (2017–2025).
- Result: **only 1 of the 10** events showed iron above 5 ppm beforehand — 25/05/2020, Reactor #3.
- Survey 2017–2025: **256 exceedance occurrences, 247 without any failure, 9 with** → ~3.5 %
  precision. Per reactor: R1 92, R2 110, R3 54. Slide 10 concludes *"o resultado de ferro não
  garante prever a falha nos reatores"*.
- Timelines: R1 9 events, R2 11, R3 8 — a small subset of the 100 records in the inspection sheet.

Three findings from re-checking that deck against `data/` (do not lose these):

1. **The single "hit" is not even a failure.** The inspection sheet for 08/05 e 12–28/05/2020 (R3)
   reads: *"Parada programada da planta (trem-3), para fazer a substituição do reator vitrificado
   pelo spare (reserva) ... porque estava com 5 plugs de tântalo ... e tempo vida útil.
   Substituição de forma preventiva"*. It was a scheduled preventive swap during a plant shutdown.
   On top of that, the series has *no samples at all* between 22/12/2019 and 24/05/2020 (154 days),
   and the three values Bayer plotted (10.0, 7.0, 5.2 ppm) were taken on 24–25/05 — the first
   samples after that gap. Normal values (~3 ppm) resume on 07/06.
2. **The deck's sample of 10 events omits both cases where iron actually worked** — 23/11/2013 and
   02/08/2018, both R3, both emergency stops the inspection sheet says were *triggered by the iron
   reading*, and both confirmed by a hole found on opening. See "Iron did work" below.
3. **Event dates disagree between the deck and the inspection sheet** — the deck's own timeline
   (slide 2) and its analysis table (slide 5) also disagree with each other:

   | Event | Deck slides 5/charts | Inspection sheet | Deck timeline |
   |---|---|---|---|
   | Substituição da BV (R1) | 26/11/2017 | *no record within 45 days* | 18/11 |
   | Vazamento pelo bocal do fundo (R1) | 19/01/2024 | 02/02/2024 (bocal N, trinca) | 04/05 |
   | Vazamento no plug do reparo (R3) | 18/10/2024 | 22/10/2024 | — |

   None of the three lines up. Every label-dependent result rests on these timestamps — settle them
   with the plant before drawing conclusions.

### Iron did work — twice, and the notebook deletes the evidence

| Sample | Value | What the inspection sheet says |
|---|---|---|
| C3 2013-11-23 12:02 | **264 ppm** | Stop 24/11: *"Parada de emergencia do reator #3, devido a analise de ferro realizado no Cris #3 apresentar valores acima de 200 ppm"* → reactor replaced 30/11 |
| C3 2018-08-02 18:44 | **999 ppm** | Stop 03/08–10/08: *"Parada de emergência ... devido analise de ferro alto"* → *"quebra do revestimento vitríficado seguido de furo no costado ... perda de material com aproximadamente 4 polegadas"* |

The old flat `limite_ppm=100` cut dropped both. That cut is now replaced by `limpar_excursoes`
(the notebook calls `carregar_crystallizer(..., limite_ppm=None)` and then cleans).

There is a third, unlabelled excursion: C3 on 24/09/2013 reads 268 / 366 / 488 ppm after the daily
median climbed 2.5 → 4.8 → 5.3 over 48 h, then the series goes silent for 7 days and returns at
5.9 ppm. No inspection record and no event covers it.

### Separating lab error from a real high reading

`limpar_excursoes` applies two layers instead of a ceiling:

1. **Implausibility ceiling** (1000 ppm). The whole dataset has exactly one sample above it
   (20000 ppm, C2 05/02/2019); the second largest value is 999 and the third is 488.
2. **`classificar_amostras_altas`** — a reading above 20 ppm is real if *either* it is
   **corroborated** (≥2 other samples above 5 ppm within ±2 days) *or* it is **followed by a stop**
   (the next normal sample is more than 2 days later). Otherwise it is an isolated lab error.

The second half of that OR is what makes it work: in an abrupt failure the operator stops the
reactor right after the reading, so the excursion never gets a chance to show up in other samples.
C3 23/11/2013 (264 ppm) is isolated by corroboration alone and is caught only by the stop criterion.

Result: 14 isolated spikes discarded (C1 5, C2 3, C3 6), all four known-real excursions kept.
Validated against the inspection sheet, which the rule never reads: 5 of 16 "real" readings have an
inspection within 25 days (31 %) against 1 of 14 "lab error" (7 %). Removing those spikes also
sharpens the detector — `max > 20 ppm` goes from 14 false positives to 5, precision 0.222 → 0.375.

Two related traps: **replicate `Labref`s can disagree by up to 20×** (44 pairs exist; 4 disagree by
more than 3×, e.g. C3 `1448338` = 2.5 and 24.0) and `aplicar_tratamentos` averages them, producing
a value that was never measured. And 999 looks like it could be a saturation code, but it appears
once and is backed by a confirmed hole, so it is treated as real.

## Environment

`.venv/` is a uv-created venv (CPython 3.12.3); the notebook kernel is named
`bayer-cristalizadores (3.12.3)`. There is **no** `requirements.txt` / `pyproject.toml` — deps were
installed ad hoc. No Jupyter server or nbconvert is installed either; the notebook is meant to be
run from an IDE kernel (VS Code).

```powershell
.\.venv\Scripts\Activate.ps1
uv pip install <package>          # how deps get added here
uv pip list                       # what is actually available
```

Installed stack: pandas, numpy, scipy, scikit-learn, imbalanced-learn (SMOTE), xgboost,
matplotlib, seaborn, plotly, ipykernel.

`DIR_DATA` / `DIR_OUTPUT` are built from `os.getcwd()` **in `utils.py`, at import time**, so the
kernel's working directory must be the repo root — that is also what puts `utils.py` on the import
path. `output/` is currently empty: all export cells (`fig.write_html(...)`, `df.to_csv(...)`) are
commented out, and when uncommented they write to the cwd, not to `DIR_OUTPUT`.

Paths contain `#` and spaces (`data/Crystallizer #1.csv`) — always quote them in shell commands.

## Data (`data/`, `aux_files/`)

| File | Contents |
|---|---|
| `Crystallizer #1.csv`, `#2`, `#3` | Lab samples: `Labref;Ponto de Amostragem;TIMESTAMP;Resultado de Ferro (ppm)`, 2011-03-21 → 2026-03-02 |
| `Vitrificados do PIA - Dados de inspeção.xlsx` | **The event source.** One sheet per reactor: `30-151`=C1, `30-251`=C2, `30-351`=C3. Loaded by `carregar_inspecoes()`. |
| `Historico_Ferro_Geral.csv` | Plant-wide export — verified to be exactly the concatenation of the three sample files (same 3 sampling points, no extra variables). Nothing to gain from it. |
| `analise-equipe-bayer.pptx` | Bayer's own study (see above) plus a final slide with the FuturAI three-stage scope. Text and chart data can be pulled straight from the OOXML: slides are `ppt/slides/slideN.xml`, the plotted series are cached in `ppt/charts/chartN.xml` |

All CSVs: `sep=";"`, `decimal="."`, UTF-8 with BOM. Reading the `.xlsx` needs `openpyxl` (installed).

### Building the event table from the inspection sheets

Columns: `ANO, DATA, TIPO DE INSPEÇÃO, SUBSTITUIDO, REPARO (VIDRO), REPARO (OUTROS), OCORRIMENTO,
OBSERVAÇÕES, SERVIÇOS EXECUTADOS` — **the header text differs between sheets** (`TIPO`/`TIPOS`,
`OBSERVAÇÃO`/`OBSERVAÇÕES`), so `carregar_inspecoes()` normalizes them. 100 records, 2003-2025;
**60 fall inside the iron series**.

`DATA` is not a clean date: it mixes real datetimes with free text ranges — `07 - 08/01/2009`,
`29/05 - 05/06/2011`, `26/06 - 03/07/2013` (crosses months), `08/05 e 12 - 28/05/2020`. All four
shapes are handled by `_parse_data_inspecao`. `SUBSTITUIDO = Sim` means *something* was replaced
(often just the agitator shaft or baffle) — an actual reactor swap is detected from the text
(`equipamento novo`, `substituição do reator`, `pelo spare`, …), which is what `TrocaDoReator` does.

**Anchoring events to the data (`ajustar_eventos_para_lacunas`).** When a reactor stops, sampling
stops with it, so the inspection date usually falls inside a sampling gap. The event is re-anchored
to the last measurement before the gap (`dias_lacuna=2` by default), one second after it so that the
triggering sample lands inside the `[ts - dias, ts)` window (`incluir_ultima_medicao=True`). Result:
**39 of the 60 in-period events move**, from 0.1 to 137 days. Without this, the "lookback window"
of those events is partly the stop itself.

Two things this interacts with:

- **The 100 ppm cut changes an anchor.** For C3 24/11/2013 the anchor lands at 12:01:30 with the cut
  and at 12:02:02 without it — the difference is exactly the 264 ppm sample that triggered the stop.
  With the cut in place that sample is outside its own event's window.
- **Plant-wide gaps are not reactor stops.** 19 gaps hit all three reactors simultaneously
  (2020-04→05, 2021-08→10, 2023-06→08, 2025-01, 2025-03, …) — those are plant/lab shutdowns.
  8 of the 39 shifted events fall in one, and re-anchoring them produces a window that describes
  normal operation weeks before a scheduled outage (the C3 08/05/2020 record moves back 137 days).
  Detect them by intersecting the gap lists of the three reactors and handle them separately.

After dropping plant-shutdown events and merging stops within 7 days: **31 failure stops for
modelling** (C1 11, C2 14, C3 6), of which 17 are emergency/unplanned and 3 mention iron.

### Data facts that change how results should be read

- **The dataset is univariate.** Iron ppm and a timestamp, nothing else — no temperature, pressure,
  campaign data or condemnation records. Etapa 1's "correlações ocultas" can only mean
  cross-reactor and temporal structure, and any large gain over the incumbent rule will probably
  need either extra process tags or features derived from the event history (e.g. time since the
  last `Troca do Reator`).
- **Strong downward drift.** Median iron per year: ~2.5–2.8 ppm (2011–2019) → 2.0 (2020) →
  **1.8–1.9 ppm (2021–2026)**; p95 drops from ~3.9 to ~2.8. A *fixed* 5 ppm limit is therefore a
  much rarer event today than in 2015. This is why the notebook keeps splitting analyses at
  01/04/2020, and why the relative (ratio/difference) features matter more than absolute ones.
  It also means the temporal train/test split straddles a regime change.
- **Sampling rate is not constant**: 5.4 to 10.3 samples/day depending on the year (median interval
  2.2–2.5 h). Hampel windows expressed in samples (`dias * amostras_por_dia`) therefore cover
  different numbers of days across the series.
- **Long gaps exist.** Plant-wide: 2020-04→05 (~50 d), 2021-08→10 (~43 d), 2023-06→08 (~50 d).
  C3 alone: 2018-08→2019-01 (148 d) and 2019-12→2020-05 (154 d). 120–146 gaps > 3 days per reactor.
- **Values are quantized to 0.1 ppm** (~180 distinct values), include small negatives
  (15/18/21 per reactor) and zeros. p99 ≈ 4.3 ppm, so 5 ppm sits around the 99.5th percentile.
- **Only 22 usable positives.** Of the 27 events, 5 (R2 2004/2007/2009, R3 2003/2008) predate the
  series and are dropped by index in the event cells. The unified table ends with 22 `Real=1` and
  32 synthesized `Real=0`.

## Code layout: `utils.py`

Every function lives in `utils.py` (~2000 lines, 43 functions), imported by the notebook's second
cell with `from utils import *`. Sections, in file order:

| Section | Functions |
|---|---|
| Configuração | `DIR_DATA`, `DIR_OUTPUT`, `COLUNA_FE`, `RANDOM_STATE`, `N_SPLITS`, `TEST_SIZE`, `STATS_FUNCS`, `STATS_FUNCS_CLUSTERING`, `STATS_FUNCS_SEPARABILIDADE`, `CORES_CLASSE`, `NOMES_CLASSE` |
| Filtro de Hampel | `hampel`, `ResultadoHampel` |
| Carga e limpeza | `carregar_crystallizer`, `aplicar_tratamentos` |
| Janelas e eventos | `valores_na_janela`, `adicionar_eventos_ultrapassagem`, `carregar_eventos`, `separar_eventos_por_data`, `unificar_eventos` |
| Inspeção (fonte dos eventos) | `carregar_inspecoes`, `ajustar_eventos_para_lacunas`, `identificar_lacunas`, `identificar_paradas_de_planta`, `montar_tabela_eventos`, `eventos_para_notebook` |
| Limpeza de excursões | `classificar_amostras_altas`, `limpar_excursoes` |
| Baseline | `avaliar_baseline_lift`, `criterios_padrao`, `avaliar_detector_diario` |
| Gráficos da série | `plot_crystallizer`, `plot_crystallizers`, `plot_mm_crystallizer`, `plot_mm_crystallizers`, `plot_crystallizer_unificado` |
| Estatística | `estatisticas`, `hex_to_rgba`, `add_scatter_regressao`, `plot_violin_classes`, `plot_violin_serie_vs_ultrapassagem`, `testar_classes_por_janela`, `separabilidade_features` |
| Clusterização | `extrair_features_clustering`, `pipeline_clustering`, `rodar_clusterizacao`, `plotar_clusterizacao_pca` |
| Classificação | `extrair_features`, `dividir_dados`, `selecionar_features`, `definir_modelos_e_grids`, `avaliar_cv_com_grid`, `avaliar_teste`, `plotar_resultados`, `rodar_classificacao`, `rodar_classificacao_por_tratamento` |
| IsolationForest | `otimizar_avaliar_iforest`, `plotar_resultados_iforest`, `rodar_iforest` |
| Cartas de controle | `calcular_ewma`, `otimizar_ewma`, `plotar_carta_ewma`, `avaliar_carta_controle`, `calcular_cusum_dinamico`, `otimizar_cusum_dinamico`, `plotar_carta_cusum` |

Conventions the module follows: the analysed column is always `COLUNA_FE`; `janelas` is a list of
days and the slice is `[ts - dias, ts)`; `metodos` is a list of `{"titulo", "df"}` (one subplot per
treatment); plotting functions **return** the `fig` and the caller decides whether to `.show()`;
the `rodar_*` functions run a whole section and return their results.

Editing `utils.py` requires **restarting the kernel** — `from utils import *` will not pick up
changes otherwise.

## Notebook architecture

The notebook is strictly linear and stateful — run top to bottom. Section order:

1. **Loading + cleaning** (one cell per crystallizer). `carregar_crystallizer(..., limite_ppm=None)`
   parses `TIMESTAMP`, coerces ppm to numeric, deduplicates by `Labref` (mean for numeric columns,
   `first` for the rest) and re-sorts chronologically; `limpar_excursoes` then removes lab errors
   while keeping real excursions. The next section ("Erro de laboratório vs medição alta
   informativa") shows every sample above 20 ppm with the verdict.
2. **Outlier treatments** — `aplicar_tratamentos` returns a `tratamentos_crystallizerN` dict plus an
   `info_crystallizerN` dict (IQR limits, Hampel outlier counts); each crystallizer fans out into
   parallel dataframes that the rest of the notebook compares side by side:
   - `df_crystallizerN` — Original
   - `df_crystallizerN_0a10` — clipped to `[0, 10]` ppm
   - `df_crystallizerN_iqr` — Tukey 1.5×IQR
   - `df_crystallizerN_hampel` — Hampel filter, `window_size = dias(15) * amostras_por_dia(5)`, `n_sigma=3`
   - `df_crystallizerN_hampel90d` — 90-day window, used only as the "full series" baseline in violin plots

   The three reactors are then concatenated into `df_crystallizer123[_treatment]` with a
   `Crystallizer` column (`C1`/`C2`/`C3`).

   **Two-lane treatment policy** (documented in the notebook's "Política de tratamento de dados"
   markdown, right after "# Carregando dados"): the treatments are **visualization-only** (lane B).
   Everything event- or model-facing (lane A) runs on `Original` = raw + `limpar_excursoes`,
   because every statistical filter deletes the failure-trigger readings (264/999 ppm) and on IQR
   data nothing above 4.0 ppm survives — the 5 ppm rule can't even fire. Consequently `CONFIGS`,
   `configs_crystallizer`, `MAP_MEDICOES` and `CONFIGS_UNIFICADO` now hold **only `"Original"`**,
   and `TRATAMENTO_ALVO = "Original"` in the classification cells. Do not re-add treatments to
   those dicts; the treatment-comparison EDA cells (descriptive stats, per-treatment effect size,
   violins) are kept deliberately as the evidence for this decision.
3. **Event tables.** `Real=1` comes from the inspection sheet: `carregar_inspecoes` →
   `identificar_paradas_de_planta` → `montar_tabela_eventos` (gap re-anchoring, plant-stop exclusion,
   7-day merge) → `eventos_para_notebook` per reactor. Result: **11 / 14 / 6 failures**.
   `Real=0` (false positives) are still synthesized by `adicionar_eventos_ultrapassagem`: every
   sample above `threshold` (**10 ppm**, not the operational 5) at least
   `DIAS_BASELINE`/`INTERVALO_MIN_DIAS` (15 days) away from any real event and from any previously
   accepted candidate. The unified table is built by `unificar_eventos` (cross-reactor dedup:
   drop `Real=0` within 15 days of any `Real=1`, then enforce 15-day spacing between `Real=0`),
   ending at 31 positives + 25 negatives = 56 rows. The per-reactor cells **append** to the
   existing event dataframe — re-running one without re-running the cell above duplicates the
   `Real=0` rows. A second event table at the **operational 5 ppm limit** (`df_eventos_*_lc5`,
   31 positives + 103 negatives = 134 rows) is built right after the unified table — supervised
   results should be reported on both.
3b. **Baseline.** `avaliar_baseline_lift` (lift vs base rate, 15 and 60 days) and
   `avaliar_detector_diario` (precision/recall/F2 per alarm statistic). This is the reference every
   later result has to beat.
4. **Plotting helpers** — `plot_crystallizer`, `plot_crystallizers`, `plot_mm_crystallizer*`
   (moving average), `plot_crystallizer_unificado`. All take `(df_med, df_eventos, titulo,
   mostrar_falsos)` and are Plotly-based; the statistical plots use matplotlib/seaborn.
5. **Statistics** — descriptive tables, Kruskal-Wallis with η² effect size, Q-Q/normality,
   Mann-Whitney per feature, cross-correlation with lags between the three reactors.
6. **Clustering (unsupervised)** — `extrair_features_clustering` + `pipeline_clustering` over
   KMeans / Bisecting KMeans / Agglomerative / Spectral / GMM.
7. **Classification (supervised)** — see below.
8. **IsolationForest** anomaly detection.
9. **Control charts** — EWMA (`calcular_ewma` + `otimizar_ewma`) and dynamic CUSUM
   (`calcular_cusum_dinamico` + `otimizar_cusum_dinamico`), both grid-searched on F2.

### Feature engineering (the core idea)

Every event (real or false) becomes one row. For each lookback window in `JANELAS` (days before the
event timestamp), the samples in `[ts - dias, ts)` are reduced by `STATS_FUNCS`
(`media, mediana, std, max, p75, p90, range`) into columns named `{stat}_{dias}d`.

The **classification** version, `extrair_features`, additionally derives `acel_media_*`,
`vel_diff_*`, `pico_max_*`, `volatilidade_std_*` as short-window-vs-longest-window
ratios/differences, then **deletes all absolute features except those of the shortest window**, so
the model sees mostly relative dynamics. The **clustering** version is `extrair_features_clustering`
and keeps the absolute statistics — the two used to share a name and shadow each other.

### Classification specifics

- `JANELAS = [15, 12, 9, 6, 3]`, `N_SPLITS=5`, `RANDOM_STATE=42`, `TEST_SIZE=0.2`.
- `dividir_dados` is a **temporal** split (sorted by `TIMESTAMP_Evento`, cut point taken from the
  positives), not a random one — do not swap it for `train_test_split`.
- `selecionar_features`: `SelectKBest(f_classif)` then drops features correlated above 0.85.
- `definir_modelos_e_grids`: LogisticRegression, RandomForest, XGBoost, SVC (RBF), all class-balanced,
  tuned with `GridSearchCV`.
- Four ablations are compared per run: base / SMOTE / feature selection / both.
- Decision threshold is hardcoded at `0.4` in `avaliar_teste`.
- **F2-score is the target metric** everywhere (rare events, recall matters more than precision).
- Which reactor/treatment a section runs on is chosen by editing `MODO` and `TRATAMENTO_ALVO` at the
  top of the cell; many older analysis cells instead select the dataset by commenting/uncommenting a
  block of `df = df_crystallizerN_xxx.copy()` lines. Match that style when adding cells.
- **68 rows, 31 positives across the three reactors — but C3 alone has 6.** Treat any single F2
  number as noise; differences between models/treatments need repeated CV or an interval, not one
  split. `avaliar_cv_com_grid` sizes `SMOTE(k_neighbors=...)` from the minority count **per CV
  fold** (`floor(n * (n_splits-1) / n_splits) - 1`), not from the whole training set, and disables
  oversampling with a warning when a fold would be left with fewer than 2 positives. Sizing it from
  the full training set is what made the C3 block crash with
  `Expected n_neighbors <= n_samples_fit`.

## Open gaps — what still needs analysis

1. **The 5 ppm rule carries almost no information — but iron does.** The control that settles it
   (`avaliar_baseline_lift`, in the notebook's "Baseline" section): compare the fraction of failure
   windows meeting a criterion against the same fraction over 2000 random windows per reactor.
   Over the 31 anchored failures, 15-day windows, after `limpar_excursoes`:

   | Criterion | In failure windows | Base rate | Lift |
   |---|---|---|---|
   | max > 5 ppm (incumbent) | 25.8 % | 22.5 % | **1.15×** |
   | max > 7 ppm | 22.6 % | 9.2 % | **2.45×** |
   | max > 10 ppm | 19.4 % | 4.3 % | **4.46×** |
   | max > 20 ppm | 9.7 % | 0.9 % | **10.61×** |
   | daily median ≥ 3.5 ppm | 35.5 % | 16.2 % | **2.19×** |
   | daily median ≥ 4.0 ppm | 22.6 % | 8.1 % | **2.78×** |

   At 60 days the incumbent rule drops to **0.89×** — *below* base rate, which is why an apparent
   "33.5-day median lead time" at 5 ppm is an artefact. The incumbent rule does not fail because
   iron is uninformative; it fails because the threshold sits where the signal is buried. Every
   comparison should be against this table, not against zero.
2. **Threshold mismatch — addressed.** `threshold = 10` still drives the default `Real=0` events,
   but the notebook now also builds the negative class at the operational 5 ppm limit
   (`df_eventos_*_lc5`: 72/67/46 per reactor before dedup, 31+103 unified). What remains open is
   actually *using* both tables in the supervised sections and reporting results side by side.
3. **Missing data for the business case.** The condemnation history that Etapa 2 mentions
   ("condenações indevidas") is not in the repo, so financial loss cannot be quantified, and no
   process variables exist to correlate against Etapa 1's "correlações ocultas".
4. **Unused free features.** Reactor campaign age (days since the last swap, now derivable from the
   inspection sheets), reactor identity, time since the previous sample, and sampling density in the
   window are all derivable from what is already loaded, and none are in `STATS_FUNCS`. Campaign age
   looks informative: 26 of 28 failure stops happened after more than a year of campaign, median
   1207 days.
4b. **The alarm statistic is wrong.** `avaliar_detector_diario` over the 31 anchored failures
   (alarm = daily statistic above the limit, grouped at 15 days, hit if the failure follows within
   15 days):

   | statistic | limit | days | VP | FP | precision | recall | F2 |
   |---|---|---|---|---|---|---|---|
   | **daily median** | **3.5** | 1 | 11 | 103 | 0.096 | **0.355** | **0.231** |
   | daily median | 4.0 | 1 | 8 | 58 | 0.121 | 0.258 | 0.211 |
   | max | 10 | 1 | 6 | 36 | 0.143 | 0.194 | 0.181 |
   | max | 5 (incumbent) | 1 | 8 | 168 | 0.045 | 0.258 | 0.133 |
   | max | 20 | 1 | 3 | 5 | **0.375** | 0.097 | 0.114 |
   | daily median | 3.5 | 2 | 5 | 31 | 0.139 | 0.161 | 0.156 |

   The daily median beats the incumbent max-based rule on both recall and F2. Requiring 2 consecutive
   days above the limit halves recall — **the signal is an impulse, not a ramp**, which is why the
   deck looked for a "perfil de elevação" and found none. `max > 20 ppm` is the high-precision
   corner (0.375) and is what an operator could act on without alarm fatigue.
5. **Window quality is never audited.** Events differ wildly in support: 2 samples in 15 days
   (R3 2020) vs 154 (R3 2013). Events whose window contains a multi-day gap should be flagged
   rather than silently averaged.
6. **Stationarity is assumed, not tested.** The drift above is visible but never quantified
   (no per-year distribution test, no ADF/Mann-Kendall), and the EWMA baseline periods are
   hand-picked.
7. **Lead time is the number that decides the project, and with correct anchors it is zero.** First
   crossing within the 60 days before each anchored failure: 5 ppm fires for 15/31 with a median
   lead of 33.5 days (but see the base rate above — that lead is noise), while 7 ppm fires for 9/31,
   10 ppm for 7/31 and 20 ppm for 4/31, all with a median lead of **0.0 days**. The confident
   thresholds fire on the very last measurement before the stop: iron above ~7 ppm is the *trigger*
   the operator acts on, not a precursor. Whether any usable lead time exists is the question
   Etapa 2 must answer, and it should be reported as a distribution, not a mean.

## Known traps

- **`hampel` comes from `utils.py`, not from PyPI.** The package would not install, so the filter is
  implemented locally. Do not add `from hampel import hampel`. `window_size` is in **samples**, not
  days — the notebook converts with `dias * amostras_por_dia` (15 d → 75, 90 d → 450, at an assumed
  5 samples/day while the data actually averages ~6).
- **Section cells re-declare `STATS_FUNCS` and shadow the `utils` default.** The IsolationForest
  section uses a 4-stat version and the separability analysis a 9-stat one. Functions in `utils.py`
  fall back to the module-level `STATS_FUNCS` (7 stats) when the argument is omitted, so those cells
  pass `stats_funcs=STATS_FUNCS` explicitly. Keep doing that when adding cells to those sections —
  omitting it silently changes the feature set.
- ~~Cells that are still broken~~ — the `valores_serie_normal` violin cells (synthetic-normal
  reference) were **deleted**: fitting a normal to a heavily right-skewed series is not a valid
  reference. The decision is noted in the violin-section markdowns of each crystallizer.
- **Positional row drops.** Cleaning uses hardcoded index labels — `drop([23, 2141])` for C1,
  `drop([6476, 6494])` for C3 (C2 drops by `Labref == 4027521`, which is safe). These depend on the
  current CSV row order; if the CSVs are ever refreshed, those drops silently remove the wrong rows.
  The event cells likewise drop out-of-period events by position (`[0,1,2]` for R2, `[0,1]` for R3).
- **Globals are reassigned across sections.** `JANELAS` alternates between `[15, 7]`,
  `[15, 12, 9, 6, 3]` and `[6, 3, 1]`; `threshold`, `df`, `serie`, `fig` are reused constantly.
  Out-of-order execution produces plausible-but-wrong results.
- **The plots draw the control limit at 5 ppm** (`add_hline(y=5)`, hardcoded in five plotting
  functions) while events are generated at `threshold = 10`. The line on screen is not the rule
  being evaluated.
- The notebook stores its outputs (base64 plots) inline; prefer targeted edits over rewriting the
  whole file.
