# Iron in the PIA crystallizers — does the signal anticipate glass-lining failure?

Exploratory univariate time series study on iron contamination in the three
crystallizers (glass-lined PIA reactors) of an industrial plant.

The lab measures iron in each reactor's liquor (`Resultado de Ferro (ppm)`, ~6 samples per
day, 2011–2026). When the value exceeds a fixed limit of **5 ppm**, the batch may be condemned and the
reactor stopped. The study's question:

> Can the measured iron **anticipate the failure of the glass lining** (reactor replacement, leak, broken
> agitator/nozzle) with **fewer false alarms** than the rule in use?

This is not an application: it is a study with all the logic in `utils.py` and the narrative (loading, analysis,
conclusions) in a notebook. Code, comments and variable names are in Portuguese.

---

## Short answer

- **The current rule (iron > 5 ppm) carries no information**: it fires in 21.6% of the windows that
  precede a failure and in 23.2% of any random window — **lift 0.93**. The threshold is in the wrong
  place; the variable is not useless (max > 20 ppm has a 5.2× lift).
- **What is informative is the comparison between reactors.** The *cross-reactor margin* (the reactor's daily
  median minus the median of the three on the same day) has a 4.5× to 5.8× lift on failures.
- **A two-condition composite rule beats the current one and the machine learning models**,
  measured out of sample over the same period:

  ```
  ALARM  if   daily maximum iron > 20 ppm
         or  ( daily median > 3.0 ppm  AND  cross-reactor margin > 0.6 ppm )
  ```

- **The anticipation is partial**: 17 of 36 failures get an alarm within 60 days, median lead of
  3 days, 8 failures with a week or more. The honest product is *"partial anticipation + confirmation
  with far fewer false alarms"*, not *"failure prediction"*.

---

## Data

**The data is not in the repository** (it belongs to the plant and is confidential). To reproduce the study, put the
files in `data/` with this layout:

| File | Content |
|---|---|
| `data/Crystallizer #1.csv`, `#2`, `#3` | Lab samples: `Labref;Ponto de Amostragem;TIMESTAMP;Resultado de Ferro (ppm)` — `sep=";"`, `decimal="."`, UTF-8 with BOM |
| `data/Vitrificados do PIA - Dados de inspeção.xlsx` | Inspection history, one sheet per reactor (`30-151` = C1, `30-251` = C2, `30-351` = C3). It is the **source of the failure events** |
| `data/Dados Recentes/Crystallizer #N_Ago2026.csv` | Recent export (new layout, `Data do Log` instead of `TIMESTAMP`), used only for the Aug 2026 check |

Characteristics that shape the whole analysis:

- **Univariate**: only iron and timestamp — no temperature, pressure, flow or condemnation history.
- **Non-stationary**: monotonic drop in level (Kendall τ ≈ −0.29; daily median from
  2.45–2.50 ppm in 2011–2015 to 1.80–1.90 in 2021–2026). A fixed 5 ppm limit is a much
  rarer event today than in 2015.
- **Irregular sampling** (5.4 to 10.3 samples/day depending on the year) and **long gaps** — reactor
  stops and whole-plant shutdowns.
- **Few positives**: 36 failures annotated in the series (C1 11, C2 16, C3 9), **83% of them up to 2018**.

---

## Methodology

The notebook is linear and runs top to bottom. The steps, in order:

1. **Loading and minimal cleaning** — `carregar_crystallizer` + `limpar_excursoes`. Only the
   physically impossible value (> 1000 ppm, a single sample) is discarded. High, isolated readings
   **stay in the series**: they are exactly the false positives a detector needs to
   learn to reject. Statistical outlier filters (IQR, Hampel) are used **only for
   visualization** — on IQR-treated data nothing above 4 ppm survives and the 5 ppm rule itself
   cannot fire.
2. **Event table from the inspection spreadsheet** — `carregar_inspecoes` →
   `identificar_paradas_de_planta` → `montar_tabela_eventos`. The inspection dates fall inside
   the sampling gap (a stopped reactor is not sampled), so each event is **re-anchored on the
   last measurement before the stop** (or on the manual anchor from the spreadsheet, when there is one). Whole-plant
   shutdowns (a simultaneous gap in all three reactors) do not count as equipment failures,
   except when physical damage is found on opening.
3. **Baseline** — `avaliar_baseline_lift`, `avaliar_detector_diario`: each criterion is compared with
   its base rate (2000 random windows per reactor), not against zero. Target metric: **F2**
   (recall weighs more than precision), with alarms grouped over 15 days.
4. **Unsupervised structure** — sliding windows over the whole operation, regimes,
   cross-reactor decomposition, campaign age and risk profile. Clustering and
   IsolationForest were tested and **discarded with the measurement recorded in the notebook**: what
   they separated was the era, not the failure. The new variables were kept: cross-reactor margin, rolling
   limit (365-day p99), EWMA/CUSUM as features, repair history and window support.
5. **Supervised** — Logistic Regression, Random Forest, XGBoost and SVC, with SMOTE and feature
   selection as ablations, evaluated with **temporal cross-validation** (expanding window).
6. **Validation** — model evaluated as a detector against the rules over the same out-of-sample
   period, temporal holdout, *leave-one-reactor-out*, failure bootstrap, *walk-forward* of the
   calibration, lead time and operating point by cost ratio.
7. **Results per reactor and failure mode** — what each reactor responds to and which failure modes
   iron can physically see.
8. **Aug 2026 check** — out-of-sample test on a real event reported by the plant.

---

## Results

### The composite rule versus the current one and the models (out of sample, same period, 13 failures)

| Rule | TP | FP | Precision | Recall | F2 |
|---|---|---|---|---|---|
| **Composite (20 / 3.5 / 0.9)** | 3 | 19 | 0.136 | 0.231 | **0.203** |
| Composite (20 / 3.0 / 0.6) | 3 | 28 | 0.097 | 0.231 | 0.181 |
| Supervised model (0.90 quantile) | 3 | 48 | 0.059 | 0.231 | 0.146 |
| Current (`max > 5 ppm`) | 2 | 64 | 0.030 | 0.154 | 0.085 |

On the full dataset (36 failures), the delivered rule gives TP 12 / FP 69 / F2 0.267 versus
TP 8 / FP 171 / F2 0.124 for the current one — **+50% recall with 60% fewer false alarms**,
or **1.5 false alarms per reactor per year versus 3.8**.

### Why the models lose

- **The label is confounded with time.** Failures and iron level both drop over the years without
  one causing the other; any feature correlated with time looks predictive. Recomputed
  within 3-year blocks, the AUC of the "best feature" (window median) drops from 0.627 to
  **0.507**. No feature exceeds 0.57 adjusted AUC.
- **The original single split inverted the prevalence** (80% positives in training), hence recall 1.00
  in almost every ablation. With temporal CV, the variance between folds (F2 standard deviation 0.2–0.5) is of the
  same order as any difference between models.
- 36 positives, 84% in a single era, in a univariate series: the model has more degrees of freedom
  than the data supports.

### A single rule, not calibrated per reactor

Calibrating the thresholds per reactor looks better in sample (F2 0.323 versus 0.267), but under
*walk-forward* validation the result flips (**0.159 versus 0.229**), and performance drops
monotonically with the number of estimated parameters. A broad grid search (1100 combinations
per reactor) confirmed that the thresholds in use are already on the plateau — the available gains are smaller
than the oscillation of the grid itself.

### Per reactor

| Reactor | Failures | TP | Recall | F2 | Reading |
|---|---|---|---|---|---|
| C3 | 9 | 5 | 0.556 | 0.439 | The 5 detected ones are lining failures; the missed ones include 3 leaks in a repair plug, which do not release iron |
| C1 | 11 | 5 | 0.455 | 0.357 | Responds to the margin, not to the level; **median lead of 15 days** |
| C2 | 16 | 4 | 0.250 | 0.222 | Stops dominated by agitator and nozzle — modes in which iron has no reason to rise |

The real scope of the rule is **lining/shell failures**. Restricting the denominator to those
modes raises recall to 0.44–0.47, but does not change F2 (it does not remove any false alarm).

### Aug 2026 check

On a real event in C3 on 2026-08-20, no rule fired before the stop: the run-up was normal
(median 1.2–2.4 ppm, margin ≤ 0.25), consistent with the post-2019 failure modes, which release
little iron. Over the same period, the current rule fired on the two reactors that did not fail.

---

## Limitations

- The thresholds were chosen by looking at the same failures. Bootstrap: F2 0.256, 95% CI
  **[0.156; 0.363]** — the floor is above the current rule, but the interval is wide.
- **No rule detects the 6 failures after 2019** (plug leaks, failures found
  during inspection): it is a physical limit of the signal, not just of statistical power.
- The dates and failure-mode classification of the events still depend on confirmation by the plant
  (`ficha_eventos_para_validacao` builds the sheet for that).
- Without the cost of a false alarm and of an unplanned stop, the operating point is chosen by F2;
  `sensibilidade_custo` already parameterizes the choice by the cost ratio.

---

## Repository structure

```
.
├── Crystallizer #1#2#3.ipynb   # narrative: loading, analyses, validation and executive summary
├── utils.py                    # every function of the study (imported with `from utils import *`)
├── pyproject.toml              # dependencies (uv)
├── uv.lock
└── data/                       # not versioned — see "Data"
```

The notebook is versioned **without outputs**; the plots and tables appear when you run it with the data.

Sections of `utils.py`:

| Section | Main functions |
|---|---|
| Configuration | `COLUNA_FE`, `RANDOM_STATE`, `STATS_FUNCS`, `FAIXAS_CAMPANHA` |
| Hampel filter | `hampel` (local implementation) |
| Loading and cleaning | `carregar_crystallizer`, `aplicar_tratamentos` |
| Windows and events | `valores_na_janela`, `adicionar_eventos_ultrapassagem`, `unificar_eventos` |
| Inspection (event source) | `carregar_inspecoes`, `ajustar_eventos_para_lacunas`, `identificar_paradas_de_planta`, `montar_tabela_eventos` |
| Excursion cleaning | `classificar_amostras_altas`, `limpar_excursoes` |
| Baseline and detectors | `avaliar_baseline_lift`, `avaliar_detector_diario`, `avaliar_detector_composto`, `comparar_detectores` |
| Plots | `plot_crystallizer(s)`, `plot_mm_crystallizer(s)`, `plot_crystallizer_unificado` |
| Statistics | `estatisticas`, `testar_classes_por_janela`, `separabilidade_features` |
| Unsupervised | `preparar_contexto`, `gerar_janelas_deslizantes`, `clusterizar_regimes`, `calcular_margem_cross_reator`, `perfil_hazard_campanha`, `avaliar_valor_features` |
| Classification | `extrair_features`, `dividir_dados`, `selecionar_features`, `rodar_classificacao` |
| Anomaly detection | `rodar_iforest_janelas` |
| Control charts | `calcular_ewma_rolante`, `otimizar_ewma`, `calcular_cusum_dinamico` |
| Validation and audit | `avaliar_cv_temporal`, `treinar_detector_janelas`, `alarmes_regra_composta`, `validar_regra_composta`, `lead_time_regra`, `ficha_eventos_para_validacao`, `sensibilidade_custo`, `testar_tendencia` |
| Per-reactor scope | `comparar_escopo_treino`, `calibrar_limiares_por_reator`, `resumo_por_reator` |
| Rule optimization | `buscar_grade_regra`, `ablacao_ramos_regra`, `validar_calibracao_walkforward`, `comparar_estrategias_calibracao` |
| Failure mode | `classificar_modo_falha`, `avaliar_regra_por_subconjunto` |

Conventions: the analyzed column is always `COLUNA_FE`; `janelas` are lists of days and the slice is
`[ts - days, ts)`; the plotting functions return the `fig` and the caller decides whether to show it.

---

## How to run

Python 3.12.

```bash
uv sync                            # creates the .venv from pyproject.toml + uv.lock
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
```

1. Put the data files in `data/` (see [Data](#data)).
2. Open `Crystallizer #1#2#3.ipynb` in a kernel whose **working directory is the repository
   root** — `utils.py` builds `DIR_DATA`/`DIR_OUTPUT` from `os.getcwd()` on
   import.
3. Run **top to bottom**. The notebook is stateful: several sections reuse global
   variables (`JANELAS`, `threshold`, `df`) and the classification and anomaly sections depend on the
   objects created in the unsupervised section.
4. After editing `utils.py`, **restart the kernel** (`from utils import *` does not reload).

The paths contain `#` and spaces — quote them in the terminal.
