# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Exploratory data-science study (not an application) on iron contamination in the three Bayer-process
crystallizers ("reatores de PIA") of a Bayer plant. The lab measures iron in each reactor
(`Resultado de Ferro (ppm)`, ~6 samples/day); when it exceeds a fixed limit the batch can be
condemned and the reactor stopped. The question is whether that signal can **predict vessel failure
(`Troca do Reator`, leaks, agitator/nozzle breaks) in advance**, and do it with fewer false alarms
than the incumbent fixed-threshold rule.

Two files carry the work: `utils.py` holds every function, `Crystallizer #1#2#3.ipynb` (~270 cells)
holds the narrative — load, configure, call, look at the output. There is no package and no tests.
Code, comments and variable names are in **Portuguese** — keep that convention.

## Objectives (`aux_files/objetivos.md`) and where they stand

| Etapa | Goal | Status in the notebook |
|---|---|---|
| 1 — Unsupervised | Deep EDA + clustering to find natural operating patterns | Done and re-done. Descriptive stats, Kruskal-Wallis + η², Q-Q, Mann-Whitney, cross-correlation with lags; the event-row clustering was **removed** and replaced by sliding-window regimes + cross-reactor decomposition + campaign hazard. Conclusion: no natural pre-failure regime exists; the value is in the new variables (see "Unsupervised stage"). |
| 2 — Supervised | Train on the failure history, cross-validate, **beat the incumbent univariate fixed-threshold model**, cutting false positives | Answered, and the answer is not the expected one: measured out-of-time on the same test period, the **composite rule wins** (F2 0.203) over the incumbent (0.085) and over the supervised models (0.146). See "Validation stage". The ML route is blocked by 36 positives, 84 % of them pre-2019, on a univariate series. |
| 3 — Handover | Ship the best algorithm in a Bayer-friendly language + make the team autonomous | Not started, but the deliverable is now defined: **the composite rule**, not a model — three conditions on daily statistics, implementable anywhere and auditable by the operators, with the scope declared (proven on C3, undetermined on C1/C2). Nothing is exported yet (`output/` is empty). |

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

### Separating lab error from a real high reading — diagnostic only, no longer a filter

`limpar_excursoes` **discards exactly one class of sample: the extreme**.

1. **Implausibility ceiling** (1000 ppm) — the only removal. The whole dataset has exactly one
   sample above it (20000 ppm, C2 05/02/2019); the second largest value is 999 and the third is 488.
2. **`classificar_amostras_altas`** still runs, but its verdict is a **label, not a cut**: a reading
   above 20 ppm is real if *either* it is **corroborated** (≥2 other samples above 5 ppm within
   ±2 days) *or* it is **followed by a stop** (the next normal sample is more than 2 days later).
   Otherwise it is *isolated* — and it **stays in the series**.

The second half of that OR is what makes the label work: in an abrupt failure the operator stops the
reactor right after the reading, so the excursion never gets a chance to show up in other samples.
C3 23/11/2013 (264 ppm) fails corroboration and is caught only by the stop criterion.

**Why isolated readings are kept** (decided 07/08/2026, `descartar_isoladas=False` is the default;
pass `True` to reproduce the old behaviour): an isolated high reading is an *exceedance that was not
followed by a failure* — i.e. exactly the false positive the model has to learn to reject, and the
hardest one, since it has the largest amplitude. Dropping the 13 of them (C1 5, C2 2, C3 6; 21–37
ppm) did three kinds of damage:

- it deleted the most informative negative examples — the negative class then only contained mild
  exceedances;
- it inflated tail-detector precision by construction: `max > 20 ppm` went from 14 false positives
  to 5 and from 0.222 to 0.375 precision **because the filter had deleted the very cases the
  detector gets wrong**, not because the detector improved;
- at least one "lab error" was informative: C1 20/05/2011, 23.7 ppm, sits **8.5 days before** an
  anchored failure. Keeping it takes `max > 10 ppm` from 6 to 7 true positives (F2 0.181 → 0.200).

Effect of keeping them: 13 extra samples, **+4 `Real=0` events at LC 10 ppm** (25 → 29) and **+1 at
LC 5 ppm** (103 → 104 at the time; the anchored failures are unchanged by the cleaning policy). Robustness to a spurious spike is
supposed to come from robust features (daily median, p75/p90) and the relative features — not from
deleting the sample. The old validation still holds as evidence that the *label* is meaningful:
5 of 16 "real" readings have an inspection within 25 days (31 %) against 1 of 13 "isolated" (8 %).

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
shapes are handled by `_parse_data_inspecao` (0 parse failures over the 100 records).
`SUBSTITUIDO = Sim` means *something* was replaced (often just the agitator shaft or baffle) — an
actual reactor swap is detected from the text (`equipamento novo`, `substituição do reator`,
`pelo spare`, and — added by the 11/08/2026 audit — `substituir o reator`, `substituído o reator`),
which is what `TrocaDoReator` does. The same audit widened `Vazamento` (`poro passante`,
`até a parte metálica`, `infiltraç`) and `Emergencia` (the sheet's own typo `emegência`), and
introduced `CORRECOES_INSPECAO` — two documented manual overrides the regexes cannot make safely:
C3 22/10/2024 marked `Vazamento` because the Bayer deck itself calls it *"Vazamento no plug do
reparo"* (18/10/2024), and C2 15/03/2009 marked `TrocaDoReator` ("o reator ... foi instalado devido
a furo"; the pattern `reator .* instalado` would false-positive on "plug instalado no local do furo
do reator"). Corrected rows carry `Corrigido=True`.

**Anchoring events to the data (`ajustar_eventos_para_lacunas`).** Two layers, in order of
precedence:

1. **Manual anchor from the sheet — the source of truth.** The column
   **"DATA ANCORADA NA SÉRIE DE FERRO"** (normalized header `data ancorada`, column C since the
   11/08/2026 update) holds three kinds of content: **56 datetimes**, **41 `Fora do Período da
   Série`** and **4 `-`** (the last two parse to NaT and fall through to the automatic path).
   **54 of the 56 datetimes match a real measurement within 60 s** — they are sample timestamps
   truncated to the minute. So the loader *matches the anchor to its measurement*
   (`tolerancia_ancora_seg=300`) and anchors **one second after that sample**, so the triggering
   reading falls inside `[ts - dias, ts)`; when the cell has no usable time (`00:00`) it falls
   back to the last measurement up to the end of that day (`AncoraCasouMedicao` distinguishes the
   two). A **typo guard** (`max_desvio_ancora_dias=180`) rejects anchors absurdly far from the
   record's own date, printing a warning and reverting to the automatic anchor
   (`AncoraRejeitada=True`). It currently catches exactly one: **C1 07–08/06/2016 anchored to
   `2018-06-05`** (+728 d) — the time-of-day `10:18` is exactly what the automatic method finds in
   **2016**-06-05, so the year was mistyped. Rows carry `AncoraManual`, `AncoraCasouMedicao`,
   `AncoraRejeitada` and `DataAncoradaManual`. With no column at all, behaviour is unchanged.
2. **Automatic re-anchoring (fallback)** — when a reactor stops, sampling stops with it, so the
   inspection date usually falls inside a sampling gap. The event is re-anchored to the last
   measurement before the gap (`dias_lacuna=2` by default), one second after it
   (`incluir_ultima_medicao=True`). Result: **39 of the 60 in-period events move**, from 0.1 to
   137 days. Without this, the "lookback window" of those events is partly the stop itself.

Two things this interacts with:

- **The 100 ppm cut changes an anchor.** For C3 24/11/2013 the anchor lands at 12:01:30 with the cut
  and at 12:02:02 without it — the difference is exactly the 264 ppm sample that triggered the stop.
  With the cut in place that sample is outside its own event's window.
- **Plant-wide gaps are not reactor stops — but what was found during them can be.** 19 gaps hit
  all three reactors simultaneously (2020-04→05, 2021-08→10, 2023-06→08, 2025-01, 2025-03, …) —
  those are plant/lab shutdowns, detected by intersecting the three reactors' gap lists. 8 in-period
  records fall in one. Since 11/08/2026 the rule is selective: records with **real physical damage
  (`Vazamento`) are rescued** — a hole found at opening existed before the stop, so the last
  operating window is exactly where a warning would be (C3 28/07/2023, hole in the top head,
  anchored back 40 d to 17/06; C3 18/07/2024, leaking plug repair) — while swaps *without* damage
  (preventive/spare, e.g. the C3 08/05/2020 record that would move back 137 days into normal
  operation) stay excluded. Rescued rows carry `DescobertoEmParada=True` because the failure DATE
  is uncertain (it was found at opening), and `TipoFalha` classifies every selected failure as
  `emergência` / `constatada em parada de planta` / `constatada/programada`.

**Anchoring reconciles the deck's date disagreements.** For the C1 bottom-nozzle leak the deck says
19/01/2024 and the sheet 02/02/2024 — C1 sampling stops on **20/01** and only resumes 04/02, so the
anchor confirms the deck (the sheet recorded the opening). For the C3 plug leak (deck 18/10/2024,
sheet 22/10/2024) the anchor lands on 20/10, two days from either source. The 2017 "Substituição da
BV" (deck 26/11/2017) has no sheet record within 45 days and stays open in the validation sheet.

After the audit, merging stops within 7 days: **36 failure stops for modelling** (C1 11, C2 16,
C3 9). The path was 31 → 37 (audit rescued 6) → 36, because the plant's anchor column gives
**C1 04/05/2015 and 13/05/2015 the same anchor** (`2015-05-04 04:07`) — it treats the poro found
at opening and the emergency stop it caused as **one episode**, so the 7-day merge collapses them.
Post-2019 failures stay at 6 (vs 3 before the audit) — all low-iron modes (plug leaks,
found-at-inspection). The merge now **propagates the absorbed record's markers**
(`Emergencia`/`Vazamento`/`TrocaDoReator`/`FerroCitado`, count in `ApontamentosFundidos`);
without that the surviving 2015 event would have lost its "emergência" label just for being the
older of the two.

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

Every function lives in `utils.py` (~3720 lines, 95 functions), imported by the notebook's second
cell with `from utils import *`. Sections, in file order:

| Section | Functions |
|---|---|
| Configuração | `DIR_DATA`, `DIR_OUTPUT`, `COLUNA_FE`, `RANDOM_STATE`, `N_SPLITS`, `TEST_SIZE`, `STATS_FUNCS`, `STATS_FUNCS_SEPARABILIDADE`, `CORES_CLASSE`, `NOMES_CLASSE`, `COLUNAS_REGIME`, `FAIXAS_CAMPANHA` |
| Filtro de Hampel | `hampel`, `ResultadoHampel` |
| Carga e limpeza | `carregar_crystallizer`, `aplicar_tratamentos` |
| Janelas e eventos | `valores_na_janela`, `adicionar_eventos_ultrapassagem`, `carregar_eventos`, `separar_eventos_por_data`, `unificar_eventos` |
| Inspeção (fonte dos eventos) | `carregar_inspecoes`, `ajustar_eventos_para_lacunas`, `identificar_lacunas`, `identificar_paradas_de_planta`, `montar_tabela_eventos`, `eventos_para_notebook` |
| Limpeza de excursões | `classificar_amostras_altas`, `limpar_excursoes` |
| Baseline e detectores | `avaliar_baseline_lift`, `criterios_padrao`, `avaliar_detector_diario`, `_agrupar_alarmes`, `_metricas_alarmes`, `avaliar_detector_composto`, `avaliar_detector_adaptativo`, `comparar_detectores` |
| Gráficos da série | `plot_crystallizer`, `plot_crystallizers`, `plot_mm_crystallizer`, `plot_mm_crystallizers`, `plot_crystallizer_unificado` |
| Estatística | `estatisticas`, `hex_to_rgba`, `add_scatter_regressao`, `plot_violin_classes`, `plot_violin_serie_vs_ultrapassagem`, `testar_classes_por_janela`, `separabilidade_features` |
| Não supervisionado (regimes e cross-reator) | `preparar_contexto`, `construir_janelas`, `gerar_janelas_deslizantes`, `janelas_nas_falhas`, `clusterizar_regimes`, `plotar_regimes_pca`, `serie_diaria`, `calcular_margem_cross_reator`, `posto_entre_reatores`, `baseline_movel`, `avaliar_lift_series`, `campanhas_por_reator`, `idade_campanha`, `faixa_campanha`, `perfil_hazard_campanha`, `avaliar_valor_features`, `perfil_temporal_falhas` |
| Classificação | `_features_contexto`, `extrair_features`, `dividir_dados`, `selecionar_features`, `definir_modelos_e_grids`, `avaliar_cv_com_grid`, `avaliar_teste`, `plotar_resultados`, `rodar_classificacao`, `rodar_classificacao_por_tratamento` |
| Detecção de anomalia | `rodar_iforest_janelas`, `plotar_iforest_janelas` |
| Cartas de controle | `calcular_ewma`, `otimizar_ewma`, `plotar_carta_ewma`, `avaliar_carta_controle`, `calcular_cusum_dinamico`, `otimizar_cusum_dinamico`, `plotar_carta_cusum` |
| Protocolo, validação e auditoria | `calcular_ewma_rolante`, `cortes_temporais`, `modelos_padrao_cv`, `avaliar_cv_temporal`, `metricas_alarmes_periodo`, `treinar_detector_janelas`, `alarmes_regra_composta`, `otimizar_regra_composta`, `validar_regra_composta`, `lead_time_regra`, `auditar_janelas_eventos`, `ficha_eventos_para_validacao`, `custo_esperado`, `sensibilidade_custo`, `testar_tendencia` |
| Escopo por reator | `comparar_escopo_treino`, `calibrar_limiares_por_reator`, `avaliar_regra_calibrada`, `resumo_por_reator` |
| Modo de falha | `REGRAS_MODO_FALHA`, `MODOS_COM_FERRO`, `CORRECOES_MODO_FALHA`, `classificar_modo_falha`, `avaliar_regra_por_subconjunto` |
| Otimização da regra | `GRADE_REGRA_AMPLA`, `REFERENCIAS_REGRA`, `preparar_arrays_regra`, `buscar_grade_regra`, `otimizar_regra_por_reator`, `limiares_da_otimizacao`, `sensibilidade_limiares`, `ablacao_ramos_regra`, `validar_calibracao_walkforward`, `comparar_estrategias_calibracao`, `comparar_limiares_por_reator` |

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
   `first` for the rest) and re-sorts chronologically; `limpar_excursoes(..., descartar_isoladas=False)`
   then removes **only** the sample above the 1000 ppm ceiling. The next section ("Erro de laboratório
   vs medição alta informativa") shows every sample above 20 ppm with its verdict — a label, not a
   cut: the isolated ones stay in the series as false-positive material.
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
3. **Event tables.** `Real=1` comes from the inspection sheet: `carregar_inspecoes` (widened
   regexes + `CORRECOES_INSPECAO`) → `identificar_paradas_de_planta` → `montar_tabela_eventos`
   (gap re-anchoring, selective plant-stop rescue, 7-day merge) → `eventos_para_notebook` per
   reactor. Result: **11 / 16 / 9 failures = 36** (31 before the 11/08/2026 audit; the notebook's
   "O que a auditoria mudou" cell lists the 6 rescued with sources). `Real=0` (false positives)
   are still synthesized by `adicionar_eventos_ultrapassagem`: every sample above `threshold`
   (**10 ppm**, not the operational 5) at least `DIAS_BASELINE`/`INTERVALO_MIN_DIAS` (15 days)
   away from any real event and from any previously accepted candidate. The unified table is
   built by `unificar_eventos` (cross-reactor dedup: drop `Real=0` within 15 days of any
   `Real=1`, then enforce 15-day spacing between `Real=0`), ending at 36 positives + 28
   negatives = 64 rows. The per-reactor cells **append** to the existing event dataframe —
   re-running one without re-running the cell above duplicates the `Real=0` rows. A second event
   table at the **operational 5 ppm limit** (`df_eventos_*_lc5`, 36 positives + 101 negatives =
   137 rows) is built right after the unified table — supervised results should be reported on
   both.
3b. **Baseline.** `avaliar_baseline_lift` (lift vs base rate, 15 and 60 days) and
   `avaliar_detector_diario` (precision/recall/F2 per alarm statistic). This is the reference every
   later result has to beat.
4. **Plotting helpers** — `plot_crystallizer`, `plot_crystallizers`, `plot_mm_crystallizer*`
   (moving average), `plot_crystallizer_unificado`. All take `(df_med, df_eventos, titulo,
   mostrar_falsos)` and are Plotly-based; the statistical plots use matplotlib/seaborn.
5. **Statistics** — descriptive tables, Kruskal-Wallis with η² effect size, Q-Q/normality,
   Mann-Whitney per feature, cross-correlation with lags between the three reactors.
6. **Estrutura não supervisionada** (replaced the old clustering section on 10/08/2026 — see
   "Unsupervised stage" below). `preparar_contexto` builds every auxiliary series at once into
   `CONTEXTO` (margin, rank, rolling limit, EWMA/CUSUM per sample, campaigns/repairs/failures),
   then `gerar_janelas_deslizantes` (~4969 windows) + `janelas_nas_falhas` (31) →
   `clusterizar_regimes` → `perfil_temporal_falhas` + `testar_tendencia` →
   `avaliar_valor_features` → cross-reactor lift → campaign hazard → `comparar_detectores`.
   **Everything downstream depends on this section**: it defines `CONTEXTO`, `df_janelas`,
   `df_janelas_falha`, `margem_cross_reator`, `MAP_MARGEM` and `CAMPANHAS`.
7. **Classification (supervised)** — see below. The four run cells pass `contexto=CONTEXTO`;
   passing `contexto=None` reproduces the old feature set, which is how the gain is measured.
8. **Validação** — the protocol section: `avaliar_cv_temporal` on both event tables (LC 10 and
   LC 5), `treinar_detector_janelas` (model as detector vs the rules on the same test period),
   `validar_regra_composta` (temporal holdout / leave-one-reactor-out / bootstrap),
   `lead_time_regra`, and `sensibilidade_custo`. This is where the project's conclusions live.
9. **Detecção de anomalia** — `rodar_iforest_janelas` over the sliding windows (no labels in the
   fit), evaluated by lift and by the same alarm convention as the baseline.
10. **Control charts** — EWMA now via `calcular_ewma_rolante` (rolling baseline; the cell compares
   it against the old hand-picked baseline) and dynamic CUSUM. Both statistics also feed the
   models as features (`ewma_z`, `cusum_rel`).
11. **Resultados por reator** — `comparar_escopo_treino` (evidence for unified fitting),
   per-reactor lift, `calibrar_limiares_por_reator`, `avaliar_regra_calibrada`, `resumo_por_reator`.
   This is where the per-equipment reading lives; it deliberately has no aggregate row.
12. **Modo de falha** — `classificar_modo_falha` + `avaliar_regra_por_subconjunto`: the effect
   of dropping events iron cannot see (see "Failure mode" below).
13. **Resumo executivo** — closing narrative written for the Bayer team. Its one code cell
   (`fim-tabela`) is deliberately **self-sufficient**: it depends only on the event section and
   recomputes `margem_cross_reator` if missing, because it had been failing on out-of-order kernel
   execution (`df_resumo_reator` was defined ~20 cells earlier). It now also reports false alarms
   per reactor-year, and is built on the **single delivery rule**, matching the scope decision.

### Feature engineering (the core idea)

Every event (real or false) becomes one row. For each lookback window in `JANELAS` (days before the
event timestamp), the samples in `[ts - dias, ts)` are reduced by `STATS_FUNCS`
(`media, mediana, std, max, p75, p90, range`) into columns named `{stat}_{dias}d`.

The **classification** version, `extrair_features`, additionally derives `acel_media_*`,
`vel_diff_*`, `pico_max_*`, `volatilidade_std_*` as short-window-vs-longest-window
ratios/differences, then **deletes all absolute features except those of the shortest window**, so
the model sees mostly relative dynamics.

`_features_contexto` then appends — **after** that deletion, because it doesn't follow the
`{stat}_{dias}d` naming the deletion filter matches — the features that came out of the
unsupervised stage. It shares `_features_auxiliares` with `construir_janelas` on purpose, so the
window detector and the event classifier see the same space:

- cross-reactor: `margem_max`, `margem_media`, `margem_inclinacao`, `posto_medio`, `frac_lider`;
- drift-proof level: `razao_limite_movel` (window peak / own rolling 365-day p99);
- control charts as features: `ewma_z`, `cusum_rel` (value at the anchor, never after it);
- equipment history: `idade_campanha`, `faixa_campanha`, `dias_desde_ultimo_reparo`,
  `n_reparos_na_campanha`, `dias_desde_reparo_vidro`, `n_reparos_vidro_campanha`
  (glass-lining repairs — the failure mechanism itself, read from the structured `ReparoVidro`
  column, which is text `"Sim"`/`"Não"`/empty, not a boolean), `n_falhas_anteriores`;
- window quality: `n_amostras_15d`, `n_amostras_3d`, `maior_lacuna_15d`.

Support features are always added; the rest only when a `contexto` is passed — omitting it
reproduces the old feature set, which is how the gain is measured. 26 → 39 features on events.
The sliding-window dataset carries the same block plus shape/persistence/cadence features
(`assimetria`, `curtose`, `maior_seq_acima_base`, `densidade_relativa`, `n_lacunas_3d`,
`dias_desde_amostra`, `amplitude`) — `COLUNAS_MODELO_JANELA`, 30 columns. `COLUNAS_REGIME`
stays frozen at the original 10 so the regime analysis remains reproducible.

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
- **The per-reactor classification blocks (C1 / C2 / C3) were removed on 10/08/2026** and replaced
  by `comparar_escopo_treino` plus a markdown explaining the scope decision. Only the **unified**
  fit remains, and it passes `map_medicoes_unif` (each event's window comes from its own reactor)
  and `contexto=CONTEXTO`. Do not re-add per-reactor fits: the evidence is in "Scope" above.
- Older analysis cells still select the dataset by commenting/uncommenting a block of
  `df = df_crystallizerN_xxx.copy()` lines. Match that style when adding cells there.
- **64 rows, 36 positives across the three reactors — but C3 alone has 9 (3 of them iron-silent plug leaks).** Treat any single F2
  number as noise; differences between models/treatments need repeated CV or an interval, not one
  split. `avaliar_cv_com_grid` sizes `SMOTE(k_neighbors=...)` from the minority count **per CV
  fold** (`floor(n * (n_splits-1) / n_splits) - 1`), not from the whole training set, and disables
  oversampling with a warning when a fold would be left with fewer than 2 positives. Sizing it from
  the full training set is what made the C3 block crash with
  `Expected n_neighbors <= n_samples_fit`.

## Unsupervised stage — what it found and what was discarded (10/08/2026)

**Discarded, with the measurement that killed it** (the justification is written into the notebook
markdown of each section, keep it there):

- **Clustering the 59 event rows** (`extrair_features_clustering`, `pipeline_clustering`,
  `rodar_clusterizacao`, `plotar_clusterizacao_pca`). It selected algorithm and `k` by ARI **against
  the label** — the ARI it reported (0.217) was the max of 36 configurations (median 0.108). By the
  internal criterion the winner put 98 % of points in one cluster; and what the clusters separated
  was the **era**, not the failure: ARI(cluster, period ≥ 2020) = **0.593** vs ARI(cluster, `Real`)
  = 0.217. 59 rows × 35 features, PC1 = 62 %, PC1+PC2 = 97 %, 38 % of feature pairs |corr| > 0.9.
- **Clustering sliding windows** (kept in the notebook as evidence, not as a model): with log1p +
  `RobustScaler` and `k` by silhouette, the winner (k=2) is degenerate — 99.6 % of windows in one
  cluster with lift **0.97**. Across every `k`, the high-lift cluster is 0.1–0.4 % of the time and
  covers **1 failure of 36** (lift rises 7.5 → 22.4, coverage stays at 2.7 %). On the 36-base the
  `lift_util` column (best lift among clusters covering ≥20 % of failures) stays **below 1.0 at
  every k** — no regime with useful coverage beats chance. (On the 31-base a k=4 regime reached
  lift 3.3; it did not survive the rescued, iron-silent failures.)
- **IsolationForest on event rows** (`otimizar_avaliar_iforest`, `rodar_iforest`,
  `plotar_resultados_iforest`): circular (the event table was already built by selecting
  exceedances) and a 108-combination grid over 59 rows. Rerun over the sliding windows it gets real
  lift (3–9×) but F2 of **0.03–0.08** — an order of magnitude below the composite rule.
- **Binary simultaneity filter** ("ignore an exceedance if another reactor also exceeded"):
  precision *drops* from 0.045 to 0.035. The continuous version (margin) is what carries signal.
- **Leave-one-out margin** (reactor minus the median of the *other two*): worse than the plain
  median of the three (56 vs 44 false positives at equal VP). With three series the median is the
  middle value, so `x_i − median` is positive only for the leading reactor and measures by how much
  it leads — that is the discriminative quantity.

**Kept, with the numbers that justify it:**

| Finding | Measurement |
|---|---|
| **Cross-reactor margin** = daily median − median of the three | lift **4.5×** (>1.2) and **5.8×** (>1.6) at the 36 failure anchors, vs 3.1× for daily median > 5 ppm at equal or lower base rate |
| C1–C2 daily medians correlate **0.76**, C3 only **0.11** (other train); common component is **6 %** of variance | so "the whole plant rose" is the exception, and reactor identity matters |
| **Rolling 365-day p99** as the limit | detector F2 **0.239** vs 0.230 for the fixed 3.5 ppm, with 95 vs 104 false positives, and drift-proof |
| **Campaign age**, exposure-normalised | on the corrected campaigns + 36-base the profile is ~flat for the first 5 years (**2.5–3.1 failures / 1000 reactor-days**) then drops ~4× (**0.72**) — still non-monotonic, so it enters as a **band**, not a linear term |
| **Window support** varies 8 → 150 samples | now a feature (`n_amostras_*`, `maior_lacuna_*`) and a hard filter in the sliding windows (371 of 5340 anchors dropped) |

**The new rule to beat** (`avaliar_detector_composto`, in `comparar_detectores`, 36-base):
`max diário > 20 ppm` **OR** (`mediana diária > 3.0` **AND** `margem > 0.6`) →
**VP 12, FP 69, precision 0.148, recall 0.333, F2 0.267**, against the incumbent `max > 5`
(VP 8, FP 171, precision 0.045, recall 0.222, F2 0.124) — +50 % recall with 60 % fewer false alarms.
Both branches are needed: the median branch misses C3 23/11/2013 (264 ppm in a single sample never
moves a daily median), the max branch misses the sustained elevations.
**These thresholds were chosen looking at the same failures** — it is the starting point for the
supervised stage to validate on the temporal split, not a validated result.

## Validation stage — the results that redirect the project (10/08/2026)

Everything here is measured; the notebook section "Validação" reproduces it.

**1. The label is confounded with time, and that explains most earlier results.**
**30 of the 36 failures (83 %) happen up to 2018**: the rate falls from 4.26 failures per 1000
reactor-days in 2016–2018 to **0.30** in 2019–2021, with a partial return (1.52) in 2022–2024
made of the audit-rescued C3 plug leaks. Iron drifts down over the same period
(Kendall tau ≈ **−0.29**, p ~1e-160 on all three reactors, Sen slope ≈ **−0.05 ppm/year**,
median 2.45–2.50 → 1.80–1.90 ppm). Iron and failures fall together without one causing the
other, so **any feature correlated with time looks predictive without predicting anything**.
`avaliar_valor_features` reports `AUC_ajustada` (AUC recomputed inside 3-year blocks, weighted
by positives) precisely to expose this.

**2. Feature value, before and after adjusting for era** (4969 windows, ~175 positive, 36-base):

| Feature | AUC | AUC adjusted | Reading |
|---|---|---|---|
| `mediana` | 0.627 | **0.507** | the "best feature" was mostly a clock |
| `p90` | 0.606 | **0.517** | idem |
| `razao_max` | 0.375 | **0.451** | the apparent inversion was the drift |
| `dias_desde_ultimo_reparo` | 0.521 | **0.567** | flips and becomes the strongest: recent repair, higher risk |
| `cusum_rel` | 0.552 | **0.556** | stable — the control chart as a feature |
| `margem_media` | 0.546 | **0.544** | stable |
| `idade_campanha` | 0.430 | **0.536** | still flips under adjustment, weaker than on the 31-base |
| `n_reparos_vidro_campanha` | 0.449 | **0.532** | more glass repairs in the campaign, more risk |
| `densidade_relativa` | 0.391 | **0.414** | stable: sampling thins out before a stop |
| `assimetria` / `curtose` | 0.382 / 0.418 | **0.417 / 0.417** | stable: the pre-failure window has a different shape |

No single feature exceeds **0.57** adjusted AUC. Note `margem_max` has adjusted AUC 0.515 but
lift **5.8×** at threshold 1.6 — AUC measures average separation, lift measures the tail; the
margin does not distinguish the ordinary day, it distinguishes the extreme one.

**3. The supervised evaluation was broken and, once fixed, the models lose.** The old single
split cuts on the positives' quantile and **inverts the prevalence** (unified: 24 pos / 6 neg in
train, 7 / 21 in test) — hence recall 1.00 in 11 of 12 ablations with precision equal to the test
base rate. Under expanding-window temporal CV the pattern is prevalence-driven: fold 1 F2 **0.978**
(test block 90 % positive — artefact), later folds swing 0.0–0.87 on 9-to-10-row test blocks,
F2 std 0.2–0.5 — of the order of any difference between models, so model ranking here is noise.
Evaluated as a **detector** on the same out-of-time test period (36-base, 3 folds, 13 failures):

| Rule | VP | FP | precision | recall | F2 |
|---|---|---|---|---|---|
| composite (20 / 3.5 / 0.9) | 3 | 19 | 0.136 | 0.231 | **0.203** |
| composite (20 / 3.0 / 0.6) | 3 | 28 | 0.097 | 0.231 | 0.181 |
| supervised model (quantile 0.90) | 3 | 48 | 0.059 | 0.231 | 0.146 |
| incumbent (`max > 5`) | 2 | 64 | 0.030 | 0.154 | 0.085 |
| supervised model (quantiles 0.95/0.98/0.99) | 0 | 5/0/0 | 0.000 | 0.000 | 0.000 |

**The rule beats the model.** The model matches the composite's recall only at its most permissive
threshold, at 2.5× the false alarms, and vanishes at the selective ones — it cannot put the
pre-failure windows at the top of the ranking. With 36 positives, 84 % of them in one era, and a univariate series, the model has more
degrees of freedom than the data supports.

**4. The composite rule out of sample (36-base).**
- **Temporal holdout hardened, and that is information**: after 01/01/2019 there are now
  **6 failures** (the audit doubled them) and no rule catches any (the incumbent fires 86 false
  alarms in the period). The post-2019 failures are plug leaks and found-at-inspection modes that
  **release little iron** — not just lack of power, but a physical limit of the signal that
  bounds the product's scope.
- **Leave-one-reactor-out says the signal is C3's**: with thresholds fitted on the other two,
  C3 gets recall **0.556** (5 of 9 — the 5 classic lining failures; the 3 rescued plug leaks are
  iron-silent) and F2 **0.357**, against C1 0.182 / 0.179 and C2 0.062 / 0.062. Both
  iron-triggered emergency stops are C3, and C3 is the reactor from the other train.
- **Bootstrap (500 resamples of the failures)**: F2 = 0.256, 95 % CI **[0.156, 0.363]** — the
  lower bound stays above the incumbent's 0.122, but the interval is wide.

**5. Lead time exists, partially.** The composite rule fires for **17 of 36** failures within
60 days, median lead **3.0 days** (p25 1.0, p75 23.0, max 39); 13 failures with ≥1 day and
**8 with ≥7 days**. Better than the 0.0 median of the raw thresholds. The honest product is
*"partial anticipation + cheaper confirmation with fewer false alarms"*, not "failure prediction".

**6. Operating point by cost** (`sensibilidade_custo`): up to ~5:1 (cost of a missed failure over
cost of a false alarm) `max > 20 ppm` wins; at 10:1 the composite 20/3.5/0.9; from 25:1 on, the
composite 20/3.0/0.6. The plant's cost numbers pick the row — this replaces the arbitrary β=2 of F2.

## Scope: what is unified and what is per reactor (decided 10/08/2026, measured)

The working rule of the study is **unified fitting, per-reactor alarm and reporting**. Each row
below was measured, not assumed:

| Stage | Decision | Evidence |
|---|---|---|
| Loading, cleaning, distributional EDA | **unify** | iron distributions are statistically identical: Kruskal-Wallis η² = 0.0008; medians 2.2 / 2.3 / 2.3 ppm; p99 = 4.3 on all three |
| Feature construction | **unify (mandatory)** | the best feature (cross-reactor margin, and `posto`) only exists with the three series together |
| Fitting (model and thresholds) | **unify** | `comparar_escopo_treino` (36-base): "own reactor" wins nowhere; unified wins C1 (lift 3.79) and C2 (1.32), and C3 learns best from the *other two* (3.47 vs 1.15 from itself) |
| Alarm calibration | **~~per reactor~~ → single fixed rule** | in-sample calibration looks better (F2 0.323 vs 0.267) but **fails walk-forward validation** (0.159 vs 0.229) — see "Rule optimisation" below. Report per reactor, calibrate globally |
| Evaluation and reporting | **per reactor, always** | the aggregate hides that the rule works on C3 and fails on C2 |

Reactor identity as a model feature does **not** help (PR-AUC 0.041 with vs 0.047 without).

**Per-reactor performance with calibrated thresholds** (`resumo_por_reator`, 36-failure base; the
totals are VP 14 / FP 59 / F2 **0.323**, against 12 / 69 / 0.267 for a single threshold and
8 / 171 / 0.124 for the incumbent):

| Reactor | Failures | Responds to | Thresholds (max/median/margin) | VP | FP | recall | F2 | Alarmed ≤60 d | Median lead |
|---|---|---|---|---|---|---|---|---|---|
| C1 | 11 | margin | 30 / 2.5 / 0.6 | 5 | 21 | 0.455 | 0.357 | 4/11 | **15.0 d** |
| C2 | 16 | margin | 20 / 3.0 / 0.6 | 4 | 22 | 0.250 | 0.222 | 7/16 | 2.0 d |
| C3 | 9 | margin | 30 / 3.5 / 0.9 | 5 | 16 | **0.556** | **0.439** | 6/9 | 2.0 d |

Per-reactor lift makes the asymmetry concrete: daily median > 3.5 has lift 4.11 on C3 and 1.30 on
C1; margin > 1.2 has lift 11.22 on C1 and 2.26 on C2. **Each reactor responds to a different
statistic**, which is exactly why a single threshold is suboptimal. C3's 4 misses include the 3
audit-rescued plug leaks (2023–2024), which release no measurable iron — the rule's real scope is
**shell-lining failures**, and on those C3 catches 5 of 5 with signal.

**Why C2 probably is not a modelling problem.** Its stop descriptions are dominated by agitator and
nozzle failures (*"quebra dos parafusos da pá superior do Hidro#1"*, *"Dano na Hélice"*,
*"vazamento pela região do selo/mesa"*), not shell-lining failures — iron in the liquor has no
reason to rise in those modes. C2 has **0 iron mentions in 16 stops**; C3 has 2 in 9. The fix is a
**failure-mode classification from the plant** (a column to add to
`ficha_eventos_para_validacao`), not a different algorithm.

## Rule optimisation — the thresholds are already optimal, and calibrating hurts (11/08/2026)

Asked whether the deployed thresholds were optimal, the answer is **yes, within the noise of this
sample** — and the search that proved it also overturned the per-reactor calibration.

**Setup.** The old calibration used a 4×4×4 = 64-combination grid and **two of the three reactors
picked a grid-edge value** (`limite_max = 30`, the ceiling) — an edge optimum is the limit of the
search, not an optimum. `GRADE_REGRA_AMPLA` widens it to 11×10×10 = **1100 combinations per
reactor** and adds `inf` (branch off) / `-inf` (condition always true), so the search itself can
discover a dead branch. The evaluation runs on integer day arrays (`preparar_arrays_regra`,
`_metricas_dias`), reproducing `_metricas_alarmes` exactly (verified VP/FP-identical) at ~1 s for
the whole analysis instead of minutes.

**1. The values in use are optimal within noise.** Gains from the 17× larger grid: C1 **+0.005**,
C2 **+0.021**, C3 **−0.015** (its value in use *is* the argmax). For scale, the 27 grid neighbours
of the operating point span F2 0.21–0.28 — the available gains are smaller than the grid's own
wobble. The global single rule ranks **50th of 1100** (top 4.5 %).

**2. The `max` branch is largely dead weight.** On C1 it never fires (F2 identical at
`limite_max` = 40, 60 or `inf`, because no C1 day exceeds that); the whole rule there is
mediana+margem. Globally, turning the branch off (`inf`) gives the same VP/FP as the best result.
It still earns its place on C3, where the two branches are complementary (2–3 failures each alone,
5 together). **Correction to an earlier claim in this repo:** the `max` branch was justified by
saying the median branch would miss C3 23/11/2013 (264 ppm in one sample; that day's median is
only 2.95 and margin 0.00). That is wrong — the alarm window is 15 days and another day in it
fires the median branch.

**3. The margin is what makes the rule work.** Ablation with calibrated thresholds:

| Reactor | full rule | max branch only | median+margin only | median only (no margin) |
|---|---|---|---|---|
| C1 | **0.362** | 0.000 (never fires) | **0.362** | 0.000 (0 VP, 60 FP) |
| C2 | **0.243** | 0.067 | 0.208 | 0.152 |
| C3 | **0.424** | 0.244 | 0.254 | 0.227 |

**4. Per-reactor calibration does not survive validation — this reverses the earlier
recommendation.** Walk-forward (calibrate on the past of each cut, measure on the next block;
reference rules measured on the *same* test slices):

| Rule | VP | FP | precision | recall | F2 out-of-sample |
|---|---|---|---|---|---|
| **fixed composite (20 / 3.0 / 0.6)** | 9 | 52 | 0.148 | 0.265 | **0.229** |
| simplified (median>2.75 AND margin>0.6) | 8 | 56 | 0.125 | 0.235 | 0.200 |
| **calibrated per reactor on the past** | 6 | 46 | 0.115 | 0.176 | **0.159** |
| margin only (> 0.8) | 5 | 38 | 0.116 | 0.147 | 0.140 |
| incumbent (`max > 5`) | 7 | 138 | 0.048 | 0.206 | 0.124 |

And performance decays **monotonically with the number of estimated parameters**
(`comparar_estrategias_calibracao`, same time cuts for all): 0 params **0.273**, 3 params
0.23–0.24, 9 params 0.213. The calibration's optimism is large — mean training F2 **0.51** against
**0.16** on test (C2: 0.46 → 0.06; C3: 0.67 → 0.21).

**Caveat that must travel with this table:** the "fixed" row has a built-in advantage — 20/3.0/0.6
was itself chosen looking at the whole series. The honest reading is "**no evidence that
calibrating helps, and evidence that it hurts**", not "0.229 is guaranteed".

**Consequence for the deliverable:** ship **one rule for all three reactors**, keep the reporting
per reactor (the C1/C2/C3 asymmetry is real), and do not touch the thresholds with the current
data — the grid has been swept end to end. What unlocks the next gain is more labelled events and
the failure-mode classification, not a better ppm number.

## Failure mode — what iron can physically see (11/08/2026)

Iron only rises when the glass lining breaks and exposes carbon steel to the liquor. Agitator
failures without steel exposure, external leaks through a gasket/seal, and leaks in a tantalum plug
repair (a tiny area) have no reason to move the reading — demanding that the detector anticipate
them is measuring against an impossible denominator. Two cuts were tested with the delivery rule
(20/3.0/0.6) on all 36 failures:

| Denominator | failures | VP | recall | F2 |
|---|---|---|---|---|
| **all** | 36 | 12 | 0.333 | **0.267** |
| emergency stops only | 18 | 8 | **0.444** | 0.268 |
| found in scheduled inspection only | 16 | 4 | 0.250 | 0.146 |
| lining/shell only (text proxy) | 17 | 8 | **0.471** | 0.276 |
| modes where iron is physically plausible | 24 | 9 | 0.375 | 0.259 |

**Recall rises to 0.44–0.47; F2 does not move.** The arithmetic reason matters: restricting the
denominator **removes no false alarm** — the detector fires exactly as often, only the count of
"failures it was supposed to catch" changes. This **corrects an expectation written earlier in this
repo** ("filtering to lining failures makes the measured performance rise") — recall rises, overall
performance does not. Two consequences: the declared scope improves the *promise*, not the product;
and the mode classification is still worth asking the plant for, but because it defines the alarm's
**contractual scope**, not because it will lift F2.

`classificar_modo_falha` reads OCORRIMENTO + OBSERVAÇÕES (what happened) and deliberately **not**
SERVIÇOS EXECUTADOS (what was done) — installing a plug is the *repair* for a hole, not the failure
mode; ignoring that classified 21 of 36 failures as "plug", including the two stops the sheet
attributes to iron. It is a **proxy, not a verdict**: `CORRECOES_MODO_FALHA` carries the two
documented overrides (C3 23/11/2013 and 02/08/2018, whose OCORRIMENTO describes the iron trigger
rather than the damage), and `ModoFalha` now travels in `ficha_eventos_para_validacao` for the
plant to correct.

**Operational load, the number the plant feels:** the delivery rule produces 69 false alarms over
~15 years and 3 reactors = **1.5 per reactor per year**, against 171 = **3.8 per reactor per year**
for the incumbent 5 ppm rule.

## Open gaps — what still needs analysis

1. **The 5 ppm rule carries almost no information — but iron does.** The control that settles it
   (`avaliar_baseline_lift`, in the notebook's "Baseline" section): compare the fraction of failure
   windows meeting a criterion against the same fraction over 2000 random windows per reactor.
   Over the 36 anchored failures, 15-day windows, after `limpar_excursoes`:

   | Criterion | In failure windows | Base rate | Lift |
   |---|---|---|---|
   | max > 5 ppm (incumbent) | 21.6 % | 23.2 % | **0.93×** |
   | max > 7 ppm | 21.6 % | 9.9 % | **2.18×** |
   | max > 10 ppm | 18.9 % | 5.5 % | **3.45×** |
   | max > 20 ppm | 10.8 % | 2.1 % | **5.22×** |
   | daily median ≥ 3.5 ppm | 29.7 % | 16.3 % | **1.82×** |
   | daily median ≥ 4.0 ppm | 18.9 % | 8.2 % | **2.30×** |

   On the audited 36-failure base the incumbent rule falls **below base rate even at 15 days**
   (it was 1.11× on the 31-base — the rescued failures are iron-silent and dilute every level
   criterion). At 60 days it drops to **0.84×**, which is why an apparent "33.5-day median lead
   time" at 5 ppm is an artefact. The incumbent rule does not fail because
   iron is uninformative; it fails because the threshold sits where the signal is buried. Every
   comparison should be against this table, not against zero. (These figures are the ones measured
   after isolated high readings stopped being deleted — the earlier `max > 20 ppm` lift of 10.6×
   was partly the filter's doing, since it had removed most of the tail's false alarms.)
2. **Threshold mismatch — closed.** Both negative classes exist (`df_eventos_*` at 10 ppm,
   `df_eventos_*_lc5` at 5 ppm: 73/67/48 per reactor before dedup, 31+104 unified) and the
   "Validação" section now runs `avaliar_cv_temporal` on **both**, side by side.
3. **Missing data for the business case — machinery ready, data still missing.** The condemnation
   history is still not in the repo, so `custo_esperado` / `sensibilidade_custo` are parameterised
   by the cost ratio instead of absolute money: the moment the plant provides cost of a false
   alarm and of an unplanned stop, the operating point is arithmetic. No process variables exist
   to correlate against Etapa 1's "correlações ocultas" — that remains blocked.
4. **Unused free features — implemented.** Campaign age, window support and cross-reactor margin are
   now in `_features_contexto` (see "Unsupervised stage"). `SelectKBest` puts 5 of them in the top 10
   (`n_amostras_15d`, `n_amostras_3d`, `margem_max_15d`, `razao_limite_movel`, `idade_campanha`),
   which is encouraging but *not* evidence of a better model — that needs the temporal-split result.
   Still unused: reactor identity as a categorical (`dividir_dados(incluir_crystallizer=True)` exists
   but is off), and time since the previous sample at the anchor.
4b. **The alarm statistic is wrong — addressed by the composite rule** (see "Unsupervised stage";
   `comparar_detectores` prints the full table in the notebook).
   `avaliar_detector_diario` over the anchored failures (measured on the 31-failure base;
   the notebook recomputes on 36)
   (alarm = daily statistic above the limit, grouped at 15 days, hit if the failure follows within
   15 days):

   | statistic | limit | days | VP | FP | precision | recall | F2 |
   |---|---|---|---|---|---|---|---|
   | **daily median** | **3.5** | 1 | 11 | 104 | 0.096 | **0.355** | **0.230** |
   | daily median | 4.0 | 1 | 8 | 59 | 0.119 | 0.258 | 0.209 |
   | max | 10 | 1 | 7 | 44 | 0.137 | 0.226 | 0.200 |
   | max | 7 | 1 | 8 | 78 | 0.093 | 0.258 | 0.190 |
   | daily median | 3.5 | 2 | 5 | 31 | 0.139 | 0.161 | 0.156 |
   | max | 20 | 1 | 4 | 14 | **0.222** | 0.129 | 0.141 |
   | max | 5 (incumbent) | 1 | 8 | 171 | 0.045 | 0.258 | 0.132 |

   The daily median beats the incumbent max-based rule on both recall and F2. Requiring 2 consecutive
   days above the limit halves recall — **the signal is an impulse, not a ramp**, which is why the
   deck looked for a "perfil de elevação" and found none. `max > 20 ppm` is still the high-precision
   corner (0.222, ~5× the incumbent's 0.045) and the only rule an operator could act on without alarm
   fatigue — but its precision is 0.222, not the 0.375 measured while isolated spikes were being
   filtered out.
5. **Window quality — closed.** Sliding windows drop anchors with fewer than 8 samples (371 of
   5340) and `n_amostras_*` / `maior_lacuna_15d` are model features. `auditar_janelas_eventos`
   audits the event table itself: over the 36 anchored failures **none** falls below 8 samples
   (median 83, min 15, max 153) — the "2 samples in 15 days" case was the R3 2020 record, which
   the plant-stop filter already removes.
6. **Stationarity — closed, and it matters more than expected.** `testar_tendencia`: Kendall tau
   −0.288/−0.293/−0.298 with p ~1e-157 to 1e-186, Sen slope ≈ −0.05 ppm/year, daily median
   2.45–2.50 (2011–2015) → 1.80–1.90 (2021–2026), Kruskal-Wallis across years H ≈ 2100 per
   reactor. The series is **not** stationary, which is why the moving limit and the relative
   features exist — and why era-adjusted AUC is mandatory when judging a feature. The EWMA
   baseline is no longer hand-picked: `calcular_ewma_rolante` uses a rolling median + robust
   IQR-based scale (ADF is not run: statsmodels is not installed, and the question here is
   monotonic trend, not unit root).
7. **Lead time — answered, and it is partial.** Raw thresholds fire on the very last measurement
   before the stop (median lead **0.0 days** at 7/10/20 ppm on the 31-base analysis) — iron above
   ~7 ppm is the *trigger*, not a precursor; the apparent 33.5-day lead at 5 ppm is base-rate noise.
   The **composite rule** does better: on the 36-base, `lead_time_regra` gives 17/36 alarmed within
   60 days, median lead **3.0 days** (p25 1.0, p75 23.0, max 40), 13 failures with ≥1 day and
   **8 with ≥7 days**. Usable anticipation exists for roughly a fifth of the failures and none for
   the rest — report it as this distribution, never as a mean.
8. **What is genuinely still open.** (a) The temporal holdout cannot be run with 3 post-2019
   failures — either the plant supplies more recent labelled events or the claim stays scoped to
   2011–2018. (b) The C1/C2 vs C3 asymmetry needs a process explanation from the plant (different
   train, different duty?) before the rule is deployed plant-wide. (c) The event dates still need
   the plant's confirmation — `ficha_eventos_para_validacao` produces the sheet to send (60
   records: 36 used as failures, 17 not failures, 6 dropped as plant stops). The sheet's
   "DATA ANCORADA NA SÉRIE DE FERRO" column is already filled and is the authoritative anchor;
   what still needs the plant is the **failure-mode classification** and a fix for the one
   mistyped anchor (C1 2016 → 2018).

## Known traps

- **`hampel` comes from `utils.py`, not from PyPI.** The package would not install, so the filter is
  implemented locally. Do not add `from hampel import hampel`. `window_size` is in **samples**, not
  days — the notebook converts with `dias * amostras_por_dia` (15 d → 75, 90 d → 450, at an assumed
  5 samples/day while the data actually averages ~6).
- **Section cells re-declare `STATS_FUNCS` and shadow the `utils` default.** The separability
  analysis uses a 9-stat version and the classification section re-declares the 7-stat one.
  Functions in `utils.py` fall back to the module-level `STATS_FUNCS` when the argument is omitted,
  so those cells pass `stats_funcs=STATS_FUNCS` explicitly. Keep doing that when adding cells to
  those sections — omitting it silently changes the feature set. (The IsolationForest section used
  to declare a 4-stat version; it no longer uses `STATS_FUNCS` at all — it runs on the sliding
  windows, whose feature set is `COLUNAS_REGIME`.)
- ~~Cells that are still broken~~ — the `valores_serie_normal` violin cells (synthetic-normal
  reference) were **deleted**: fitting a normal to a heavily right-skewed series is not a valid
  reference. The decision is noted in the violin-section markdowns of each crystallizer.
- **Positional row drops.** Cleaning uses hardcoded index labels — `drop([23, 2141])` for C1,
  `drop([6476, 6494])` for C3 (C2 drops by `Labref == 4027521`, which is safe). These depend on the
  current CSV row order; if the CSVs are ever refreshed, those drops silently remove the wrong rows.
  The event cells likewise drop out-of-period events by position (`[0,1,2]` for R2, `[0,1]` for R3).
- **Globals are reassigned across sections.** `JANELAS` alternates between `[15, 7]` and
  `[15, 12, 9, 6, 3]`; `threshold`, `df`, `serie`, `fig` are reused constantly. Out-of-order
  execution produces plausible-but-wrong results. The unsupervised section adds a hard dependency:
  it defines `MAP_MARGEM`, `MAP_BASELINE_MOVEL`, `CAMPANHAS`, `df_janelas` and `df_janelas_falha`,
  which the classification and anomaly sections consume — running those without it raises `NameError`.
- **The plots draw the control limit at 5 ppm** (`add_hline(y=5)`, hardcoded in five plotting
  functions) while events are generated at `threshold = 10`. The line on screen is not the rule
  being evaluated.
- The notebook stores its outputs (base64 plots) inline; prefer targeted edits over rewriting the
  whole file.
