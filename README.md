# Ferro nos cristalizadores de PIA — o sinal antecipa a falha do vitrificado?

Estudo exploratório de séries temporais univariadas sobre a contaminação por ferro nos três
cristalizadores (reatores vitrificados de PIA) de uma planta do processo industrial.

O laboratório mede o ferro no licor de cada reator (`Resultado de Ferro (ppm)`, ~6 amostras por
dia, 2011–2026). Quando o valor passa de um limite fixo de **5 ppm**, o lote pode ser condenado e o
reator parado. A pergunta do estudo:

> O ferro medido consegue **antecipar a falha do vitrificado** (troca do reator, vazamento, quebra
> de agitador/bocal) com **menos alarmes falsos** do que a regra em uso?

Não é uma aplicação: é um estudo com toda a lógica em `utils.py` e a narrativa (carga, análise,
conclusões) em um notebook. Código, comentários e nomes de variáveis estão em português.

---

## Resposta curta

- **A regra vigente (ferro > 5 ppm) não carrega informação**: dispara em 21,6 % das janelas que
  antecedem falha e em 23,2 % de uma janela qualquer — **lift 0,93**. O limiar está no lugar
  errado; a variável não é inútil (max > 20 ppm tem lift 5,2×).
- **O que informa é a comparação entre reatores.** A *margem cross-reator* (mediana diária do
  reator menos a mediana dos três no mesmo dia) tem lift de 4,5× a 5,8× nas falhas.
- **Uma regra composta de duas condições supera a vigente e os modelos de machine learning**,
  medida fora da amostra no mesmo período:

  ```
  ALARME  se   máximo diário de ferro > 20 ppm
          ou  ( mediana diária > 3,0 ppm  E  margem cross-reator > 0,6 ppm )
  ```

- **A antecipação é parcial**: 17 de 36 falhas recebem alarme em até 60 dias, lead mediano de
  3 dias, 8 falhas com uma semana ou mais. O produto honesto é *"antecipação parcial + confirmação
  com muito menos alarme falso"*, não *"predição de falha"*.

---

## Dados

**Os dados não estão no repositório** (são da planta, confidenciais). Para reproduzir, coloque os
arquivos em `data/` com este layout:

| Arquivo | Conteúdo |
|---|---|
| `data/Crystallizer #1.csv`, `#2`, `#3` | Amostras de laboratório: `Labref;Ponto de Amostragem;TIMESTAMP;Resultado de Ferro (ppm)` — `sep=";"`, `decimal="."`, UTF-8 com BOM |
| `data/Vitrificados do PIA - Dados de inspeção.xlsx` | Histórico de inspeção, uma aba por reator (`30-151` = C1, `30-251` = C2, `30-351` = C3). É a **fonte dos eventos de falha** |
| `data/Dados Recentes/Crystallizer #N_Ago2026.csv` | Export recente (layout novo, `Data do Log` no lugar de `TIMESTAMP`), usado só na verificação de Ago/2026 |

Características que condicionam toda a análise:

- **Univariado**: só ferro e timestamp — sem temperatura, pressão, vazão ou histórico de condenação.
- **Não estacionário**: queda monotônica de nível (Kendall τ ≈ −0,29; mediana diária de
  2,45–2,50 ppm em 2011–2015 para 1,80–1,90 em 2021–2026). Um limite fixo de 5 ppm é um evento
  muito mais raro hoje do que em 2015.
- **Amostragem irregular** (5,4 a 10,3 amostras/dia conforme o ano) e **lacunas longas** — paradas
  de reator e paradas de planta inteira.
- **Poucos positivos**: 36 falhas anotadas na série (C1 11, C2 16, C3 9), **83 % delas até 2018**.

---

## Metodologia

O notebook é linear e é executado de cima para baixo. As etapas, na ordem:

1. **Carga e limpeza mínima** — `carregar_crystallizer` + `limpar_excursoes`. Só o valor
   fisicamente impossível (> 1000 ppm, uma única amostra) é descartado. Leituras altas e
   isoladas **ficam na série**: são justamente os falsos positivos que um detector precisa
   aprender a rejeitar. Filtros estatísticos de outlier (IQR, Hampel) são usados **apenas para
   visualização** — em dado tratado por IQR nada acima de 4 ppm sobrevive e a própria regra de
   5 ppm não consegue disparar.
2. **Tabela de eventos a partir da planilha de inspeção** — `carregar_inspecoes` →
   `identificar_paradas_de_planta` → `montar_tabela_eventos`. As datas de inspeção caem dentro
   da lacuna de amostragem (o reator parado não é amostrado), então cada evento é **reancorado na
   última medição antes da parada** (ou na âncora manual da planilha, quando existe). Paradas de
   planta inteira (lacuna simultânea nos três reatores) não contam como falha de equipamento,
   exceto quando há dano físico constatado na abertura.
3. **Baseline** — `avaliar_baseline_lift`, `avaliar_detector_diario`: cada critério é comparado com
   a sua taxa base (2000 janelas aleatórias por reator), não contra zero. Métrica-alvo: **F2**
   (recall pesa mais que precisão), com alarmes agrupados em 15 dias.
4. **Estrutura não supervisionada** — janelas deslizantes de toda a operação, regimes,
   decomposição cross-reator, idade de campanha e perfil de risco. Clusterização e
   IsolationForest foram testados e **descartados com a medição registrada no notebook**: o que
   separavam era a época, não a falha. Ficaram as variáveis novas: margem cross-reator, limite
   móvel (p99 de 365 dias), EWMA/CUSUM como features, histórico de reparos e suporte da janela.
5. **Supervisionado** — Regressão Logística, Random Forest, XGBoost e SVC, com SMOTE e seleção de
   features como ablações, avaliados com **validação cruzada temporal** (janela expansiva).
6. **Validação** — modelo avaliado como detector contra as regras no mesmo período fora da
   amostra, holdout temporal, *leave-one-reactor-out*, bootstrap das falhas, *walk-forward* da
   calibração, lead time e ponto de operação por razão de custo.
7. **Resultados por reator e modo de falha** — o que cada reator responde e quais modos de falha
   o ferro tem fisicamente como enxergar.
8. **Verificação Ago/2026** — teste fora da amostra num evento real reportado pela planta.

---

## Resultados

### A regra composta contra a vigente e os modelos (fora da amostra, mesmo período, 13 falhas)

| Regra | VP | FP | Precisão | Recall | F2 |
|---|---|---|---|---|---|
| **Composta (20 / 3,5 / 0,9)** | 3 | 19 | 0,136 | 0,231 | **0,203** |
| Composta (20 / 3,0 / 0,6) | 3 | 28 | 0,097 | 0,231 | 0,181 |
| Modelo supervisionado (quantil 0,90) | 3 | 48 | 0,059 | 0,231 | 0,146 |
| Vigente (`max > 5 ppm`) | 2 | 64 | 0,030 | 0,154 | 0,085 |

Na base completa (36 falhas), a regra de entrega dá VP 12 / FP 69 / F2 0,267 contra
VP 8 / FP 171 / F2 0,124 da vigente — **+50 % de recall com 60 % menos alarmes falsos**,
ou **1,5 alarme falso por reator por ano contra 3,8**.

### Por que os modelos perdem

- **O rótulo é confundido com o tempo.** Falhas e nível de ferro caem juntos ao longo dos anos sem
  que um cause o outro; qualquer feature correlacionada com o tempo parece preditiva. Recalculada
  dentro de blocos de 3 anos, a AUC da "melhor feature" (mediana da janela) cai de 0,627 para
  **0,507**. Nenhuma feature passa de 0,57 de AUC ajustada.
- **O split único original invertia a prevalência** (80 % de positivos no treino), daí recall 1,00
  em quase todas as ablações. Com CV temporal, a variância entre folds (desvio de F2 0,2–0,5) é da
  ordem de qualquer diferença entre modelos.
- 36 positivos, 84 % numa única época, numa série univariada: o modelo tem mais graus de liberdade
  do que o dado sustenta.

### Uma regra única, não calibrada por reator

Calibrar os limiares por reator parece melhor dentro da amostra (F2 0,323 contra 0,267), mas na
validação *walk-forward* o resultado se inverte (**0,159 contra 0,229**), e o desempenho cai
monotonicamente com o número de parâmetros estimados. Uma busca em grade ampla (1100 combinações
por reator) confirmou que os limiares em uso já estão no platô — os ganhos disponíveis são menores
que a oscilação da própria grade.

### Por reator

| Reator | Falhas | VP | Recall | F2 | Leitura |
|---|---|---|---|---|---|
| C3 | 9 | 5 | 0,556 | 0,439 | As 5 detectadas são falhas de revestimento; as perdidas incluem 3 vazamentos em plug de reparo, que não liberam ferro |
| C1 | 11 | 5 | 0,455 | 0,357 | Responde à margem, não ao nível; **lead mediano de 15 dias** |
| C2 | 16 | 4 | 0,250 | 0,222 | Paradas dominadas por agitador e bocal — modos em que o ferro não tem por que subir |

O escopo real da regra são as **falhas de revestimento/casco**. Restringir o denominador a esses
modos eleva o recall para 0,44–0,47, mas não muda o F2 (não remove nenhum alarme falso).

### Verificação Ago/2026

Num evento real no C3 em 20/08/2026, nenhuma regra disparou antes da parada: o run-up foi normal
(mediana 1,2–2,4 ppm, margem ≤ 0,25), consistente com os modos de falha pós-2019, que liberam
pouco ferro. A regra vigente, no mesmo período, disparou nos dois reatores que não falharam.

---

## Limitações

- Os limiares foram escolhidos olhando as mesmas falhas. Bootstrap: F2 0,256, IC 95 %
  **[0,156; 0,363]** — o piso fica acima da regra vigente, mas o intervalo é largo.
- **Nenhuma regra detecta as 6 falhas posteriores a 2019** (vazamentos em plug, falhas constatadas
  em inspeção): é um limite físico do sinal, não só de poder estatístico.
- As datas e a classificação de modo de falha dos eventos ainda dependem de confirmação da planta
  (`ficha_eventos_para_validacao` gera a ficha para isso).
- Sem custo de alarme falso e de parada não planejada, o ponto de operação é escolhido por F2;
  `sensibilidade_custo` já deixa a escolha parametrizada pela razão de custo.

---

## Estrutura do repositório

```
.
├── Crystallizer #1#2#3.ipynb   # narrativa: carga, análises, validação e resumo executivo
├── utils.py                    # todas as funções do estudo (importado com `from utils import *`)
├── requirements.txt
└── data/                       # não versionado — ver "Dados"
```

O notebook é versionado **sem saídas**; os gráficos e tabelas aparecem ao executá-lo com os dados.

Seções de `utils.py`:

| Seção | Principais funções |
|---|---|
| Configuração | `COLUNA_FE`, `RANDOM_STATE`, `STATS_FUNCS`, `FAIXAS_CAMPANHA` |
| Filtro de Hampel | `hampel` (implementação local) |
| Carga e limpeza | `carregar_crystallizer`, `aplicar_tratamentos` |
| Janelas e eventos | `valores_na_janela`, `adicionar_eventos_ultrapassagem`, `unificar_eventos` |
| Inspeção (fonte dos eventos) | `carregar_inspecoes`, `ajustar_eventos_para_lacunas`, `identificar_paradas_de_planta`, `montar_tabela_eventos` |
| Limpeza de excursões | `classificar_amostras_altas`, `limpar_excursoes` |
| Baseline e detectores | `avaliar_baseline_lift`, `avaliar_detector_diario`, `avaliar_detector_composto`, `comparar_detectores` |
| Gráficos | `plot_crystallizer(s)`, `plot_mm_crystallizer(s)`, `plot_crystallizer_unificado` |
| Estatística | `estatisticas`, `testar_classes_por_janela`, `separabilidade_features` |
| Não supervisionado | `preparar_contexto`, `gerar_janelas_deslizantes`, `clusterizar_regimes`, `calcular_margem_cross_reator`, `perfil_hazard_campanha`, `avaliar_valor_features` |
| Classificação | `extrair_features`, `dividir_dados`, `selecionar_features`, `rodar_classificacao` |
| Detecção de anomalia | `rodar_iforest_janelas` |
| Cartas de controle | `calcular_ewma_rolante`, `otimizar_ewma`, `calcular_cusum_dinamico` |
| Validação e auditoria | `avaliar_cv_temporal`, `treinar_detector_janelas`, `alarmes_regra_composta`, `validar_regra_composta`, `lead_time_regra`, `ficha_eventos_para_validacao`, `sensibilidade_custo`, `testar_tendencia` |
| Escopo por reator | `comparar_escopo_treino`, `calibrar_limiares_por_reator`, `resumo_por_reator` |
| Otimização da regra | `buscar_grade_regra`, `ablacao_ramos_regra`, `validar_calibracao_walkforward`, `comparar_estrategias_calibracao` |
| Modo de falha | `classificar_modo_falha`, `avaliar_regra_por_subconjunto` |

Convenções: a coluna analisada é sempre `COLUNA_FE`; `janelas` são listas de dias e o recorte é
`[ts - dias, ts)`; as funções de gráfico devolvem a `fig` e quem chama decide se mostra.

---

## Como executar

Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

1. Coloque os arquivos de dados em `data/` (ver [Dados](#dados)).
2. Abra `Crystallizer #1#2#3.ipynb` num kernel cujo **diretório de trabalho seja a raiz do
   repositório** — `utils.py` monta `DIR_DATA`/`DIR_OUTPUT` a partir de `os.getcwd()` na
   importação.
3. Execute **de cima para baixo**. O notebook é stateful: várias seções reutilizam variáveis
   globais (`JANELAS`, `threshold`, `df`) e as seções de classificação e anomalia dependem dos
   objetos criados na seção não supervisionada.
4. Depois de editar `utils.py`, **reinicie o kernel** (`from utils import *` não recarrega).

Os caminhos contêm `#` e espaços — use aspas no terminal.
