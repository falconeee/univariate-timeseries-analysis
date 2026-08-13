"""Funções auxiliares do estudo de contaminação por ferro nos crystallizers.

Concentra tudo o que o notebook `Crystallizer #1#2#3.ipynb` repetia célula a célula:
carga e limpeza, tratamentos de outlier, recortes por janela, gráficos e os
análise não supervisionada (regimes, cross-reator), classificação, detecção de
anomalia e cartas de controle.

Uso no notebook:

    from utils import *

Convenções mantidas do notebook:
  - a coluna analisada é sempre "Resultado de Ferro (ppm)";
  - `janelas` são listas de dias (ex.: [15, 12, 9, 6, 3]) e o recorte é `[ts - dias, ts)`;
  - `metodos` é uma lista de {"titulo": str, "df": DataFrame} — um subplot por tratamento;
  - as funções de gráfico devolvem a `fig` (quem chama decide se dá `.show()`).
"""

import os
import re
import datetime as dt
import itertools
import unicodedata

import numpy as np
import pandas as pd
import openpyxl
from numpy.lib.stride_tricks import sliding_window_view

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.colors as pc
from plotly.subplots import make_subplots

from scipy.stats import (skew, kurtosis, mannwhitneyu, ks_2samp, linregress,
                         kendalltau, kruskal)

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import (confusion_matrix, classification_report,
                             make_scorer, f1_score, precision_score,
                             recall_score, precision_recall_curve, average_precision_score,
                             roc_curve, auc, fbeta_score, silhouette_score)
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.feature_selection import SelectKBest, f_classif

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier


# =============================================================================
# Configuração
# =============================================================================

DIR_DATA   = os.getcwd() + "/data/"
DIR_OUTPUT = os.getcwd() + "/output/"

COLUNA_FE = "Resultado de Ferro (ppm)"

RANDOM_STATE = 42
N_SPLITS     = 5
TEST_SIZE    = 0.2

CORES_CLASSE = {0: "#4878CF", 1: "#D65F5F"}
NOMES_CLASSE = {0: "Falso Positivo", 1: "Real"}

# Estatísticas usadas na classificação e no IsolationForest
STATS_FUNCS = {
    'media':   np.mean,
    'mediana': np.median,
    'std':     np.std,
    'max':     np.max,
    'p75':     lambda x: np.percentile(x, 75),
    'p90':     lambda x: np.percentile(x, 90),
    'range':   lambda x: np.max(x) - np.min(x),
}

# Conjunto estendido usado na análise de separabilidade feature a feature
STATS_FUNCS_SEPARABILIDADE = {
    'media':    np.mean,
    'mediana':  np.median,
    'std':      np.std,
    'max':      np.max,
    'p75':      lambda x: np.percentile(x, 75),
    'p90':      lambda x: np.percentile(x, 90),
    'skewness': lambda x: float(skew(x)),
    'kurtosis': lambda x: float(kurtosis(x)),
    'range':    lambda x: np.max(x) - np.min(x),
}

# =============================================================================
# Filtro de Hampel
# =============================================================================

class ResultadoHampel:
    """Resultado do filtro de Hampel (mesma interface da biblioteca `hampel`)."""

    def __init__(self, filtered_data, outlier_indices, medians,
                 median_absolute_deviations, thresholds):
        self.filtered_data              = filtered_data
        self.outlier_indices            = outlier_indices
        self.medians                    = medians
        self.median_absolute_deviations = median_absolute_deviations
        self.thresholds                 = thresholds

    def __repr__(self):
        return (f"ResultadoHampel(outliers={len(self.outlier_indices)}, "
                f"amostras={len(self.filtered_data)})")


def hampel(data, window_size=5, n_sigma=3.0, k_mad=1.4826, tamanho_bloco=4096):
    """Filtro de Hampel (mediana móvel + MAD) implementado localmente.

    Para cada ponto é usada uma janela centrada de `window_size` amostras: se
    |x_i - mediana| > n_sigma * k_mad * MAD, o ponto é marcado como outlier e
    substituído pela mediana da janela. As bordas são replicadas para que todo
    ponto tenha uma janela completa.

    data          : pd.Series ou array com a série (já ordenada no tempo)
    window_size   : tamanho total da janela em AMOSTRAS (não em dias)
    n_sigma       : número de desvios para o limiar
    k_mad         : 1.4826 converte o MAD em estimativa de desvio padrão
    tamanho_bloco : processamento em blocos, apenas para controlar memória

    Retorna um ResultadoHampel com `filtered_data` e `outlier_indices`.

    Obs.: janelas com MAD = 0 (série constante no trecho) geram limiar 0 e
    marcam como outlier qualquer valor diferente da mediana.
    """
    if window_size < 1:
        raise ValueError("window_size deve ser >= 1")

    eh_series       = isinstance(data, pd.Series)
    indice_original = data.index if eh_series else None
    valores         = np.asarray(data, dtype=float)
    n               = len(valores)

    medianas  = valores.copy()
    mads      = np.zeros(n)
    limiares  = np.zeros(n)
    filtrados = valores.copy()

    meia_janela = window_size // 2

    if n == 0 or meia_janela == 0:
        outlier_indices = np.array([], dtype=int)
    else:
        # Replica as bordas para que todo ponto tenha uma janela completa
        estendida = np.pad(valores, meia_janela, mode="edge")
        janelas   = sliding_window_view(estendida, 2 * meia_janela + 1)

        # np.median é bem mais rápido, só usamos nanmedian se a série tiver NaN
        func_mediana = np.nanmedian if np.isnan(valores).any() else np.median

        for inicio in range(0, n, tamanho_bloco):
            fim   = min(inicio + tamanho_bloco, n)
            bloco = janelas[inicio:fim]

            med = func_mediana(bloco, axis=1)
            mad = func_mediana(np.abs(bloco - med[:, None]), axis=1)

            medianas[inicio:fim] = med
            mads[inicio:fim]     = mad
            limiares[inicio:fim] = n_sigma * k_mad * mad

        eh_outlier      = np.abs(valores - medianas) > limiares
        outlier_indices = np.flatnonzero(eh_outlier)
        filtrados[eh_outlier] = medianas[eh_outlier]

    if eh_series:
        filtrados = pd.Series(filtrados, index=indice_original, name=data.name)
        medianas  = pd.Series(medianas,  index=indice_original)

    return ResultadoHampel(filtrados, outlier_indices, medianas, mads, limiares)

# =============================================================================
# Carga e limpeza
# =============================================================================

def carregar_crystallizer(base_name, linhas_remover=None, labrefs_remover=None, limite_ppm=100):
    """Carrega e limpa a base de medições de um crystallizer.

    linhas_remover  : índices do CSV original (amostras duplicadas sem valor significativo)
    labrefs_remover : Labrefs descartados por terem valor muito discrepante
    limite_ppm      : acima disso a amostra é considerada discrepante

    Retorna (df_medicoes, df_duplicados).
    """
    df = pd.read_csv(DIR_DATA + base_name, sep=";", decimal=".")
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], format="%Y-%m-%d %H:%M:%S")
    df["Resultado de Ferro (ppm)"] = pd.to_numeric(df["Resultado de Ferro (ppm)"], errors="coerce")
    df.sort_values(by="TIMESTAMP", inplace=True)

    # Verificação de amostras duplicadas
    df_duplicados = df[df.duplicated(subset=["Labref"], keep=False)]

    # Retirando linhas duplicadas que não possuem amostras significativas
    if linhas_remover:
        df = df.drop(linhas_remover)

    # Aplicando média para medidas com Labref iguais (apenas as que possuem valores próximos)
    agg_logic = {col: "mean" if df[col].dtype.kind in "biufc" else "first"
                 for col in df.columns if col != "Labref"}
    df = df.groupby("Labref", as_index=False).agg(agg_logic)

    # Removendo amostras específicas com valor muito discrepante
    if labrefs_remover:
        df = df.drop(df[df["Labref"].isin(labrefs_remover)].index)

    # Removendo amostras com valores muito discrepantes.
    # limite_ppm=None desliga o corte fixo — use `limpar_excursoes`, que descarta
    # só o valor extremo (acima do teto de implausibilidade) e preserva todas as
    # leituras altas, inclusive as isoladas (falsos positivos para o modelo).
    if limite_ppm is not None:
        remove = df[df["Resultado de Ferro (ppm)"] > limite_ppm]
        print(f"Amostras acima de {limite_ppm} ppm: {len(remove)}")
        df = df.drop(remove.index)

    # O groupby por Labref reordena a base, então garantimos a ordem cronológica
    df = df.sort_values("TIMESTAMP")

    return df, df_duplicados


def aplicar_tratamentos(df_medicoes, coluna="Resultado de Ferro (ppm)",
                        dias_hampel=15, dias_hampel_longo=90,
                        amostras_por_dia=5, n_sigma=3.0):
    """Gera as versões tratadas de uma base de medições.

    Retorna (tratamentos, info):
      tratamentos -> {"Original", "Intervalo 0-10", "IQR", "Hampel", "Hampel 90d"}
      info        -> limites do IQR e quantidade de outliers de cada Hampel
    """
    serie = df_medicoes[coluna]

    # Intervalo fixo de 0 a 10 ppm
    df_0a10 = df_medicoes[serie.between(0, 10)].reset_index(drop=True)

    # IQR
    Q1, Q3 = serie.quantile(0.25), serie.quantile(0.75)
    IQR = Q3 - Q1
    limite_inf = Q1 - 1.5 * IQR
    limite_sup = Q3 + 1.5 * IQR
    df_iqr = df_medicoes[serie.between(limite_inf, limite_sup)].reset_index(drop=True)

    # Filtro de Hampel — janela curta e janela longa
    serie_ordenada = serie.reset_index(drop=True)

    res_hampel = hampel(serie_ordenada, window_size=dias_hampel * amostras_por_dia, n_sigma=n_sigma)
    df_hampel  = df_medicoes.copy().reset_index(drop=True)
    df_hampel[coluna] = res_hampel.filtered_data

    res_hampel_longo = hampel(serie_ordenada, window_size=dias_hampel_longo * amostras_por_dia, n_sigma=n_sigma)
    df_hampel_longo  = df_medicoes.copy().reset_index(drop=True)
    df_hampel_longo[coluna] = res_hampel_longo.filtered_data

    tratamentos = {
        "Original":       df_medicoes,
        "Intervalo 0-10": df_0a10,
        "IQR":            df_iqr,
        "Hampel":         df_hampel,
        "Hampel 90d":     df_hampel_longo,
    }
    info = {
        "limites_iqr":           (limite_inf, limite_sup),
        "outliers_hampel":       len(res_hampel.outlier_indices),
        "outliers_hampel_longo": len(res_hampel_longo.outlier_indices),
    }
    return tratamentos, info

# =============================================================================
# Janelas e eventos
# =============================================================================

def valores_na_janela(df_medicoes, ts, dias, coluna="Resultado de Ferro (ppm)"):
    """Valores da coluna na janela [ts - dias, ts) — recorte usado em todas as análises por evento."""
    inicio = ts - pd.Timedelta(days=dias)
    mask   = (df_medicoes["TIMESTAMP"] >= inicio) & (df_medicoes["TIMESTAMP"] < ts)
    return df_medicoes[mask][coluna].dropna().values


def adicionar_eventos_ultrapassagem(df_medicoes, df_eventos, threshold, dias_baseline=15,
                                    coluna="Resultado de Ferro (ppm)"):
    """Cria eventos Real=0 a partir das ultrapassagens do LC sem problema relatado.

    Uma ultrapassagem só vira evento se estiver a mais de `dias_baseline` de
    qualquer evento já existente e do último candidato aceito.
    """
    df_over = df_medicoes[df_medicoes[coluna] > threshold].sort_values("TIMESTAMP")

    eventos_detectados = []
    last_date = None

    for _, row in df_over.iterrows():
        candidato = row["TIMESTAMP"]

        # Ignora se muito próximo do último evento detectado neste loop
        if last_date is not None and (candidato - last_date).days < dias_baseline:
            continue

        # Ignora se já existe um evento próximo no df_eventos original
        diffs = (df_eventos["TIMESTAMP"] - candidato).abs()
        if (diffs <= pd.Timedelta(days=dias_baseline)).any():
            last_date = candidato  # avança o ponteiro para evitar acúmulo
            continue

        eventos_detectados.append(candidato)
        last_date = candidato

    # Cria dataframe com os eventos detectados
    df_novos_eventos = pd.DataFrame({
        "TIMESTAMP": eventos_detectados,
        "Evento": f"Ultrapassagem Fe > {threshold}ppm mas sem problema relatado",
        "Real": 0,
    })

    print(f"Eventos de ultrapassagem adicionados: {len(df_novos_eventos)}")

    # Adiciona ao df_eventos existente
    df_final = pd.concat([df_eventos, df_novos_eventos], ignore_index=True)
    return df_final.sort_values("TIMESTAMP").reset_index(drop=True)


def carregar_eventos(base_name, indices_remover=None, eventos_extra=None):
    """Carrega a tabela de eventos reais de um reator (todos com Real=1).

    indices_remover : posições a descartar (ex.: eventos fora do período de dados)
    eventos_extra   : lista de dicts com eventos adicionais (ex.: falso alarme manual)
    """
    df = pd.read_csv(DIR_DATA + base_name, sep=";", decimal=".")
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], format="%Y-%m-%d %H:%M:%S")
    df["Real"] = 1

    # Removendo eventos fora do período de dados
    if indices_remover:
        df = df.drop(indices_remover)

    if eventos_extra:
        df = pd.concat([df, pd.DataFrame(eventos_extra)], ignore_index=True)

    return df.sort_values("TIMESTAMP").reset_index(drop=True)


def separar_eventos_por_data(df_eventos, data_corte="2020-04-01"):
    """Divide a tabela de eventos em antes e depois de uma data de corte.

    Retorna (df_ate, df_apos) — a data de corte entra nos dois recortes, como no notebook.
    """
    corte = pd.to_datetime(data_corte)
    df_ate  = df_eventos[df_eventos["TIMESTAMP"] <= corte].reset_index(drop=True)
    df_apos = df_eventos[df_eventos["TIMESTAMP"] >= corte].reset_index(drop=True)
    return df_ate, df_apos

# =============================================================================
# Inspeção dos vitrificados (fonte dos eventos)
# =============================================================================

ARQUIVO_INSPECOES = "Vitrificados do PIA - Dados de inspeção.xlsx"

# Tag do equipamento -> nome do crystallizer usado no resto do estudo
TAG_CRYSTALLIZER = {"30-151": "C1", "30-251": "C2", "30-351": "C3"}

# Correções pontuais da planilha, aplicadas em carregar_inspecoes. Cada entrada existe
# porque o texto da planilha não permite inferir o marcador com regex segura, mas outra
# fonte (ou o contexto) estabelece o fato. Formato: (Crystallizer, data, campo, valor, fonte).
CORRECOES_INSPECAO = [
    # A planilha registra só "Reinstalação de plug", mas a linha do tempo do próprio deck
    # da equipe Bayer chama este evento de "Vazamento no plug do reparo" (18/10/2024).
    # A âncora na série cai em 20/10 — 2 dias da data do deck, 2 da planilha.
    ("C3", "2024-10-22", "Vazamento", True,
     "deck Bayer: 'Vazamento no plug do reparo' em 18/10/2024"),
    # "o reator (com o revestimento vitrificado novo) foi instalado devido a furo..." —
    # é uma troca de reator, mas o padrão 'reator ... instalado' não é seguro como regex
    # (colide com 'plug instalado no local do furo do reator'). Pré-série; só afeta a
    # idade de campanha do C2 no início da década de 2010.
    ("C2", "2009-03-15", "TrocaDoReator", True,
     "texto: 'o reator ... foi instalado devido a furo no revestimento'"),
]


def _normalizar_cabecalho(valor):
    """Normaliza o nome da coluna: sem acento, sem plural, minusculo."""
    if valor is None:
        return ""
    txt = unicodedata.normalize("NFKD", str(valor)).encode("ascii", "ignore").decode()
    txt = re.sub(r"[^a-z ]", " ", txt.lower())
    txt = re.sub(r"\s+", " ", txt).strip()
    # as abas alternam entre singular e plural; a coluna de âncora manual pode vir
    # com variações de texto
    return {"tipos de inspecao": "tipo de inspecao", "observacao": "observacoes",
            "data ancorada na serie de ferro": "data ancorada",
            "data ancorada na serie": "data ancorada"}.get(txt, txt)


def _parse_data_inspecao(valor, ano):
    """Converte a coluna DATA em (inicio, fim).

    A planilha mistura quatro formatos: datetime, 'DD - DD/MM/AAAA',
    'DD/MM - DD/MM/AAAA' (pode virar o mes) e texto livre como
    '08/05 e 12 - 28/05/2020'.
    """
    if valor is None:
        return None, None
    if isinstance(valor, (dt.datetime, dt.date)):
        d = pd.Timestamp(valor)
        return d, d

    txt = str(valor).strip()
    if not txt:
        return None, None

    m = re.match(r"^(\d{1,2})/(\d{1,2})\s*[-–]\s*(\d{1,2})/(\d{1,2})/(\d{4})$", txt)
    if m:
        d1, m1, d2, m2, ano_fim = map(int, m.groups())
        ano_ini = ano_fim if m1 <= m2 else ano_fim - 1
        return pd.Timestamp(ano_ini, m1, d1), pd.Timestamp(ano_fim, m2, d2)

    m = re.match(r"^(\d{1,2})\s*[-–]\s*(\d{1,2})/(\d{1,2})/(\d{4})$", txt)
    if m:
        d1, d2, mes, ano_fim = map(int, m.groups())
        return pd.Timestamp(ano_fim, mes, d1), pd.Timestamp(ano_fim, mes, d2)

    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", txt)
    if m:
        d = pd.Timestamp(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        return d, d

    # Texto livre: pega todas as datas que aparecerem e usa a menor e a maior
    ano_m = re.search(r"(\d{4})", txt)
    if ano_m:
        ano_txt = int(ano_m.group(1))
        datas = []
        for d_, m_ in re.findall(r"(\d{1,2})/(\d{1,2})", txt):
            try:
                datas.append(pd.Timestamp(ano_txt, int(m_), int(d_)))
            except ValueError:
                pass
        if datas:
            return min(datas), max(datas)

    return None, None


def carregar_inspecoes(arquivo=None):
    """Lê a planilha de inspeção dos vitrificados (uma aba por reator).

    Devolve uma linha por apontamento, com as colunas do Excel mais os
    marcadores derivados do texto:
      Emergencia    — parada de emergência / não programada
      Programada    — parada programada, preventiva ou por tempo de vida
      Vazamento     — menciona vazamento, furo, trinca ou quebra
      TrocaDoReator — o reator inteiro foi substituído (não apenas eixo/baffle)
      FerroCitado   — o texto menciona ferro
      Falha         — Emergencia ou Vazamento ou TrocaDoReator
    """
    caminho = arquivo or (DIR_DATA + ARQUIVO_INSPECOES)
    wb = openpyxl.load_workbook(caminho, data_only=True)

    linhas = []
    for ws in wb.worksheets:
        crystallizer = TAG_CRYSTALLIZER.get(ws.title, ws.title)
        cabecalho = [_normalizar_cabecalho(c.value) for c in ws[4]]
        ano_corrente = None

        for linha in ws.iter_rows(min_row=5, max_col=len(cabecalho)):
            valores = [c.value for c in linha]
            if not any(v is not None and str(v).strip() for v in valores):
                continue

            reg = {h: v for h, v in zip(cabecalho, valores) if h}
            if reg.get("ano"):
                ano_corrente = int(reg["ano"])

            inicio, fim = _parse_data_inspecao(reg.get("data"), ano_corrente)
            # "DATA ANCORADA NA SÉRIE DE FERRO": âncora definida à mão na planilha
            # (validação da planta). Quando preenchida, tem precedência sobre a
            # reancoragem automática — ver ajustar_eventos_para_lacunas.
            ancora_manual, _ = _parse_data_inspecao(reg.get("data ancorada"), ano_corrente)
            linhas.append({
                "Crystallizer": crystallizer,
                "Tag":          ws.title,
                "Ano":          ano_corrente,
                "DataTexto":    "" if reg.get("data") is None else str(reg["data"]),
                "Inicio":       inicio,
                "Fim":          fim,
                "DataAncoradaManual": ancora_manual,
                "Tipo":         str(reg.get("tipo de inspecao") or "").strip(),
                "Substituido":  str(reg.get("substituido") or "").strip(),
                "ReparoVidro":  str(reg.get("reparo vidro") or "").strip(),
                "ReparoOutros": str(reg.get("reparo outros") or "").strip(),
                "Ocorrimento":  str(reg.get("ocorrimento") or "").strip(),
                "Observacoes":  str(reg.get("observacoes") or "").strip(),
                "Servicos":     str(reg.get("servicos executados") or "").strip(),
            })

    df = pd.DataFrame(linhas)

    texto = (df["Tipo"] + " " + df["Ocorrimento"] + " " + df["Observacoes"] + " " + df["Servicos"]).str.lower()
    # 'emeg[êe]ncia' cobre o erro de digitação da planilha (C1 2009: "Parada de emegência")
    df["Emergencia"]    = texto.str.contains(r"emerg|emeg[êe]ncia|n[ãa]o programad", regex=True)
    df["Programada"]    = texto.str.contains(r"programad|preventiv|tempo de opera|tempo de vida|vida [uú]til", regex=True)
    # 'poro passante' / 'até a parte metálica' = vidro rompido até o aço — é a condição
    # que expõe ferro ao licor, mesmo quando encontrada em inspeção programada (C1 2015).
    # 'infiltraç' cobre reparo existente vazando (C3 2024).
    df["Vazamento"]     = texto.str.contains(
        r"vazamento|furo|trinca|quebra|perfura|poro passante|"
        r"at[ée] a parte met[áa]lica|infiltra[çc]", regex=True)
    # 'substituir o reator' / 'substituído o reator' cobrem as trocas registradas em
    # voz ativa (C2 2012: "R.I. ... para substituir o reator"; C2 2016: "Substituido o reator")
    df["TrocaDoReator"] = texto.str.contains(
        r"substitui[çc][ãa]o do reator|equipamento novo|reator novo|troca do reator|"
        r"substitui[çc][ãa]o do equipamento|equipamento substitu[íi]do|pelo spare|"
        r"substituir o reator|substitu[íi]do o reator", regex=True)
    df["FerroCitado"]   = texto.str.contains("ferro")

    # Correções documentadas (ver CORRECOES_INSPECAO no topo da seção)
    df["Corrigido"] = False
    for cryst, data, campo, valor, _fonte in CORRECOES_INSPECAO:
        m = (df["Crystallizer"] == cryst) & (df["Inicio"] == pd.Timestamp(data))
        df.loc[m, campo] = valor
        df.loc[m, "Corrigido"] = True

    df["Falha"] = df["Emergencia"] | df["Vazamento"] | df["TrocaDoReator"]

    return df.sort_values(["Crystallizer", "Inicio"]).reset_index(drop=True)


def ajustar_eventos_para_lacunas(df_eventos, map_medicoes, dias_lacuna=2,
                                 coluna_ts="Inicio", incluir_ultima_medicao=True,
                                 tolerancia_ancora_seg=300, max_desvio_ancora_dias=180,
                                 verbose=True):
    """Reancora os eventos que caem dentro de uma lacuna de amostragem.

    Quando o reator para, a amostragem para junto: o apontamento da planilha
    costuma cair no meio do período sem medição. Nesses casos o evento é
    deslocado para a última medição anterior à lacuna, de modo que a janela
    `[ts - dias, ts)` cubra os dados que realmente antecedem a parada.

    **Âncora manual tem precedência.** Se a linha traz `DataAncoradaManual`
    (coluna "DATA ANCORADA NA SÉRIE DE FERRO" da planilha — a âncora validada
    pela planta), ela define a âncora do evento. A planilha registra o timestamp
    da **própria medição**, truncado ao minuto (54 das 56 âncoras batem com uma
    amostra a menos de 60 s), então:

      - se existe uma medição a menos de `tolerancia_ancora_seg` da data
        informada, o evento é ancorado **um segundo depois dessa medição** — ela
        é a amostra que disparou a parada e precisa cair dentro da janela
        `[ts - dias, ts)`;
      - se a célula traz só a data (00:00) ou uma hora que não corresponde a
        nenhuma medição, usa-se a última medição **até o fim daquele dia**.

    A reancoragem automática vira apenas o fallback das linhas sem âncora manual
    (célula vazia, "-" ou "Fora do Período da Série").

    map_medicoes           : {crystallizer: df_medicoes}
    dias_lacuna            : intervalo entre medições consecutivas a partir do
                             qual o trecho é considerado lacuna
    incluir_ultima_medicao : se True, ancora um segundo depois da última medição,
                             para que ela entre na janela (é a amostra que
                             costuma disparar a parada)
    tolerancia_ancora_seg  : distância máxima para casar a âncora manual com uma
                             medição (padrão 300 s, absorve o truncamento ao minuto)
    max_desvio_ancora_dias : guarda contra erro de digitação — âncora manual a mais
                             de N dias da data do apontamento é ignorada (cai no
                             automático) e reportada. O maior desvio legítimo
                             observado é de 41 dias (evento constatado em parada de
                             planta), então 180 é folga confortável.

    Acrescenta as colunas TS_Ajustado, Deslocado, DiasDeslocado, LacunaDias,
    AncoraManual, AncoraCasouMedicao e AncoraRejeitada.
    """
    df = df_eventos.copy()
    ajustados, deslocados, dias_desl, lacunas, manuais, casou = [], [], [], [], [], []
    rejeitadas, avisos = [], []

    for _, ev in df.iterrows():
        ts = ev[coluna_ts]
        med = map_medicoes.get(ev["Crystallizer"])
        manual = ev.get("DataAncoradaManual", None)

        # guarda de sanidade: âncora absurdamente longe da data do apontamento
        rejeitada = False
        if (manual is not None and pd.notna(manual) and pd.notna(ts)
                and abs((pd.Timestamp(manual) - ts).days) > max_desvio_ancora_dias):
            avisos.append(f"    {ev['Crystallizer']} DATA={ts:%Y-%m-%d} "
                          f"ANCORADA={pd.Timestamp(manual):%Y-%m-%d} "
                          f"({(pd.Timestamp(manual) - ts).days:+d} dias) — ignorada")
            manual, rejeitada = None, True
        rejeitadas.append(rejeitada)

        # --- âncora manual da planilha: precedência total ---
        if manual is not None and pd.notna(manual) and med is not None and not med.empty:
            alvo = pd.Timestamp(manual)
            marcos = med["TIMESTAMP"].values
            i = np.searchsorted(marcos, np.datetime64(alvo), side="left")

            prox, dist = None, None
            for j in (i - 1, i):
                if 0 <= j < len(marcos):
                    t = pd.Timestamp(marcos[j])
                    d = abs((t - alvo).total_seconds())
                    if dist is None or d < dist:
                        prox, dist = t, d

            if prox is not None and dist <= tolerancia_ancora_seg:
                novo = prox + pd.Timedelta(seconds=1) if incluir_ultima_medicao else prox
                casou.append(True)
            else:
                # só a data, sem hora útil: última medição até o fim daquele dia
                fim_do_dia = alvo.normalize() + pd.Timedelta(days=1)
                i_prev = np.searchsorted(marcos, np.datetime64(fim_do_dia), side="left") - 1
                if i_prev >= 0:
                    t_prev = pd.Timestamp(marcos[i_prev])
                    novo = t_prev + pd.Timedelta(seconds=1) if incluir_ultima_medicao else t_prev
                else:  # âncora anterior ao início da série — usa a data como veio
                    novo = alvo
                casou.append(False)

            ajustados.append(novo)
            deslocados.append(pd.notna(ts) and abs((novo - ts).total_seconds()) > 86400)
            dias_desl.append((ts - novo).total_seconds() / 86400 if pd.notna(ts) else np.nan)
            lacunas.append(np.nan)
            manuais.append(True)
            continue

        manuais.append(False)
        casou.append(False)

        if pd.isna(ts) or med is None or med.empty:
            ajustados.append(ts); deslocados.append(False)
            dias_desl.append(np.nan); lacunas.append(np.nan)
            continue

        marcos = med["TIMESTAMP"].values
        i_prev = np.searchsorted(marcos, np.datetime64(ts), side="left") - 1
        i_next = np.searchsorted(marcos, np.datetime64(ts), side="left")

        if i_prev < 0:  # evento anterior ao início da série
            ajustados.append(ts); deslocados.append(False)
            dias_desl.append(np.nan); lacunas.append(np.nan)
            continue

        t_prev = pd.Timestamp(marcos[i_prev])
        t_next = pd.Timestamp(marcos[i_next]) if i_next < len(marcos) else None
        lacuna = (t_next - t_prev).total_seconds() / 86400 if t_next is not None else np.inf

        if lacuna > dias_lacuna:
            novo = t_prev + pd.Timedelta(seconds=1) if incluir_ultima_medicao else t_prev
            ajustados.append(novo)
            deslocados.append(True)
            dias_desl.append((ts - t_prev).total_seconds() / 86400)
            lacunas.append(lacuna)
        else:
            ajustados.append(ts); deslocados.append(False)
            dias_desl.append(0.0); lacunas.append(lacuna)

    df["TS_Ajustado"]        = ajustados
    df["Deslocado"]          = deslocados
    df["DiasDeslocado"]      = dias_desl
    df["LacunaDias"]         = lacunas
    df["AncoraManual"]       = manuais
    df["AncoraCasouMedicao"] = casou
    df["AncoraRejeitada"]    = rejeitadas

    if verbose and avisos:
        print(f"  ATENÇÃO: {len(avisos)} âncora(s) manual(is) a mais de "
              f"{max_desvio_ancora_dias} dias do apontamento — provável erro de digitação "
              f"na planilha. Foram ignoradas (usou-se a reancoragem automática):")
        for a in avisos:
            print(a)

    return df

# =============================================================================
# Limpeza de excursões (erro de laboratório vs medição real alta)
# =============================================================================

def classificar_amostras_altas(df_medicoes, lc_alto=20, lc_corroboracao=5, min_corroboracao=2,
                               dias_vizinhanca=2, dias_parada=2, coluna=COLUNA_FE):
    """Separa erro de laboratório de excursão real entre as amostras acima de `lc_alto`.

    Uma amostra alta é considerada REAL se qualquer um dos dois critérios valer:

      (a) corroborada     — pelo menos `min_corroboracao` outras amostras acima de
                            `lc_corroboracao` ppm na vizinhança de ±`dias_vizinhanca` dias;
      (b) seguida de parada — o próximo ponto normal da série só aparece mais de
                            `dias_parada` dias depois.

    O critério (b) é indispensável: numa falha abrupta a operação para o reator logo
    após a leitura, então a excursão não chega a aparecer em outras amostras. Foi
    exatamente o que aconteceu em 23/11/2013 no C3 (264 ppm, isolada, mas real).

    Acrescenta as colunas Alta, Corroborada, SeguidaDeParada, ExcursaoReal e ErroLab.
    """
    out = df_medicoes.copy().reset_index(drop=True)
    ts = out["TIMESTAMP"]

    out["Alta"] = out[coluna] > lc_alto
    out["Corroborada"] = False
    out["SeguidaDeParada"] = False

    for i in out.index[out["Alta"]]:
        t = ts.loc[i]

        viz = out[(ts >= t - pd.Timedelta(days=dias_vizinhanca)) &
                  (ts <= t + pd.Timedelta(days=dias_vizinhanca)) & (out.index != i)]
        out.loc[i, "Corroborada"] = (viz[coluna] > lc_corroboracao).sum() >= min_corroboracao

        # o próximo ponto NORMAL (ignora outras leituras altas do mesmo episódio)
        posteriores = out[(ts > t) & (out[coluna] <= lc_alto)]["TIMESTAMP"]
        if len(posteriores) == 0:
            out.loc[i, "SeguidaDeParada"] = True
        else:
            out.loc[i, "SeguidaDeParada"] = (posteriores.min() - t).total_seconds() / 86400 > dias_parada

    out["ExcursaoReal"] = out["Alta"] & (out["Corroborada"] | out["SeguidaDeParada"])
    out["ErroLab"] = out["Alta"] & ~out["ExcursaoReal"]
    return out


def limpar_excursoes(df_medicoes, teto_implausivel=1000, coluna=COLUNA_FE,
                     descartar_isoladas=False, verbose=True, **kwargs):
    """Descarta APENAS o valor extremo (fisicamente impossível) e preserva o resto.

    Política vigente (`descartar_isoladas=False`, padrão): a única remoção é o
    **teto de implausibilidade** — valores acima de `teto_implausivel` ppm não são
    operacionalmente possíveis (na base inteira existe 1 amostra acima de 1000 ppm,
    de 20000 ppm; a segunda maior é 999). Tudo abaixo do teto fica na série.

    Por que a leitura alta ISOLADA fica: ela é uma ultrapassagem do LC que não foi
    seguida de falha, ou seja, é exatamente o **falso positivo** que o modelo precisa
    aprender a rejeitar — e é o falso positivo mais difícil, porque é o de maior
    amplitude. Descartá-la retirava da base o exemplo negativo mais informativo e
    inflava artificialmente a precisão de qualquer detector de cauda (max > 20 ppm
    ia de 14 para 5 falsos positivos só por causa do filtro, não por mérito do
    detector). Robustez a spike espúrio deve vir de feature robusta (mediana, p75/p90),
    não de apagar a amostra.

    Uma dessas leituras "isoladas" (C1, 20/05/2011, 23.7 ppm) está 8.5 dias antes de uma
    falha ancorada — mantê-la leva `max > 10 ppm` de 6 para 7 verdadeiros positivos.

    `classificar_amostras_altas` continua sendo calculada, mas o veredito
    (corroborada / seguida de parada / isolada) é **diagnóstico**: rotula, não filtra.
    Para reproduzir a política antiga (remover as isoladas), passe
    `descartar_isoladas=True`.

    Retorna (df_limpo, df_descartado). `df_descartado` contém SÓ o que saiu da base
    de fato (coluna `Motivo`) — com o padrão, apenas as amostras acima do teto. As
    leituras altas isoladas que ficam podem ser listadas com
    `classificar_amostras_altas(df_limpo)` (colunas Corroborada / SeguidaDeParada /
    ExcursaoReal / ErroLab).
    """
    df = df_medicoes.copy()

    acima_teto = df[coluna] > teto_implausivel
    implausiveis = df[acima_teto].copy()
    implausiveis["Motivo"] = "acima do teto de implausibilidade"
    df = df[~acima_teto]

    classificado = classificar_amostras_altas(df, coluna=coluna, **kwargs)
    isoladas = classificado[classificado["ErroLab"]].copy()
    colunas_diag = ["Alta", "Corroborada", "SeguidaDeParada", "ExcursaoReal", "ErroLab"]

    if descartar_isoladas:
        isoladas["Motivo"] = "leitura alta isolada"
        removidas = [implausiveis, isoladas[list(implausiveis.columns)]]
        mantidas = ~classificado["ErroLab"]
    else:
        removidas = [implausiveis]
        mantidas = pd.Series(True, index=classificado.index)

    df_limpo = classificado[mantidas].drop(columns=colunas_diag).reset_index(drop=True)

    descartado = pd.concat(removidas, ignore_index=True)
    descartado = descartado.sort_values("TIMESTAMP").reset_index(drop=True)

    if verbose:
        n_real = int(classificado["ExcursaoReal"].sum())
        print(f"Amostras acima do teto de {teto_implausivel} ppm (descartadas): {len(implausiveis)}")
        print(f"Excursões reais (corroboradas ou seguidas de parada)   : {n_real}")
        if descartar_isoladas:
            print(f"Leituras altas isoladas DESCARTADAS                    : {len(isoladas)}")
        else:
            print(f"Leituras altas isoladas MANTIDAS (falsos positivos)    : {len(isoladas)}")

    return df_limpo, descartado


def identificar_lacunas(df_medicoes, minimo_dias=10):
    """Lista os intervalos (inicio, fim) sem amostra acima de `minimo_dias`."""
    d = df_medicoes["TIMESTAMP"].diff().dt.total_seconds() / 86400
    return [(df_medicoes.loc[i, "TIMESTAMP"] - pd.Timedelta(days=d.loc[i]),
             df_medicoes.loc[i, "TIMESTAMP"]) for i in d.index[d > minimo_dias]]


def identificar_paradas_de_planta(map_medicoes, minimo_dias=10):
    """Lacunas que atingem TODOS os reatores ao mesmo tempo.

    Quando os três param juntos não é falha de equipamento: é parada de planta
    ou de laboratório. Eventos que caem numa dessas janelas não devem entrar no
    conjunto de falhas — a janela anterior descreve operação normal.
    """
    lacunas = {c: identificar_lacunas(df, minimo_dias) for c, df in map_medicoes.items()}
    referencia = list(lacunas)[0]
    demais = [c for c in lacunas if c != referencia]

    paradas = []
    for inicio, fim in lacunas[referencia]:
        if all(any(not (f2 < inicio or i2 > fim) for i2, f2 in lacunas[c]) for c in demais):
            paradas.append((inicio, fim))
    return paradas


def montar_tabela_eventos(inspecoes, map_medicoes, paradas_planta=None,
                          dias_lacuna=2, fundir_dias=7, verbose=True):
    """Constrói a tabela de eventos definitiva a partir da planilha de inspeção.

    Etapas:
      1. reancora os apontamentos que caem dentro de uma lacuna de amostragem
         (`ajustar_eventos_para_lacunas`);
      2. marca os que caem numa parada de planta (lacuna simultânea nos 3 reatores);
      3. mantém os apontamentos que são falha de equipamento. Em parada de planta a
         regra é seletiva: **dano físico real (`Vazamento`) é resgatado** — o furo
         encontrado na abertura existia antes da parada, então a última janela de
         operação é o lugar certo para procurar sinal — enquanto troca SEM dano
         (preventiva/spare, ex.: C3 05/2020) continua fora, porque ali a janela
         anterior descreve operação normal antes de uma parada programada;
      4. funde apontamentos do mesmo reator a menos de `fundir_dias` dias — a
         planilha frequentemente registra a mesma parada em duas linhas.

    Colunas de saída além das da planilha:
      Selecionado        — entra como positivo no estudo
      DescobertoEmParada — resgatado do filtro de parada de planta (âncora recua
                           para a última medição antes da parada; janela íntegra,
                           mas a DATA da falha é incerta — foi constatada na abertura)
      TipoFalha          — 'emergência' | 'constatada em parada de planta' |
                           'constatada/programada' (como a falha foi encontrada)
    """
    if paradas_planta is None:
        paradas_planta = identificar_paradas_de_planta(map_medicoes)

    df = ajustar_eventos_para_lacunas(inspecoes, map_medicoes, dias_lacuna=dias_lacuna,
                                      verbose=verbose)

    inicio_serie = min(m["TIMESTAMP"].min() for m in map_medicoes.values())
    df["NoPeriodo"] = df["Inicio"] >= inicio_serie
    df["LacunaDePlanta"] = [
        any(ini <= ts <= fim for ini, fim in paradas_planta) if pd.notna(ts) else False
        for ts in df["Inicio"]
    ]

    candidatos = df["NoPeriodo"] & df["Falha"] & (~df["LacunaDePlanta"] | df["Vazamento"])

    # Funde apontamentos consecutivos do mesmo reator. Os marcadores do registro
    # absorvido são propagados para o que fica (OR): a planilha costuma separar a
    # constatação da parada de emergência em duas linhas, e sem isso o evento
    # sobrevivente perderia o rótulo "emergência" só por ser o mais antigo dos dois.
    df["Selecionado"] = False
    df["ApontamentosFundidos"] = 0
    for cryst in df["Crystallizer"].unique():
        sel = df[candidatos & (df["Crystallizer"] == cryst)].sort_values("TS_Ajustado")
        ultimo, i_mantido = None, None
        for i, linha in sel.iterrows():
            if ultimo is None or (linha["TS_Ajustado"] - ultimo).days > fundir_dias:
                df.loc[i, "Selecionado"] = True
                i_mantido = i
            elif i_mantido is not None:
                for col in ("Emergencia", "Vazamento", "TrocaDoReator", "FerroCitado"):
                    df.loc[i_mantido, col] = bool(df.loc[i_mantido, col]) or bool(linha[col])
                df.loc[i_mantido, "ApontamentosFundidos"] += 1
            ultimo = linha["TS_Ajustado"]

    df["DescobertoEmParada"] = df["Selecionado"] & df["LacunaDePlanta"]
    df["TipoFalha"] = np.where(~df["Selecionado"], "",
                      np.where(df["Emergencia"], "emergência",
                      np.where(df["DescobertoEmParada"], "constatada em parada de planta",
                               "constatada/programada")))

    if verbose:
        descartadas_pp = int((df["NoPeriodo"] & df["Falha"] & df["LacunaDePlanta"]
                              & ~df["Selecionado"]).sum())
        print(f"Apontamentos na planilha            : {len(df)}")
        print(f"  dentro do período da série        : {int(df['NoPeriodo'].sum())}")
        n_man = int((df["NoPeriodo"] & df["AncoraManual"]).sum())
        n_casou = int((df["NoPeriodo"] & df["AncoraManual"] & df["AncoraCasouMedicao"]).sum())
        print(f"  com âncora manual da planilha     : {n_man}"
              f"  ({n_casou} casaram com uma medição; coluna 'DATA ANCORADA NA SÉRIE DE FERRO')")
        print(f"  reancorados por cair em lacuna    : {int((df['NoPeriodo'] & df['Deslocado'] & ~df['AncoraManual']).sum())}")
        print(f"  em parada de planta               : {int((df['NoPeriodo'] & df['LacunaDePlanta']).sum())}"
              f"  (resgatadas com dano físico: {int(df['DescobertoEmParada'].sum())}, "
              f"descartadas: {descartadas_pp})")
        print(f"  falhas selecionadas               : {int(df['Selecionado'].sum())}")
        print()
        print(df[df["Selecionado"]].groupby("Crystallizer").agg(
            falhas=("TS_Ajustado", "size"),
            emergencia=("Emergencia", "sum"),
            troca_reator=("TrocaDoReator", "sum"),
            cita_ferro=("FerroCitado", "sum"),
            em_parada=("DescobertoEmParada", "sum"),
        ).to_string())

    return df


def eventos_para_notebook(df_eventos, crystallizer):
    """Converte a tabela de inspeção no formato usado pelo resto do notebook.

    Devolve um dataframe com TIMESTAMP / Evento / Real=1, uma linha por falha
    selecionada do reator pedido.
    """
    sel = df_eventos[(df_eventos["Crystallizer"] == crystallizer) & df_eventos["Selecionado"]]
    descricao = sel["Ocorrimento"].where(sel["Ocorrimento"].str.strip() != "", sel["Observacoes"])
    return pd.DataFrame({
        "TIMESTAMP": sel["TS_Ajustado"].values,
        "Evento":    descricao.str.replace(r"\s+", " ", regex=True).str.slice(0, 120).values,
        "Real":      1,
    }).sort_values("TIMESTAMP").reset_index(drop=True)


def unificar_eventos(eventos_por_reator, intervalo_min_dias=15, verbose=True):
    """Concatena as tabelas de evento dos reatores e deduplica a classe negativa.

    Duas etapas, ambas cross-reator (uma ultrapassagem quase simultânea em dois
    reatores é o mesmo episódio de processo, não dois negativos independentes):
      1. remove Real=0 a menos de `intervalo_min_dias` de qualquer Real=1;
      2. entre os Real=0 restantes, mantém apenas o primeiro de cada grupo com
         menos de `intervalo_min_dias` de espaçamento (ordem cronológica).

    eventos_por_reator : {"C1": df_eventos, ...} — cada df com TIMESTAMP/Evento/Real
    """
    df = pd.concat(
        [d.assign(Crystallizer=c) for c, d in eventos_por_reator.items()],
        ignore_index=True
    ).sort_values("TIMESTAMP")

    # Etapa 1 — Real=0 perto de qualquer Real=1 descreve a mesma janela da falha
    ts_real1 = df.loc[df["Real"] == 1, "TIMESTAMP"]
    perto_de_real1 = df["TIMESTAMP"].apply(
        lambda ts: ((ts_real1 - ts).abs() <= pd.Timedelta(days=intervalo_min_dias)).any()
    )
    mask_remover = (df["Real"] == 0) & perto_de_real1
    removidos_por_real1 = int(mask_remover.sum())
    df = df[~mask_remover].copy()

    # Etapa 2 — espaçamento mínimo entre Real=0, mantendo o primeiro cronologicamente
    df_real1 = df[df["Real"] == 1].copy()
    df_real0 = df[df["Real"] == 0].sort_values("TIMESTAMP").copy()

    mantidos, ultimo_ts = [], None
    for _, linha in df_real0.iterrows():
        ts = linha["TIMESTAMP"]
        if ultimo_ts is None or (ts - ultimo_ts).days > intervalo_min_dias:
            mantidos.append(linha)
            ultimo_ts = ts
    df_real0_limpo = pd.DataFrame(mantidos)

    resultado = pd.concat([df_real1, df_real0_limpo], ignore_index=True) \
        .sort_values("TIMESTAMP").reset_index(drop=True)

    if verbose:
        print(f"Real=0 removidos por proximidade a Real=1: {removidos_por_real1}")
        print(f"Real=0 antes da deduplicação  : {len(df_real0)}")
        print(f"Real=0 após a deduplicação    : {len(df_real0_limpo)}")
        print(f"Removidos na deduplicação     : {len(df_real0) - len(df_real0_limpo)}")
        print(f"\nDistribuição final:")
        print(resultado["Real"].value_counts().to_string())
        print(f"Total de eventos: {len(resultado)}")

    return resultado


# =============================================================================
# Baseline: a regra vigente e as alternativas
# =============================================================================

def avaliar_baseline_lift(map_medicoes, map_eventos, criterios=None, janela_dias=15,
                          n_janelas=2000, random_state=RANDOM_STATE, coluna=COLUNA_FE):
    """Compara a janela que antecede a falha com uma janela qualquer (taxa base).

    O lift responde à pergunta que decide o projeto: uma janela antes de falha é
    mesmo diferente de uma janela sorteada ao acaso? Sem esse controle, um recall
    alto pode ser apenas reflexo de o critério disparar o tempo todo.

    criterios : lista de (nome, função que recebe a janela e devolve bool)
    map_eventos : {crystallizer: DataFrame com TIMESTAMP}
    """
    if criterios is None:
        criterios = criterios_padrao()

    rng = np.random.default_rng(random_state)
    linhas = []

    for nome, teste in criterios:
        # nas janelas que antecedem falha
        acertos = total_falhas = 0
        for cryst, df_ev in map_eventos.items():
            df = map_medicoes[cryst]
            for ts in df_ev["TIMESTAMP"]:
                jan = df[(df["TIMESTAMP"] >= ts - pd.Timedelta(days=janela_dias)) &
                         (df["TIMESTAMP"] < ts)]
                if jan.empty:
                    continue
                total_falhas += 1
                acertos += bool(teste(jan, coluna))

        # nas janelas sorteadas
        base = total_base = 0
        for cryst, df in map_medicoes.items():
            t0 = df["TIMESTAMP"].min() + pd.Timedelta(days=janela_dias)
            t1 = df["TIMESTAMP"].max()
            sorteios = rng.uniform(0, (t1 - t0).total_seconds(), n_janelas)
            for s in sorteios:
                ts = t0 + pd.Timedelta(seconds=float(s))
                jan = df[(df["TIMESTAMP"] >= ts - pd.Timedelta(days=janela_dias)) &
                         (df["TIMESTAMP"] < ts)]
                if jan.empty:
                    continue
                total_base += 1
                base += bool(teste(jan, coluna))

        p_falha = acertos / total_falhas if total_falhas else np.nan
        p_base = base / total_base if total_base else np.nan
        linhas.append({
            "criterio":   nome,
            "nas_falhas": p_falha,
            "taxa_base":  p_base,
            "lift":       p_falha / p_base if p_base else np.nan,
            "n_falhas":   total_falhas,
        })

    return pd.DataFrame(linhas)


def criterios_padrao():
    """Critérios comparados no baseline: a regra vigente e as alternativas."""
    return [
        ("max > 5 ppm  (regra vigente)", lambda j, c: (j[c] > 5).any()),
        ("max > 7 ppm",                  lambda j, c: (j[c] > 7).any()),
        ("max > 10 ppm",                 lambda j, c: (j[c] > 10).any()),
        ("max > 20 ppm",                 lambda j, c: (j[c] > 20).any()),
        ("mediana diária >= 3.5 ppm",    lambda j, c: bool(
            (j.set_index("TIMESTAMP")[c].resample("D").median().dropna() >= 3.5).any())),
        ("mediana diária >= 4.0 ppm",    lambda j, c: bool(
            (j.set_index("TIMESTAMP")[c].resample("D").median().dropna() >= 4.0).any())),
        ("p90 da janela >= 4.5 ppm",     lambda j, c: j[c].quantile(0.9) >= 4.5),
    ]


def avaliar_detector_diario(map_medicoes, map_eventos, estatistica="median", limite=3.5,
                            dias_seguidos=1, janela_alarme=15, agrupar_dias=15, coluna=COLUNA_FE):
    """Precisão, recall e F2 de um detector aplicado à estatística diária.

    Alarme = estatística diária acima de `limite` por `dias_seguidos` dias; alarmes
    a menos de `agrupar_dias` viram um só. Acerto se a falha ocorre em até
    `janela_alarme` dias depois do alarme.
    """
    alarmes, vp, fp, total = {}, 0, 0, 0

    for cryst, df in map_medicoes.items():
        serie = df.set_index("TIMESTAMP")[coluna].resample("D").agg(estatistica).dropna()
        acima = (serie > limite).rolling(dias_seguidos).sum() >= dias_seguidos
        agrupados, ultimo = [], None
        for data in acima.index[acima.fillna(False)]:
            if ultimo is None or (data - ultimo).days > agrupar_dias:
                agrupados.append(data)
            ultimo = data
        alarmes[cryst] = agrupados

    for cryst, df_ev in map_eventos.items():
        for ts in df_ev["TIMESTAMP"]:
            total += 1
            if any(0 <= (ts - a).days <= janela_alarme for a in alarmes.get(cryst, [])):
                vp += 1

    for cryst, lista in alarmes.items():
        ts_falhas = map_eventos.get(cryst, pd.DataFrame({"TIMESTAMP": []}))["TIMESTAMP"]
        for a in lista:
            if not any(0 <= (ts - a).days <= janela_alarme for ts in ts_falhas):
                fp += 1

    precisao = vp / (vp + fp) if vp + fp else 0.0
    recall = vp / total if total else 0.0
    f2 = 5 * precisao * recall / (4 * precisao + recall) if precisao + recall else 0.0
    return {"estatistica": estatistica, "limite": limite, "dias_seguidos": dias_seguidos,
            "VP": vp, "FP": fp, "precisao": precisao, "recall": recall, "F2": f2}



def _agrupar_alarmes(datas, agrupar_dias=15):
    """Alarmes a menos de `agrupar_dias` de distância contam como um só."""
    agrupados, ultimo = [], None
    for d in sorted(datas):
        if ultimo is None or (d - ultimo).days > agrupar_dias:
            agrupados.append(d)
        ultimo = d
    return agrupados


def _metricas_alarmes(alarmes, map_falhas, janela_alarme=15, agrupar_dias=15, rotulo=""):
    """Precisão / recall / F2 de um conjunto de alarmes já datados, por reator."""
    alarmes = {c: _agrupar_alarmes(d, agrupar_dias) for c, d in alarmes.items()}
    total = sum(len(df) for df in map_falhas.values())
    vp = fp = 0
    for cryst, df_ev in map_falhas.items():
        for ts in df_ev["TIMESTAMP"]:
            if any(0 <= (ts - a).days <= janela_alarme for a in alarmes.get(cryst, [])):
                vp += 1
    for cryst, lista in alarmes.items():
        ts_falhas = map_falhas.get(cryst, pd.DataFrame({"TIMESTAMP": []}))["TIMESTAMP"]
        for a in lista:
            if not any(0 <= (ts - a).days <= janela_alarme for ts in ts_falhas):
                fp += 1
    precisao = vp / (vp + fp) if vp + fp else 0.0
    recall = vp / total if total else 0.0
    f2 = 5 * precisao * recall / (4 * precisao + recall) if precisao + recall else 0.0
    return {"regra": rotulo, "VP": vp, "FP": fp, "precisao": round(precisao, 3),
            "recall": round(recall, 3), "F2": round(f2, 3)}


def avaliar_detector_composto(map_medicoes, map_falhas, limite_max=20, limite_mediana=3.0,
                              limite_margem=0.6, janela_alarme=15, agrupar_dias=15,
                              margem=None, coluna=COLUNA_FE, rotulo=None):
    """Regra composta: `max diário > limite_max` OU (`mediana diária > limite_mediana`
    E `margem cross-reator > limite_margem`).

    Os dois ramos existem porque o sinal aparece de duas formas diferentes e nenhuma
    regra única pega as duas:
      - IMPULSO: uma única leitura altíssima (C3 23/11/2013, 264 ppm) não move a mediana
        diária — só o ramo do `max` pega;
      - ELEVAÇÃO SUSTENTADA: vários dias em patamar acima do normal, que o `max` de um dia
        não distingue de um pico isolado de laboratório — o ramo da mediana + margem pega,
        e a margem é o que separa "este reator subiu" de "a planta inteira subiu".

    Medida sobre as 31 falhas ancoradas, esta regra entrega mais recall que a regra
    vigente (5 ppm) com bem menos alarme falso. ATENÇÃO: os limiares foram escolhidos
    olhando essas mesmas 31 falhas — é ponto de partida para a etapa supervisionada
    validar no split temporal, não resultado validado.
    """
    diario_max = serie_diaria(map_medicoes, "max", coluna)
    diario_med = serie_diaria(map_medicoes, "median", coluna)
    if margem is None:
        _, _, margem = calcular_margem_cross_reator(map_medicoes, coluna, verbose=False)

    alarmes = {}
    for cryst in map_medicoes:
        s_max = diario_max[cryst].dropna()
        s_med = diario_med[cryst].dropna()
        s_mar = margem[cryst].dropna()
        impulso = set(s_max[s_max > limite_max].index)
        sustentado = {d for d in s_med[s_med > limite_mediana].index
                      if d in s_mar.index and s_mar.loc[d] > limite_margem}
        alarmes[cryst] = sorted(impulso | sustentado)

    if rotulo is None:
        rotulo = (f"max > {limite_max} OU (mediana > {limite_mediana} "
                  f"E margem > {limite_margem})")
    return _metricas_alarmes(alarmes, map_falhas, janela_alarme, agrupar_dias, rotulo)


def avaliar_detector_adaptativo(map_medicoes, map_falhas, quantil=0.99, dias=365,
                                janela_alarme=15, agrupar_dias=15, coluna=COLUNA_FE):
    """Detector com limite móvel: mediana diária acima do próprio quantil dos últimos `dias`.

    Alternativa ao limite fixo que sobrevive ao drift da série sem recalibração manual.
    """
    diario = serie_diaria(map_medicoes, "median", coluna)
    limites = baseline_movel(map_medicoes, quantil, dias, coluna=coluna)
    alarmes = {}
    for cryst in map_medicoes:
        s = diario[cryst].dropna()
        lim = limites[cryst]
        alarmes[cryst] = list(s[s > lim].dropna().index)
    return _metricas_alarmes(alarmes, map_falhas, janela_alarme, agrupar_dias,
                             f"adaptativo: mediana > quantil {quantil} de {dias} d")


def comparar_detectores(map_medicoes, map_falhas, margem=None, coluna=COLUNA_FE):
    """Tabela única com a regra vigente e as alternativas — a régua da Etapa supervisionada.

    Tudo aqui é medido sobre as mesmas 31 falhas ancoradas, com a mesma convenção de
    alarme (agrupamento de 15 dias, acerto se a falha vem em até 15 dias).
    """
    if margem is None:
        _, _, margem = calcular_margem_cross_reator(map_medicoes, coluna, verbose=False)

    diario_max = serie_diaria(map_medicoes, "max", coluna)
    diario_med = serie_diaria(map_medicoes, "median", coluna)

    linhas = []
    for lim in (5, 10, 20):
        al = {c: list(diario_max[c].dropna()[diario_max[c].dropna() > lim].index)
              for c in map_medicoes}
        rot = f"max diário > {lim} ppm" + (" (regra vigente)" if lim == 5 else "")
        linhas.append(_metricas_alarmes(al, map_falhas, rotulo=rot))
    for lim in (3.0, 3.5):
        al = {c: list(diario_med[c].dropna()[diario_med[c].dropna() > lim].index)
              for c in map_medicoes}
        linhas.append(_metricas_alarmes(al, map_falhas, rotulo=f"mediana diária > {lim} ppm"))
    for lim in (0.8, 1.2, 1.6):
        al = {c: list(margem[c].dropna()[margem[c].dropna() > lim].index) for c in map_medicoes}
        linhas.append(_metricas_alarmes(al, map_falhas, rotulo=f"margem cross-reator > {lim}"))
    linhas.append(avaliar_detector_adaptativo(map_medicoes, map_falhas))
    for lmax, lmed, lmar in ((20, 3.0, 0.6), (20, 3.5, 0.9), (10, 3.5, 0.9)):
        linhas.append(avaliar_detector_composto(map_medicoes, map_falhas, lmax, lmed, lmar,
                                                margem=margem, coluna=coluna))
    return pd.DataFrame(linhas).sort_values("F2", ascending=False).reset_index(drop=True)

# =============================================================================
# Gráficos da série
# =============================================================================

def plot_crystallizer(df_med, df_eventos, titulo, mostrar_falsos=True):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_med['TIMESTAMP'],
        y=df_med["Resultado de Ferro (ppm)"],
        mode='lines',
        name="Resultado de Ferro (ppm)",
        line=dict(color='black')
    ))
    fig.add_hline(y=5, line_width=3, line_dash="dash", line_color="red")

    # Traces fantasma apenas para a legenda
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        name="Evento Real",
        line=dict(color='red', width=1.5, dash='dash')
    ))
    if mostrar_falsos:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            name="Evento Falso Positivo",
            line=dict(color='blue', width=1.5, dash='dash')
        ))

    for _, row in df_eventos.iterrows():
        if not mostrar_falsos and row["Real"] == 0:
            continue
        cor = "red" if row["Real"] == 1 else "blue"
        fig.add_shape(
            type="line",
            x0=str(row["TIMESTAMP"]),
            x1=str(row["TIMESTAMP"]),
            y0=0, y1=1,
            yref="paper",
            line=dict(color=cor, width=1.5, dash="dash")
        )

    fig.update_layout(
        template='plotly_white',
        hovermode='x unified',
        title=titulo
    )
    return fig


def plot_crystallizers(crystallizers, titulo, mostrar_falsos=True):
    fig = go.Figure()

    for c in crystallizers:
        fig.add_trace(go.Scatter(
            x=c["df"]['TIMESTAMP'],
            y=c["df"]["Resultado de Ferro (ppm)"],
            mode='lines',
            name=c["nome"],
            line=dict(color=c["cor"])
        ))

        for _, row in c["df_eventos"].iterrows():
            if not mostrar_falsos and row["Real"] == 0:
                continue

            cor = c["cor"] if row["Real"] == 1 else "gray"
            fig.add_shape(
                type="line",
                x0=str(row["TIMESTAMP"]),
                x1=str(row["TIMESTAMP"]),
                y0=0, y1=1,
                yref="paper",
                line=dict(color=cor, width=1.5, dash="dash")
            )
            fig.add_annotation(
                x=str(row["TIMESTAMP"]),
                y=1,
                yref="paper",
                text=row["EVENTO"] if "EVENTO" in c["df_eventos"].columns else "",
                showarrow=False,
                textangle=-90,
                yanchor="top",
                font=dict(color=cor)
            )

    fig.add_hline(y=5, line_width=3, line_dash="dash", line_color="red")
    fig.update_layout(
        template='plotly_white',
        hovermode='x unified',
        title=titulo
    )
    return fig


def plot_mm_crystallizer(df_med, df_eventos, titulo, janelas=None, mostrar_falsos=True):
    if janelas is None:
        janelas = [15, 12, 9, 6, 3]
    cores_mm = ['blue', 'green', 'orange', 'purple', 'red']

    df_mm = df_med.copy().set_index('TIMESTAMP')
    for dias in janelas:
        df_mm[f'MM_{dias}D'] = df_mm["Resultado de Ferro (ppm)"].rolling(window=f'{dias}D').mean()
    df_mm = df_mm.reset_index()

    fig = go.Figure()

    # Série original
    fig.add_trace(go.Scatter(
        x=df_mm['TIMESTAMP'],
        y=df_mm["Resultado de Ferro (ppm)"],
        mode='lines',
        name="Resultado de Ferro (ppm)",
        line=dict(color='gray', width=1),
        opacity=0.6
    ))

    # Médias móveis
    for dias, cor in zip(janelas, cores_mm):
        fig.add_trace(go.Scatter(
            x=df_mm['TIMESTAMP'],
            y=df_mm[f'MM_{dias}D'],
            mode='lines',
            name=f"MM {dias}D",
            line=dict(color=cor, width=2)
        ))

    fig.add_hline(y=5, line_width=3, line_dash="dash", line_color="red")

    for _, row in df_eventos.iterrows():
        if not mostrar_falsos and row["Real"] == 0:
            continue

        cor = "red" if row["Real"] == 1 else "blue"
        fig.add_shape(
            type="line",
            x0=str(row["TIMESTAMP"]),
            x1=str(row["TIMESTAMP"]),
            y0=0, y1=1,
            yref="paper",
            line=dict(color=cor, width=1.5, dash="dash")
        )
        fig.add_annotation(
            x=str(row["TIMESTAMP"]),
            y=1,
            yref="paper",
            text=row["EVENTO"] if "EVENTO" in df_eventos.columns else "",
            showarrow=False,
            textangle=-90,
            yanchor="top"
        )

    fig.update_layout(
        template='plotly_white',
        hovermode='x unified',
        title=titulo
    )
    return fig


def plot_mm_crystallizers(crystallizers, num_dias=7, titulo=None, mostrar_falsos=True):
    CORES = {
        "Crystallizer #1": "blue",
        "Crystallizer #2": "green",
        "Crystallizer #3": "orange"
    }

    fig = go.Figure()

    for c in crystallizers:
        cor = CORES[c["nome"]]

        df_mm = c["df"].copy().set_index('TIMESTAMP')
        df_mm[f'MM_{num_dias}D'] = df_mm["Resultado de Ferro (ppm)"].rolling(window=f'{num_dias}D').mean()
        df_mm = df_mm.reset_index()

        fig.add_trace(go.Scatter(
            x=df_mm['TIMESTAMP'],
            y=df_mm["Resultado de Ferro (ppm)"],
            mode='lines',
            name=f"{c['nome']} — original",
            line=dict(color=cor, width=1),
            opacity=0.3
        ))

        fig.add_trace(go.Scatter(
            x=df_mm['TIMESTAMP'],
            y=df_mm[f'MM_{num_dias}D'],
            mode='lines',
            name=f"{c['nome']} — MM {num_dias}D",
            line=dict(color=cor, width=2)
        ))

        for _, row in c["df_eventos"].iterrows():
            if not mostrar_falsos and row["Real"] == 0:
                continue

            dash_evento = "solid" if row["Real"] == 1 else "dash"
            fig.add_shape(
                type="line",
                x0=str(row["TIMESTAMP"]),
                x1=str(row["TIMESTAMP"]),
                y0=0, y1=1,
                yref="paper",
                line=dict(color=cor, width=1.5, dash=dash_evento)
            )
            fig.add_annotation(
                x=str(row["TIMESTAMP"]),
                y=1,
                yref="paper",
                text=row["EVENTO"] if "EVENTO" in c["df_eventos"].columns else "",
                showarrow=False,
                textangle=-90,
                yanchor="top",
                font=dict(color=cor)
            )

    fig.add_hline(y=5, line_width=3, line_dash="dash", line_color="red")
    fig.update_layout(
        template='plotly_white',
        hovermode='x unified',
        title=titulo or f"Fe (ppm) MM {num_dias}D — Crystallizers #1, #2 e #3"
    )
    return fig


def plot_crystallizer_unificado(df_med, df_eventos, titulo, mostrar_falsos=True):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_med['TIMESTAMP'],
        y=df_med["Resultado de Ferro (ppm)"],
        mode='lines',
        name="Resultado de Ferro (ppm)",
        line=dict(color='black')
    ))

    fig.add_hline(y=5, line_width=3, line_dash="dash", line_color="red")

    for _, row in df_eventos.iterrows():
        if not mostrar_falsos and row["Real"] == 0:
            continue

        cor = "red" if row["Real"] == 1 else "blue"
        fig.add_shape(
            type="line",
            x0=str(row["TIMESTAMP"]),
            x1=str(row["TIMESTAMP"]),
            y0=0, y1=1,
            yref="paper",
            line=dict(color=cor, width=1.5, dash="dash")
        )
        fig.add_annotation(
            x=str(row["TIMESTAMP"]),
            y=1,
            yref="paper",
            text=row["EVENTO"] if "EVENTO" in df_eventos.columns else "",
            showarrow=False,
            textangle=-90,
            yanchor="top",
            font=dict(color=cor)
        )

    fig.update_layout(
        template='plotly_white',
        hovermode='x unified',
        title=titulo
    )
    return fig

# =============================================================================
# Estatística e comparação entre as classes
# =============================================================================

def estatisticas(serie, nome):
    Q1 = serie.quantile(0.25)
    Q3 = serie.quantile(0.75)
    return pd.Series({
        "Contagem"     : serie.count(),
        "Média"        : serie.mean(),
        "Mediana"      : serie.median(),
        "Desvio Padrão": serie.std(),
        "Variância"    : serie.var(),
        "Mínimo"       : serie.min(),
        "Máximo"       : serie.max(),
        "Amplitude"    : serie.max() - serie.min(),
        "Q1 (25%)"     : Q1,
        "Q3 (75%)"     : Q3,
        "IQR"          : Q3 - Q1,
        "Assimetria"   : serie.skew(),
        "Curtose"      : serie.kurt()
    }, name=nome)


def hex_to_rgba(cor, alpha=0.15):
    rgb = pc.hex_to_rgb(cor)
    return f'rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha})'


def add_scatter_regressao(fig, x, y, row, col):
    lr = LinearRegression().fit(x.reshape(-1, 1), y)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = lr.predict(x_line.reshape(-1, 1))
    r2 = lr.score(x.reshape(-1, 1), y)
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers',
        marker=dict(size=6, opacity=0.6, color='steelblue'),
        showlegend=False
    ), row=row, col=col)
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line,
        mode='lines',
        line=dict(color='red', width=2),
        showlegend=False
    ), row=row, col=col)
    fig.add_annotation(
        x=x.min(), y=y.max(),
        text=f"R² = {r2:.3f}",
        showarrow=False,
        font=dict(size=12, color='red'),
        xanchor='left',
        row=row, col=col
    )


def plot_violin_classes(df_eventos, metodos, janelas, titulo,
                        coluna=COLUNA_FE, min_amostras=2):
    """Violin plot Real vs Falso Positivo por janela, um subplot por tratamento.

    metodos : lista de {"titulo": str, "df": DataFrame de medições}
    """
    n_metodos = len(metodos)
    fig = make_subplots(
        rows=n_metodos, cols=1,
        subplot_titles=[f"Real vs Falso Positivo — {m['titulo']}" for m in metodos],
        vertical_spacing=0.06
    )

    for row_idx, metodo in enumerate(metodos, start=1):
        df_atual = metodo["df"]

        for dias in janelas:
            for classe in [0, 1]:
                valores_classe = []

                for _, evento in df_eventos.iterrows():
                    if int(evento["Real"]) != classe:
                        continue
                    v = valores_na_janela(df_atual, evento["TIMESTAMP"], dias, coluna)
                    if len(v) >= min_amostras:
                        valores_classe.extend(v.tolist())

                fig.add_trace(go.Violin(
                    y=valores_classe,
                    x=[f"{dias}d"] * len(valores_classe),
                    name=NOMES_CLASSE[classe],
                    legendgroup=NOMES_CLASSE[classe],
                    showlegend=(row_idx == 1 and dias == janelas[0]),
                    side="negative" if classe == 0 else "positive",
                    line_color=CORES_CLASSE[classe],
                    meanline_visible=True,
                    points=False
                ), row=row_idx, col=1)

        fig.update_yaxes(title_text="Fe (ppm)", row=row_idx, col=1)
        fig.update_xaxes(title_text="Janela", row=row_idx, col=1)

    fig.update_layout(
        height=500 * n_metodos,
        template="plotly_white",
        violingap=0.05,
        violinmode="overlay",
        title=titulo
    )
    return fig


def plot_violin_serie_vs_ultrapassagem(df_eventos, metodos, janelas, titulo,
                                       valores_referencia, coluna=COLUNA_FE, min_amostras=2):
    """Violin plot: distribuição de referência vs janelas anteriores a qualquer ultrapassagem do LC.

    valores_referencia : lista de valores da classe "Série completa" (normalmente a série
                         inteira tratada com Hampel de 90 dias)
    df_eventos         : todos os eventos são usados aqui — Real=0 e Real=1
    """
    cores = {"Serie completa": "#4878CF", "Ultrapassagem LC": "#D65F5F"}

    n_metodos = len(metodos)
    fig = make_subplots(
        rows=n_metodos, cols=1,
        subplot_titles=[f"Série Completa vs Ultrapassagem LC — {m['titulo']}" for m in metodos],
        vertical_spacing=0.06
    )

    for row_idx, metodo in enumerate(metodos, start=1):
        df_atual = metodo["df"]

        for dias in janelas:
            # Classe 1: distribuição de referência
            fig.add_trace(go.Violin(
                y=valores_referencia,
                x=[f"{dias}d"] * len(valores_referencia),
                name="Serie completa",
                legendgroup="Serie completa",
                showlegend=(row_idx == 1 and dias == janelas[0]),
                side="negative",
                line_color=cores["Serie completa"],
                meanline_visible=True,
                points=False,
            ), row=row_idx, col=1)

            # Classe 2: janelas anteriores a QUALQUER ultrapassagem do LC
            valores_lc = []
            for _, evento in df_eventos.iterrows():
                v = valores_na_janela(df_atual, evento["TIMESTAMP"], dias, coluna)
                if len(v) >= min_amostras:
                    valores_lc.extend(v.tolist())

            fig.add_trace(go.Violin(
                y=valores_lc,
                x=[f"{dias}d"] * len(valores_lc),
                name="Ultrapassagem LC",
                legendgroup="Ultrapassagem LC",
                showlegend=(row_idx == 1 and dias == janelas[0]),
                side="positive",
                line_color=cores["Ultrapassagem LC"],
                meanline_visible=True,
                points=False,
            ), row=row_idx, col=1)

        fig.update_yaxes(title_text="Fe (ppm)", row=row_idx, col=1)
        fig.update_xaxes(
            title_text="Janela",
            range=[-0.5, len(janelas) - 0.5],  # Adiciona espaço nas extremidades
            row=row_idx, col=1
        )

    fig.update_layout(
        height=500 * n_metodos,
        template="plotly_white",
        violingap=0.05,
        violinmode="overlay",  # Garante que negative/positive formem um único violino
        title=titulo
    )
    return fig


def testar_classes_por_janela(df_medicoes, df_eventos, janelas,
                              coluna=COLUNA_FE, min_amostras=2):
    """Quantifica se a diferença entre as classes é significativa, janela a janela.

    Junta todos os valores brutos das janelas de cada classe e aplica Mann-Whitney
    (diferença de medianas), Kolmogorov-Smirnov (forma da distribuição) e o effect
    size rank-biserial.
    """
    resultados = []

    for dias in janelas:
        vals = {0: [], 1: []}

        for _, evento in df_eventos.iterrows():
            v = valores_na_janela(df_medicoes, evento["TIMESTAMP"], dias, coluna)
            if len(v) >= min_amostras:
                vals[int(evento["Real"])].extend(v.tolist())

        v0, v1 = np.array(vals[0]), np.array(vals[1])

        stat_mw, p_mw = mannwhitneyu(v0, v1, alternative='two-sided')
        stat_ks, p_ks = ks_2samp(v0, v1)
        effect = 1 - (2 * stat_mw) / (len(v0) * len(v1))

        resultados.append({
            'janela':      f"{dias}D",
            'p_mw':        round(p_mw, 4),
            'p_ks':        round(p_ks, 4),
            'effect_size': round(abs(effect), 4),
            'media_r0':    round(np.mean(v0), 3),
            'media_r1':    round(np.mean(v1), 3),
            'delta_media': round(np.mean(v1) - np.mean(v0), 3),
        })

    return pd.DataFrame(resultados)


def separabilidade_features(df_medicoes, df_eventos, janelas, titulo,
                            stats_funcs=None, top_n=30, coluna=COLUNA_FE, min_amostras=2):
    """Effect size de Mann-Whitney feature a feature (absolutas + relativas).

    Mostra diretamente qual estatística e qual janela têm maior poder discriminativo.
    Retorna (df_effect, fig).
    """
    if stats_funcs is None:
        stats_funcs = STATS_FUNCS_SEPARABILIDADE

    janela_base = max(janelas)  # Usada como referência de longo prazo

    # 1. Extração de todas as features (absolutas e relativas) por evento
    dados_por_classe = {0: [], 1: []}

    for _, evento in df_eventos.iterrows():
        ts     = evento["TIMESTAMP"]
        classe = int(evento["Real"])
        feat_evento = {}

        # A. Absolutas, para todas as janelas
        for dias in janelas:
            v = valores_na_janela(df_medicoes, ts, dias, coluna)
            if len(v) < min_amostras:
                for nome_stat in stats_funcs:
                    feat_evento[f"{nome_stat}_{dias}d"] = np.nan
            else:
                for nome_stat, func in stats_funcs.items():
                    feat_evento[f"{nome_stat}_{dias}d"] = float(func(v))

        # B. Relativas (variação), usando a janela_base como âncora
        for dias in janelas:
            if dias == janela_base:
                continue

            media_curta = feat_evento.get(f"media_{dias}d", np.nan)
            media_longa = feat_evento.get(f"media_{janela_base}d", np.nan)
            max_curta   = feat_evento.get(f"max_{dias}d", np.nan)
            std_curta   = feat_evento.get(f"std_{dias}d", np.nan)
            std_longa   = feat_evento.get(f"std_{janela_base}d", np.nan)

            # Evita divisão por zero ou uso de dados faltantes
            if pd.notna(media_curta) and pd.notna(media_longa):
                feat_evento[f'aceleracao_media_{dias}d_vs_{janela_base}d'] = media_curta / (media_longa + 0.001)
                feat_evento[f'velocidade_diff_{dias}d_vs_{janela_base}d']  = media_curta - media_longa
                feat_evento[f'pico_max_{dias}d_vs_media_{janela_base}d']   = max_curta / (media_longa + 0.001)
            else:
                feat_evento[f'aceleracao_media_{dias}d_vs_{janela_base}d'] = np.nan
                feat_evento[f'velocidade_diff_{dias}d_vs_{janela_base}d']  = np.nan
                feat_evento[f'pico_max_{dias}d_vs_media_{janela_base}d']   = np.nan

            if pd.notna(std_curta) and pd.notna(std_longa):
                feat_evento[f'volatilidade_std_{dias}d_vs_{janela_base}d'] = std_curta / (std_longa + 0.001)
            else:
                feat_evento[f'volatilidade_std_{dias}d_vs_{janela_base}d'] = np.nan

        dados_por_classe[classe].append(feat_evento)

    df_c0 = pd.DataFrame(dados_por_classe[0])
    df_c1 = pd.DataFrame(dados_por_classe[1])

    # 2. Teste de Mann-Whitney para cada feature
    registros = []
    for feature in df_c0.columns:
        v0 = df_c0[feature].dropna().values
        v1 = df_c1[feature].dropna().values

        if len(v0) < 2 or len(v1) < 2:
            continue

        stat_mw, p = mannwhitneyu(v0, v1, alternative='two-sided')
        effect = abs(1 - (2 * stat_mw) / (len(v0) * len(v1)))

        registros.append({
            'feature':     feature,
            'effect_size': round(effect, 4),
            'p_value':     round(p, 4)
        })

    df_effect = pd.DataFrame(registros).sort_values(by='effect_size', ascending=True)

    # 3. Visualização — Top N features para o gráfico não ficar esmagado
    df_plot = df_effect.tail(top_n)
    cores = ['#2ca02c' if 'vs' in feat else '#1f77b4' for feat in df_plot['feature']]

    fig = go.Figure(go.Bar(
        x=df_plot['effect_size'],
        y=df_plot['feature'],
        orientation='h',
        marker_color=cores,
        text=df_plot['effect_size'],
        textposition='outside'
    ))

    fig.update_layout(
        title=f"Top {top_n} Features por Effect Size (Mann-Whitney) {titulo}<br>"
              "<sup><span style='color:#2ca02c'>Verde = Relativas (Variação)</span> | "
              "<span style='color:#1f77b4'>Azul = Absolutas</span></sup>",
        template="plotly_white",
        xaxis_title="Effect Size (Rank-Biserial Correlation)",
        yaxis_title="Feature",
        height=800,
        margin=dict(l=250)  # Espaço extra para ler os nomes longos das features
    )

    return df_effect, fig

# =============================================================================
# Estrutura não supervisionada — janelas deslizantes, regimes e cross-reator
# =============================================================================
#
# O que existia aqui antes (extrair_features_clustering / pipeline_clustering /
# rodar_clusterizacao / plotar_clusterizacao_pca) foi removido. Três motivos, todos
# medidos sobre os próprios dados:
#
#   1. o algoritmo e o k eram escolhidos pelo ARI contra o rótulo `Real` — a
#      "clusterização não supervisionada" era selecionada com o rótulo na mão, e o
#      ARI publicado (0.217) era o MÁXIMO de 36 configurações, cuja mediana era 0.108;
#   2. pelo critério interno (silhueta), o melhor agrupamento colocava 98% dos pontos
#      em um único cluster — naquele espaço não havia estrutura natural nenhuma;
#   3. o que o agrupamento separava era a ÉPOCA, não a falha:
#      ARI(cluster, período >= 01/04/2020) = 0.593 contra ARI(cluster, Real) = 0.217.
#   + havia um bug no modo unificado: a janela era recortada sobre o dataframe
#     concatenado dos três reatores (182 amostras em vez das 65 do próprio reator).
#
# O que entra no lugar: janelas deslizantes de TODA a série, log1p + RobustScaler,
# k por silhueta e avaliação por lift contra taxa base.

# Features calculadas sobre a própria janela. COLUNAS_REGIME é o conjunto original,
# mantido fixo para que a análise de regimes continue reproduzível; as demais entram
# nos modelos (COLUNAS_MODELO_JANELA) sem mexer naquela conclusão.
COLUNAS_REGIME = ["mediana", "p90", "max", "std", "inclinacao", "frac_acima_lc",
                  "razao_mediana", "razao_max", "n_amostras", "maior_lacuna"]

COLUNAS_JANELA_EXTRA = ["assimetria", "curtose", "maior_seq_acima_base", "densidade_relativa",
                        "n_lacunas_3d", "dias_desde_amostra", "amplitude"]

COLUNAS_CONTEXTO = ["margem_max", "margem_media", "margem_inclinacao", "posto_medio",
                    "frac_lider", "razao_limite_movel", "ewma_z", "cusum_rel",
                    "idade_campanha", "faixa_campanha", "dias_desde_ultimo_reparo",
                    "n_reparos_na_campanha", "dias_desde_reparo_vidro",
                    "n_reparos_vidro_campanha", "n_falhas_anteriores"]

COLUNAS_MODELO_JANELA = COLUNAS_REGIME + COLUNAS_JANELA_EXTRA + COLUNAS_CONTEXTO

_DIA_NS = np.int64(86_400_000_000_000)

FAIXAS_CAMPANHA = [0, 365, 730, 1095, 1825, np.inf]
ROTULOS_CAMPANHA = ["<1 ano", "1-2 anos", "2-3 anos", "3-5 anos", ">5 anos"]


def _para_arrays(serie):
    """Series -> (índice em int64 ns, valores float), para fatiar janela com searchsorted."""
    s = serie.dropna()
    return s.index.values.astype("datetime64[ns]").astype("int64"), s.values.astype(float)


def _arrays_medicoes(df_medicoes, coluna=COLUNA_FE):
    d = df_medicoes.sort_values("TIMESTAMP")
    return (d["TIMESTAMP"].values.astype("datetime64[ns]").astype("int64"),
            d[coluna].values.astype(float))


def _ultimo_antes(arrays, t_ns):
    """Último valor de uma série auxiliar estritamente antes de `t_ns` (NaN se não há)."""
    ts, v = arrays
    i = np.searchsorted(ts, t_ns, "left")
    return float(v[i - 1]) if i > 0 else np.nan


def _features_janela(ts, v, t_ns, janela_dias=15, janela_base_dias=90,
                     min_amostras=8, min_amostras_base=30, lc=5.0):
    """Estatísticas de uma janela [t - janela_dias, t), com contexto de `janela_base_dias`.

    Devolve None quando a janela não tem suporte suficiente — decisão de projeto: um evento
    com 2 amostras em 15 dias não descreve operação, descreve a parada.

    Além das estatísticas clássicas, carrega quatro grupos acrescentados depois da avaliação
    da etapa não supervisionada:
      - forma da distribuição (`assimetria`, `curtose`): distingue "patamar deslocado" de
        "um pico dentro do normal", que média e mediana confundem;
      - persistência (`maior_seq_acima_base`): maior sequência consecutiva acima da base de
        90 dias — o teste direto de "impulso vs rampa", que é a dúvida central do estudo;
      - cadência de amostragem (`densidade_relativa`, `n_lacunas_3d`, `dias_desde_amostra`):
        a taxa varia de 5.4 a 10.3 amostras/dia ao longo dos anos e cai perto de parada;
      - `amplitude` (max - min), que separa janela agitada de janela alta e estável.
    """
    i0 = np.searchsorted(ts, t_ns - janela_dias * _DIA_NS, "left")
    i1 = np.searchsorted(ts, t_ns, "left")
    j0 = np.searchsorted(ts, t_ns - janela_base_dias * _DIA_NS, "left")
    w, w_base = v[i0:i1], v[j0:i1]
    if len(w) < min_amostras or len(w_base) < min_amostras_base:
        return None

    ts_jan = ts[i0:i1]
    dias = (ts_jan - (t_ns - janela_dias * _DIA_NS)) / _DIA_NS
    base = max(float(np.median(w_base)), 0.1)
    gaps = np.diff(ts_jan) / _DIA_NS if len(w) > 1 else np.array([0.0])

    # maior sequência consecutiva acima da base de 90 dias
    acima = w > base
    maior_seq = corrente = 0
    for a in acima:
        corrente = corrente + 1 if a else 0
        maior_seq = max(maior_seq, corrente)

    dens_janela = len(w) / janela_dias
    dens_base = len(w_base) / janela_base_dias

    return {
        "mediana":              float(np.median(w)),
        "p90":                  float(np.percentile(w, 90)),
        "max":                  float(np.max(w)),
        "std":                  float(np.std(w)),
        "inclinacao":           float(np.polyfit(dias, w, 1)[0]) if len(w) > 2 else 0.0,
        "frac_acima_lc":        float((w > lc).mean()),
        "base_longa":           base,
        "razao_mediana":        float(np.median(w)) / base,
        "razao_max":            float(np.max(w)) / base,
        "n_amostras":           int(len(w)),
        "maior_lacuna":         float(gaps.max()) if len(gaps) else 0.0,
        "assimetria":           float(skew(w)) if len(w) > 2 else 0.0,
        "curtose":              float(kurtosis(w)) if len(w) > 3 else 0.0,
        "maior_seq_acima_base": float(maior_seq),
        "densidade_relativa":   float(dens_janela / dens_base) if dens_base else np.nan,
        "n_lacunas_3d":         float((gaps > 3).sum()),
        "dias_desde_amostra":   float((t_ns - ts_jan[-1]) / _DIA_NS),
        "amplitude":            float(np.max(w) - np.min(w)),
    }


# -----------------------------------------------------------------------------
# Contexto: tudo o que a série do próprio reator não sabe
# -----------------------------------------------------------------------------

def preparar_contexto(map_medicoes, df_inspecoes=None, quantil=0.99, dias_baseline=365,
                      lambd=0.2, coluna=COLUNA_FE, verbose=True):
    """Monta de uma vez as séries auxiliares que alimentam features e modelos.

    Devolve um dicionário com:
      - `diario`, `comum`, `margem`, `posto` — decomposição cross-reator;
      - `baseline`  — quantil móvel de 365 d de cada reator (limite adaptativo);
      - `controle`  — EWMA (baseline rolante) e CUSUM dinâmico por amostra;
      - `campanhas`, `reparos`, `falhas` — histórico do equipamento.

    Passar um único `contexto` em vez de meia dúzia de mapas evita o erro de calcular
    uma feature com um recorte e outra com outro.
    """
    diario, comum, margem = calcular_margem_cross_reator(map_medicoes, coluna, verbose=verbose)
    ctx = {
        "diario": diario, "comum": comum, "margem": margem,
        "posto": posto_entre_reatores(diario),
        "baseline": baseline_movel(map_medicoes, quantil, dias_baseline, coluna=coluna),
        "controle": {}, "campanhas": {}, "reparos": {}, "falhas": {},
    }

    for cryst, df in map_medicoes.items():
        serie = df.set_index("TIMESTAMP")[coluna].dropna()
        ewma = calcular_ewma_rolante(serie, lambd=lambd)
        cusum = calcular_cusum_dinamico(serie, janela_baseline=30, k=0.5, h=4)
        ctx["controle"][cryst] = {
            "ewma_z": _para_arrays(ewma["Z"]),
            "cusum_rel": _para_arrays(
                (cusum["CUSUM_Positivo"] / cusum["Limite_H"].replace(0, np.nan)).fillna(0)),
        }

    if df_inspecoes is not None:
        ctx["campanhas"] = campanhas_por_reator(df_inspecoes)
        ctx["reparos_vidro"] = {}
        for cryst, g in df_inspecoes.groupby("Crystallizer"):
            ctx["reparos"][cryst] = sorted(pd.to_datetime(g["Inicio"].dropna()))
            sel = g[g["Selecionado"]] if "Selecionado" in g.columns else g.iloc[0:0]
            ctx["falhas"][cryst] = sorted(pd.to_datetime(sel["TS_Ajustado"].dropna()))
            # reparo no revestimento vitrificado: é o mecanismo de falha em si (o vaso falha
            # quando o vidro trinca e o costado é atacado), então o histórico de reparo de
            # vidro é a variável de degradação mais próxima da física do problema que existe
            # neste repositório — e vem de coluna estruturada, não de mineração de texto
            # ReparoVidro é texto na planilha ("Sim" / "Não" / vazio), não booleano
            if "ReparoVidro" in g.columns:
                marca = g["ReparoVidro"].astype(str).str.strip().str.lower().str.startswith("sim")
                vidro = g[marca]
            else:
                vidro = g.iloc[0:0]
            ctx["reparos_vidro"][cryst] = sorted(pd.to_datetime(vidro["Inicio"].dropna()))

    if verbose:
        print(f"\nContexto montado: {len(ctx['controle'])} reatores com EWMA/CUSUM, "
              f"campanhas de {list(ctx['campanhas'])}")
    return ctx


def _features_auxiliares(cryst, t, contexto=None, janela_dias=15):
    """Features de contexto no instante `t` — nenhuma usa informação posterior a `t`."""
    if contexto is None:
        return {}
    extra = {}
    t0 = t - pd.Timedelta(days=janela_dias)

    margem = contexto.get("margem")
    if margem is not None and cryst in margem.columns:
        s = margem[cryst].dropna()
        jan = s[(s.index >= t0) & (s.index < t)]
        extra["margem_max"] = float(jan.max()) if len(jan) else np.nan
        extra["margem_media"] = float(jan.mean()) if len(jan) else np.nan
        if len(jan) > 2:
            x = (jan.index - t0).days.values.astype(float)
            extra["margem_inclinacao"] = float(np.polyfit(x, jan.values, 1)[0])
        else:
            extra["margem_inclinacao"] = np.nan

    posto = contexto.get("posto")
    if posto is not None and cryst in posto.columns:
        s = posto[cryst].dropna()
        jan = s[(s.index >= t0) & (s.index < t)]
        extra["posto_medio"] = float(jan.mean()) if len(jan) else np.nan
        extra["frac_lider"] = float((jan == 1).mean()) if len(jan) else np.nan

    base = contexto.get("baseline")
    if base is not None and cryst in base:
        s = base[cryst].dropna()
        anteriores = s[s.index < t]
        limite = float(anteriores.iloc[-1]) if len(anteriores) else np.nan
        extra["limite_movel"] = limite

    ctrl = contexto.get("controle", {}).get(cryst)
    if ctrl is not None:
        extra["ewma_z"] = _ultimo_antes(ctrl["ewma_z"], t.value)
        extra["cusum_rel"] = _ultimo_antes(ctrl["cusum_rel"], t.value)

    campanhas = contexto.get("campanhas")
    if campanhas:
        idade = idade_campanha(campanhas, cryst, t)
        extra["idade_campanha"] = idade
        extra["faixa_campanha"] = faixa_campanha(idade)

    reparos = contexto.get("reparos", {}).get(cryst)
    if reparos:
        anteriores = [d for d in reparos if d < t]
        extra["dias_desde_ultimo_reparo"] = (float((t - max(anteriores)).days)
                                             if anteriores else np.nan)
        inicio_campanha = [d for d in campanhas.get(cryst, []) if d <= t] if campanhas else []
        marco = max(inicio_campanha) if inicio_campanha else None
        extra["n_reparos_na_campanha"] = float(
            len([d for d in anteriores if marco is None or d >= marco]))

    vidro = contexto.get("reparos_vidro", {}).get(cryst)
    if vidro is not None:
        anteriores = [d for d in vidro if d < t]
        extra["dias_desde_reparo_vidro"] = (float((t - max(anteriores)).days)
                                            if anteriores else np.nan)
        inicio_campanha = [d for d in campanhas.get(cryst, []) if d <= t] if campanhas else []
        marco = max(inicio_campanha) if inicio_campanha else None
        extra["n_reparos_vidro_campanha"] = float(
            len([d for d in anteriores if marco is None or d >= marco]))

    falhas_ant = contexto.get("falhas", {}).get(cryst)
    if falhas_ant is not None:
        extra["n_falhas_anteriores"] = float(len([d for d in falhas_ant if d < t]))

    return extra


def construir_janelas(map_medicoes, ancoras_por_reator, map_falhas=None, janela_dias=15,
                      janela_base_dias=90, min_amostras=8, min_amostras_base=30,
                      horizonte_dias=15, contexto=None, coluna=COLUNA_FE):
    """Monta a tabela de janelas a partir de âncoras explícitas (uma linha por âncora).

    `map_falhas` só marca `pre_falha` (a âncora é seguida de falha em até `horizonte_dias`).
    Esse rótulo é usado para AVALIAR — nunca para ajustar o não supervisionado.
    """
    arr = {c: _arrays_medicoes(df, coluna) for c, df in map_medicoes.items()}
    linhas = []
    for cryst, ancoras in ancoras_por_reator.items():
        ts, v = arr[cryst]
        falhas_ns = (map_falhas[cryst]["TIMESTAMP"].values.astype("datetime64[ns]").astype("int64")
                     if map_falhas is not None and cryst in map_falhas else np.array([], dtype="int64"))
        for t in ancoras:
            t_ns = pd.Timestamp(t).value
            f = _features_janela(ts, v, t_ns, janela_dias, janela_base_dias,
                                 min_amostras, min_amostras_base)
            if f is None:
                continue
            f["Crystallizer"] = cryst
            f["TIMESTAMP"] = pd.Timestamp(t)
            if len(falhas_ns):
                prox = falhas_ns[(falhas_ns > t_ns) & (falhas_ns <= t_ns + horizonte_dias * _DIA_NS)]
                f["pre_falha"] = int(len(prox) > 0)
                seguintes = falhas_ns[falhas_ns > t_ns]
                f["dias_ate_falha"] = ((seguintes.min() - t_ns) / _DIA_NS) if len(seguintes) else np.nan
            f.update(_features_auxiliares(cryst, pd.Timestamp(t), contexto, janela_dias))
            if "limite_movel" in f:
                lim = f.pop("limite_movel")
                f["razao_limite_movel"] = f["max"] / lim if lim and not np.isnan(lim) else np.nan
            linhas.append(f)

    meta = ["Crystallizer", "TIMESTAMP", "pre_falha", "dias_ate_falha"]
    df = pd.DataFrame(linhas)
    ordem = [c for c in meta if c in df.columns] + [c for c in df.columns if c not in meta]
    return df[ordem]


def gerar_janelas_deslizantes(map_medicoes, map_falhas=None, passo_dias=3, janela_dias=15,
                              janela_base_dias=90, min_amostras=8, min_amostras_base=30,
                              horizonte_dias=15, contexto=None, coluna=COLUNA_FE, verbose=True):
    """Varre a série inteira com uma âncora a cada `passo_dias` — a base do não supervisionado.

    É a diferença central em relação à versão antiga: aqui o objeto de estudo é a OPERAÇÃO
    (toda a série), não a tabela de eventos. Sem isso não existe regime a descobrir, só o
    rótulo a reencontrar — e não existe amostra suficiente para treinar nada.
    """
    ancoras = {}
    for cryst, df in map_medicoes.items():
        ts = df["TIMESTAMP"]
        inicio = ts.min() + pd.Timedelta(days=janela_base_dias + 30)
        ancoras[cryst] = pd.date_range(inicio, ts.max(), freq=f"{passo_dias}D")

    tentadas = sum(len(v) for v in ancoras.values())
    df = construir_janelas(map_medicoes, ancoras, map_falhas, janela_dias, janela_base_dias,
                           min_amostras, min_amostras_base, horizonte_dias, contexto, coluna)
    if verbose:
        print(f"Janelas deslizantes: {len(df)}  "
              f"(passo {passo_dias} d, janela {janela_dias} d, contexto {janela_base_dias} d)")
        print(f"  descartadas por suporte insuficiente: {tentadas - len(df)} de {tentadas} "
              f"(< {min_amostras} amostras na janela ou < {min_amostras_base} no contexto)")
        if "pre_falha" in df.columns:
            print(f"  janelas que antecedem falha em até {horizonte_dias} d: "
                  f"{int(df['pre_falha'].sum())}  (taxa base {df['pre_falha'].mean():.2%})")
    return df


def janelas_nas_falhas(map_medicoes, map_falhas, **kwargs):
    """Mesmas features, calculadas exatamente nas âncoras de falha."""
    ancoras = {c: list(df["TIMESTAMP"]) for c, df in map_falhas.items()}
    kwargs.pop("map_falhas", None)
    return construir_janelas(map_medicoes, ancoras, None, **kwargs)


def clusterizar_regimes(df_base, df_falhas=None, colunas=None, k_grid=range(2, 8),
                        random_state=RANDOM_STATE, verbose=True):
    """Agrupa as janelas em regimes de operação e mede o enriquecimento de cada regime.

    Decisões (e o porquê de cada uma):
      - log1p + RobustScaler: a série é assimétrica e tem excursões de 3 ordens de
        grandeza; sem isso a silhueta escolhe sempre o corte degenerado (99% x 1%);
      - k escolhido pela SILHUETA, critério interno — o rótulo entra só depois, para
        medir o lift de cada regime;
      - lift = (% das falhas no regime) / (% do tempo no regime). Lift 1 é o que se
        esperaria de um agrupamento sem informação.
    """
    colunas = colunas or [c for c in COLUNAS_REGIME if c in df_base.columns]
    X = np.log1p(df_base[colunas].values.clip(min=0))
    escala = RobustScaler().fit(X)
    Xs = escala.transform(X)

    Xf = None
    if df_falhas is not None and len(df_falhas):
        Xf = escala.transform(np.log1p(df_falhas[colunas].values.clip(min=0)))

    grid = []
    melhor = None
    for k in k_grid:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(Xs)
        sil = silhouette_score(Xs, km.labels_, sample_size=min(4000, len(Xs)),
                               random_state=random_state)
        linha = {"k": k, "silhueta": round(sil, 4),
                 "maior_cluster_%": round(100 * pd.Series(km.labels_).value_counts(normalize=True).max(), 1)}
        # o melhor lift alcançável em cada k entra na tabela de propósito: sem isso o
        # leitor não sabe se o k vencedor da silhueta escondeu um agrupamento útil
        if Xf is not None:
            lf = km.predict(Xf)
            perfis = [((lf == c).mean() / max((km.labels_ == c).mean(), 1e-9),
                       100 * (lf == c).mean(), 100 * (km.labels_ == c).mean())
                      for c in range(k)]
            lift_max, cob, tempo = max(perfis)
            linha["melhor_lift"] = round(lift_max, 2)
            linha["cobre_%_das_falhas"] = round(cob, 1)
            linha["em_%_do_tempo"] = round(tempo, 1)
            # o lift do cluster minúsculo é fácil e inútil; este é o lift do melhor
            # regime que cobre pelo menos 20% das falhas — o que teria uso operacional
            uteis = [p for p in perfis if p[1] >= 20]
            linha["lift_util(>=20%_falhas)"] = round(max(uteis)[0], 2) if uteis else np.nan
        grid.append(linha)
        if melhor is None or sil > melhor[1]:
            melhor = (k, sil, km)
    k, sil, km = melhor

    labels = km.labels_
    df_base = df_base.copy()
    df_base["Regime"] = labels
    labels_falha = km.predict(Xf) if Xf is not None else None

    linhas = []
    for cl in range(k):
        p_base = float((labels == cl).mean())
        linha = {"Regime": cl, "janelas": int((labels == cl).sum()),
                 "%_do_tempo": round(100 * p_base, 1)}
        if labels_falha is not None:
            p_falha = float((labels_falha == cl).mean())
            linha["%_das_falhas"] = round(100 * p_falha, 1)
            linha["lift"] = round(p_falha / p_base, 2) if p_base else np.nan
        for col in ["mediana", "max", "razao_max", "inclinacao", "n_amostras"]:
            if col in df_base.columns:
                linha[col] = round(float(df_base.loc[labels == cl, col].mean()), 2)
        linha["ano_medio"] = int(df_base.loc[labels == cl, "TIMESTAMP"].dt.year.mean())
        linhas.append(linha)
    resumo = pd.DataFrame(linhas)

    if verbose:
        print(pd.DataFrame(grid).to_string(index=False))
        print(f"\nk escolhido pela silhueta: {k}  (silhueta {sil:.3f})")
        print(resumo.to_string(index=False))
        if labels_falha is not None:
            print("\nLeitura: lift ~1 = regime sem informação sobre falha; o interessante é um "
                  "regime com lift alto que cubra uma fração relevante das falhas.")

    return {"modelo": km, "escala": escala, "colunas": colunas, "k": k, "silhueta": sil,
            "labels": labels, "labels_falha": labels_falha, "resumo": resumo,
            "grid": pd.DataFrame(grid), "df_base": df_base}


def plotar_regimes_pca(res, df_falhas=None, titulo="Regimes de operação"):
    """PCA 2D das janelas coloridas por regime, com as âncoras de falha sobrepostas."""
    Xs = res["escala"].transform(np.log1p(res["df_base"][res["colunas"]].values.clip(min=0)))
    pca = PCA(n_components=2).fit(Xs)
    P = pca.transform(Xs)
    var = pca.explained_variance_ratio_ * 100

    fig = go.Figure()
    for cl in sorted(set(res["labels"])):
        m = res["labels"] == cl
        fig.add_trace(go.Scattergl(
            x=P[m, 0], y=P[m, 1], mode="markers", name=f"Regime {cl} ({m.sum()})",
            marker=dict(size=4, opacity=0.5)))
    if df_falhas is not None and len(df_falhas):
        Pf = pca.transform(res["escala"].transform(
            np.log1p(df_falhas[res["colunas"]].values.clip(min=0))))
        fig.add_trace(go.Scatter(
            x=Pf[:, 0], y=Pf[:, 1], mode="markers", name="Janela antes da falha",
            marker=dict(size=11, symbol="x", color="black", line=dict(width=1))))

    fig.update_layout(title=f"{titulo} — PCA 2D (k={res['k']}, silhueta {res['silhueta']:.2f})",
                      xaxis_title=f"PC1 ({var[0]:.1f}%)", yaxis_title=f"PC2 ({var[1]:.1f}%)",
                      template="plotly_white", height=600)
    return fig


# -----------------------------------------------------------------------------
# Decomposição cross-reator: o que é da planta e o que é do reator
# -----------------------------------------------------------------------------

def serie_diaria(map_medicoes, estatistica="median", coluna=COLUNA_FE):
    """Uma coluna por reator com a estatística diária (median | max | mean)."""
    return pd.concat(
        {c: df.set_index("TIMESTAMP")[coluna].resample("D").agg(estatistica)
         for c, df in map_medicoes.items()}, axis=1)


def calcular_margem_cross_reator(map_medicoes, coluna=COLUNA_FE, verbose=True):
    """Separa o nível comum (planta) do excedente de cada reator.

    margem_i(t) = mediana diária do reator i − mediana dos três reatores no mesmo dia.

    Com três séries, a mediana é o valor do meio: a margem é positiva SÓ para o reator
    que está liderando, e o quanto ele lidera. É a variável mais informativa achada na
    etapa não supervisionada — lift de 5.3x (margem > 1.2) e 6.9x (margem > 1.6) nas 31
    âncoras de falha, contra 3.6x da mediana diária absoluta acima de 5 ppm.

    Cuidado de leitura: C1 e C2 têm mediana diária correlacionada em 0.76, mas o C3 é
    quase independente (0.11) — ele é de outro trem. O componente comum responde por
    apenas ~6% da variância, então "todo mundo subiu junto" é a exceção.
    """
    diario = serie_diaria(map_medicoes, "median", coluna)
    comum = diario.median(axis=1)
    margem = diario.sub(comum, axis=0)

    if verbose:
        print("Correlação entre as medianas diárias:")
        print(diario.corr().round(3).to_string())
        v_total = float(diario.var().mean())
        print(f"\nVariância média da mediana diária : {v_total:.3f}")
        print(f"  componente comum (planta)       : {comum.var():.3f} "
              f"({100 * comum.var() / v_total:.0f}%)")
        print(f"  margem (específica do reator)   : {margem.var().mean():.3f} "
              f"({100 * margem.var().mean() / v_total:.0f}%)")

    return diario, comum, margem


def posto_entre_reatores(diario):
    """Posição do reator no dia (1 = mais alto). Versão categórica da margem.

    A margem responde "quanto acima"; o posto responde "é o líder?". As duas juntas
    separam uma liderança pequena e persistente de um pico isolado grande.
    """
    return diario.rank(axis=1, ascending=False, method="min")


def baseline_movel(map_medicoes, quantil=0.99, dias=365, min_periodos=60, coluna=COLUNA_FE):
    """Limite adaptativo: quantil móvel da própria série diária de cada reator.

    Existe por causa do drift documentado (mediana anual ~2.7 -> ~1.8 ppm): um limite
    fixo fica frouxo no regime antigo e apertado no atual. Como detector, o quantil
    móvel de 0.99 em 365 d entrega F2 0.239 contra 0.230 do limite fixo de 3.5 ppm,
    com menos falsos positivos (95 contra 104) — e não precisa ser recalibrado.
    """
    diario = serie_diaria(map_medicoes, "median", coluna)
    return {c: diario[c].dropna().rolling(dias, min_periods=min_periodos).quantile(quantil)
            for c in diario.columns}


def avaliar_lift_series(map_series, df_ancoras_base, df_ancoras_falha, limiares,
                        rotulo="série", janela_dias=15, min_pontos=3):
    """Lift de 'a série passou de X em algum dia da janela' — âncoras de falha vs base.

    Serve para qualquer série diária (absoluta, margem, resíduo), o que permite comparar
    critérios de naturezas diferentes na mesma régua.
    """
    linhas = []
    for th in limiares:
        nf = tf = nb = tb = 0
        for cryst, s in map_series.items():
            s = s.dropna()
            idx = s.index
            for nome, df_anc in (("falha", df_ancoras_falha), ("base", df_ancoras_base)):
                sel = df_anc[df_anc["Crystallizer"] == cryst]["TIMESTAMP"]
                for t in sel:
                    w = s[(idx >= t - pd.Timedelta(days=janela_dias)) & (idx < t)]
                    if len(w) < min_pontos:
                        continue
                    disparo = bool((w > th).any())
                    if nome == "falha":
                        tf += 1; nf += disparo
                    else:
                        tb += 1; nb += disparo
        pf = nf / tf if tf else np.nan
        pb = nb / tb if tb else np.nan
        linhas.append({"criterio": f"{rotulo} > {th}", "detectadas": f"{nf}/{tf}",
                       "nas_falhas": round(pf, 3), "taxa_base": round(pb, 3),
                       "lift": round(pf / pb, 2) if pb else np.nan})
    return pd.DataFrame(linhas)


# -----------------------------------------------------------------------------
# Idade de campanha (tempo desde a última troca do reator)
# -----------------------------------------------------------------------------

def campanhas_por_reator(df_inspecoes, fundir_dias=7):
    """{crystallizer: [datas de troca]} a partir da planilha de inspeção.

    Trocas a menos de `fundir_dias` de distância contam como uma só — a planilha
    registra a mesma troca em duas linhas (C1 26/06/2013 aparece duas vezes;
    C1 18 e 21/07/2017 são o evento e a inspeção pós-troca).
    """
    trocas = df_inspecoes[df_inspecoes["TrocaDoReator"]]
    campanhas = {}
    for c, g in trocas.groupby("Crystallizer"):
        datas, dedup = sorted(pd.to_datetime(g["Inicio"].dropna())), []
        for d in datas:
            if not dedup or (d - dedup[-1]).days > fundir_dias:
                dedup.append(d)
        campanhas[c] = dedup
    return campanhas


def idade_campanha(campanhas, cryst, ts):
    """Dias desde a última troca do reator antes de `ts` (NaN se não há troca conhecida)."""
    anteriores = [d for d in campanhas.get(cryst, []) if d <= ts]
    return float((ts - max(anteriores)).days) if anteriores else np.nan


def faixa_campanha(dias):
    """Faixa ordinal da idade de campanha (0 = <1 ano ... 4 = >5 anos)."""
    if dias is None or (isinstance(dias, float) and np.isnan(dias)):
        return np.nan
    return float(np.digitize(dias, FAIXAS_CAMPANHA[1:-1], right=False))


def perfil_hazard_campanha(map_medicoes, map_falhas, campanhas, verbose=True):
    """Falhas por 1000 reator-dias em cada faixa de idade de campanha.

    Normalizar pela EXPOSIÇÃO é o que muda a conclusão: contar só as falhas sugere
    "quanto mais velho, pior", mas dividindo pelo tempo que cada reator passou em cada
    faixa o risco tem pico em 1-2 anos (3.9 por 1000 reator-dias) e cai para 0.7 acima
    de 5 anos. Ou seja, a idade entra no modelo como FAIXA, não como variável linear.
    """
    expo = np.zeros(len(ROTULOS_CAMPANHA))
    falh = np.zeros(len(ROTULOS_CAMPANHA))

    for cryst, df in map_medicoes.items():
        for d in pd.date_range(df["TIMESTAMP"].min(), df["TIMESTAMP"].max(), freq="D"):
            f = faixa_campanha(idade_campanha(campanhas, cryst, d))
            if not np.isnan(f):
                expo[int(f)] += 1
    idades = []
    for cryst, df_ev in map_falhas.items():
        for ts in df_ev["TIMESTAMP"]:
            i = idade_campanha(campanhas, cryst, ts)
            f = faixa_campanha(i)
            if not np.isnan(f):
                falh[int(f)] += 1
                idades.append(i)

    tab = pd.DataFrame({"faixa": ROTULOS_CAMPANHA, "falhas": falh.astype(int),
                        "reator_dias": expo.astype(int)})
    tab["falhas_por_1000_reator_dias"] = (1000 * tab["falhas"] /
                                          tab["reator_dias"].replace(0, np.nan)).round(3)
    if verbose and idades:
        print(f"Idade de campanha na falha: mediana {np.median(idades):.0f} dias "
              f"(p25 {np.percentile(idades, 25):.0f}, p75 {np.percentile(idades, 75):.0f})")
    return tab


def _auc_rank(x, y):
    """AUC de Mann-Whitney a partir dos postos (sem sklearn, tolerante a empate)."""
    m = ~np.isnan(x)
    n1 = int((y[m] == 1).sum()); n0 = int((y[m] == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan, m.sum()
    ranks = pd.Series(x[m]).rank().values
    return (ranks[y[m] == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0), int(m.sum())


def avaliar_valor_features(df, colunas=None, alvo="pre_falha", min_validos=100,
                           anos_por_bloco=3, coluna_ts="TIMESTAMP"):
    """Valor individual de cada feature: AUC bruta e AUC ajustada por época.

    Roda sobre as ~5000 janelas (e não sobre os ~60 eventos), que é onde existe amostra
    para a medida significar alguma coisa. AUC 0.5 = a feature não separa nada; abaixo de
    0.5 significa que ela separa na direção contrária (o que também é informação).

    A coluna `AUC_ajustada` existe por causa de um problema que domina este conjunto:
    **28 das 31 falhas acontecem antes de 2019**. Qualquer variável correlacionada com o
    tempo (idade de campanha, número de falhas anteriores, densidade de amostragem, que
    cresceu ao longo dos anos) aparece como preditiva sem prever nada — está apenas
    dizendo "esta janela é antiga". A AUC ajustada recalcula a separação DENTRO de cada
    bloco de `anos_por_bloco` anos e faz a média ponderada pelo número de positivos:
    quando a AUC bruta se afasta de 0.5 e a ajustada volta para perto, a feature era um
    relógio disfarçado.
    """
    colunas = colunas or [c for c in COLUNAS_MODELO_JANELA if c in df.columns]
    y = df[alvo].values.astype(int)
    bloco = (df[coluna_ts].dt.year // anos_por_bloco) if coluna_ts in df.columns else None

    linhas = []
    for col in colunas:
        x = df[col].values.astype(float)
        auc, n_validos = _auc_rank(x, y)
        if np.isnan(auc) or n_validos < min_validos:
            continue

        auc_aj = np.nan
        if bloco is not None:
            pesos, aucs = [], []
            for b, idx in df.groupby(bloco).groups.items():
                pos = df.loc[idx, alvo].sum()
                if pos < 2 or len(idx) - pos < 20:
                    continue
                a, _ = _auc_rank(x[df.index.get_indexer(idx)], y[df.index.get_indexer(idx)])
                if not np.isnan(a):
                    aucs.append(a); pesos.append(pos)
            if aucs:
                auc_aj = float(np.average(aucs, weights=pesos))

        linhas.append({"feature": col, "AUC": round(auc, 3),
                       "|AUC-0.5|": round(abs(auc - 0.5), 3),
                       "AUC_ajustada": round(auc_aj, 3) if auc_aj == auc_aj else np.nan,
                       "|AUCaj-0.5|": round(abs(auc_aj - 0.5), 3) if auc_aj == auc_aj else np.nan,
                       "media_pre_falha": round(float(np.nanmean(x[y == 1])), 3),
                       "media_base": round(float(np.nanmean(x[y == 0])), 3),
                       "n_validos": n_validos})
    df_out = pd.DataFrame(linhas)
    ordem = "|AUCaj-0.5|" if df_out["|AUCaj-0.5|"].notna().any() else "|AUC-0.5|"
    return df_out.sort_values(ordem, ascending=False).reset_index(drop=True)


def perfil_temporal_falhas(map_falhas, map_medicoes, anos_por_bloco=3, verbose=True):
    """Distribuição das falhas ao longo do tempo — o confundidor central deste conjunto.

    Se as falhas se concentram em um período, qualquer feature que também mude com o tempo
    vira preditora espúria, e a validação temporal fica sem poder: treinar no passado e
    testar no futuro significa testar com pouquíssimos positivos.
    """
    linhas = []
    for cryst, df_ev in map_falhas.items():
        for ts in df_ev["TIMESTAMP"]:
            linhas.append({"Crystallizer": cryst, "ano": ts.year})
    df = pd.DataFrame(linhas)
    df["bloco"] = (df["ano"] // anos_por_bloco) * anos_por_bloco

    expo = {}
    for cryst, med in map_medicoes.items():
        dias = pd.date_range(med["TIMESTAMP"].min(), med["TIMESTAMP"].max(), freq="D")
        for d in dias:
            b = (d.year // anos_por_bloco) * anos_por_bloco
            expo[b] = expo.get(b, 0) + 1

    tab = (df.groupby("bloco").size().rename("falhas").to_frame()
           .join(pd.Series(expo, name="reator_dias")).fillna(0))
    tab["falhas_por_1000_reator_dias"] = (1000 * tab["falhas"] /
                                          tab["reator_dias"].replace(0, np.nan)).round(3)
    tab.index = [f"{b}-{b + anos_por_bloco - 1}" for b in tab.index]
    if verbose:
        print(tab.to_string())
        print(f"\nFalhas até 2018: {(df['ano'] <= 2018).sum()} de {len(df)}  "
              f"({100 * (df['ano'] <= 2018).mean():.0f}%)")
        print("Consequência: a validação temporal testa com pouquíssimos positivos, e "
              "features correlacionadas com o tempo viram preditoras espúrias — veja a "
              "coluna AUC_ajustada em avaliar_valor_features.")
    return tab


# =============================================================================
# Classificação (supervisionada)
# =============================================================================

def _features_contexto(cryst, ts, df_med, janela_base, janela_curta, contexto=None):
    """Features do evento que não vêm da própria série — mesmas das janelas deslizantes.

    Compartilhar `_features_auxiliares` com `construir_janelas` é intencional: o detector
    treinado sobre janelas e o classificador treinado sobre eventos passam a ver o mesmo
    espaço de features, o que torna os dois resultados comparáveis.

    O que cada grupo acrescenta, e por que foi escolhido:
      - `margem_*` / `posto_medio` / `frac_lider` — decomposição cross-reator. Lift de
        5.3x a 6.9x nas 31 âncoras de falha, contra 3.6x da mediana diária absoluta acima
        de 5 ppm. A margem diz "quanto acima dos outros", o posto diz "é o líder";
      - `razao_limite_movel` — pico da janela sobre o quantil 0.99 dos últimos 365 dias do
        próprio reator; imuniza contra o drift (mediana anual ~2.7 -> ~1.8 ppm);
      - `ewma_z` / `cusum_rel` — as cartas de controle viravam gráfico e paravam ali;
        aqui a estatística de controle no instante da âncora vira FEATURE, que é onde ela
        rende: as duas acumulam desvio pequeno e persistente, que max e mediana ignoram;
      - `idade_campanha` / `faixa_campanha` / `dias_desde_ultimo_reparo` /
        `n_reparos_na_campanha` / `n_falhas_anteriores` — histórico do equipamento, que a
        série de ferro não tem como saber. O risco tem pico em 1-2 anos de campanha;
      - `n_amostras_*` / `maior_lacuna_*` — qualidade da janela virando variável em vez de
        viés silencioso (as janelas variam de 8 a 150 amostras).

    Nada aqui usa informação posterior a `ts`.
    """
    extra = {}

    jan_base = df_med[(df_med["TIMESTAMP"] >= ts - pd.Timedelta(days=janela_base)) &
                      (df_med["TIMESTAMP"] < ts)]
    tsel = jan_base["TIMESTAMP"].sort_values()
    extra[f"n_amostras_{janela_base}d"] = float(len(jan_base))
    extra[f"maior_lacuna_{janela_base}d"] = (
        float(tsel.diff().dt.total_seconds().max() / 86400) if len(tsel) > 1 else np.nan)
    extra[f"n_amostras_{janela_curta}d"] = float(len(
        df_med[(df_med["TIMESTAMP"] >= ts - pd.Timedelta(days=janela_curta)) &
               (df_med["TIMESTAMP"] < ts)]))

    aux = _features_auxiliares(cryst, ts, contexto, janela_base)
    if "limite_movel" in aux:
        limite = aux.pop("limite_movel")
        vals = valores_na_janela(df_med, ts, janela_base)
        aux["razao_limite_movel"] = (float(np.max(vals)) / limite
                                     if len(vals) and limite and not np.isnan(limite) else np.nan)
    extra.update(aux)
    return extra


def extrair_features(crystallizer, df_medicoes, df_eventos, janelas, stats_funcs,
                     map_medicoes_unif=None, contexto=None):
    dataset_linhas = []
    janela_base = max(janelas)
    colunas_meta = ['Crystallizer', 'TIMESTAMP_Evento', 'Real']
    
    for _, evento in df_eventos.iterrows():
        ts     = evento["TIMESTAMP"]
        classe = int(evento["Real"])
        cryst  = evento.get("Crystallizer", crystallizer)
        df_med = map_medicoes_unif[cryst] if map_medicoes_unif else df_medicoes

        features = {'Crystallizer': cryst, 'TIMESTAMP_Evento': ts, 'Real': classe}
        
        for dias in janelas:
            y_ppm = valores_na_janela(df_med, ts, dias)
            
            if len(y_ppm) < 2:
                for nome_stat in stats_funcs:
                    features[f"{nome_stat}_{dias}d"] = np.nan
                continue
                
            for nome_stat, func in stats_funcs.items():
                features[f"{nome_stat}_{dias}d"] = float(func(y_ppm))

        if pd.isna(features.get(f"media_{janela_base}d", np.nan)):
            continue
            
        for dias in janelas:
            if dias == janela_base:
                continue
                
            features[f'acel_media_{dias}d_vs_{janela_base}d'] = features[f'media_{dias}d'] / (features[f'media_{janela_base}d'] + 0.001)
            features[f'vel_diff_{dias}d_vs_{janela_base}d'] = features[f'media_{dias}d'] - features[f'media_{janela_base}d']
            features[f'pico_max_{dias}d_vs_media_{janela_base}d'] = features[f'max_{dias}d'] / (features[f'media_{janela_base}d'] + 0.001)
            features[f'volatilidade_std_{dias}d_vs_{janela_base}d'] = features[f'std_{dias}d'] / (features[f'std_{janela_base}d'] + 0.001)

        janela_curta = min(janelas)
        chaves_para_remover = [k for k in features.keys() if ('d' in k and 'vs' not in k and f'_{janela_curta}d' not in k)]
        for k in chaves_para_remover:
            del features[k]

        # Acrescentadas DEPOIS da poda acima de propósito: são features de contexto
        # (margem cross-reator, limite móvel, idade de campanha, suporte da janela) e
        # não seguem a convenção `{stat}_{dias}d` que a poda usa.
        features.update(_features_contexto(cryst, ts, df_med, janela_base, janela_curta,
                                           contexto))

        dataset_linhas.append(features)

    if not dataset_linhas:
        return pd.DataFrame(columns=colunas_meta), []

    df = pd.DataFrame(dataset_linhas)
    colunas_features = [c for c in df.columns if c not in colunas_meta]
    return df[colunas_meta + colunas_features], colunas_features


def dividir_dados(df_features, colunas_features, test_size=TEST_SIZE, incluir_crystallizer=False):
    X = df_features.sort_values('TIMESTAMP_Evento').copy()
    
    if incluir_crystallizer:
        le = LabelEncoder()
        X['Crystallizer_enc'] = le.fit_transform(X['Crystallizer'])
        
    y = X['Real']
    eventos_pos = X[X['Real'] == 1]
    
    if len(eventos_pos) >= 2:
        idx_corte_pos = int(len(eventos_pos) * (1 - test_size))
        ts_corte = eventos_pos.iloc[idx_corte_pos]['TIMESTAMP_Evento']
        train_mask = X['TIMESTAMP_Evento'] < ts_corte
        test_mask  = X['TIMESTAMP_Evento'] >= ts_corte
    else:
        split_idx = int(len(X) * (1 - test_size))
        train_mask = np.arange(len(X)) < split_idx
        test_mask  = np.arange(len(X)) >= split_idx

    X_train = X[train_mask][colunas_features]
    X_test  = X[test_mask][colunas_features]
    y_train = y[train_mask]
    y_test  = y[test_mask]
    
    return X_train, X_test, y_train, y_test, X[colunas_features], y


def selecionar_features(X_train, y_train, colunas_brutas, k_features=8, limite_correlacao=0.85):
    X = X_train[colunas_brutas].copy()
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median())
        
    matriz_corr = X.corr(method='spearman').abs()
    upper = matriz_corr.where(np.triu(np.ones(matriz_corr.shape), k=1).astype(bool))
    colunas_para_dropar = [column for column in upper.columns if any(upper[column] > limite_correlacao)]
    X_sem_corr = X.drop(columns=colunas_para_dropar)
    
    n_features_disponiveis = X_sem_corr.shape[1]
    k_final = min(k_features, n_features_disponiveis)
    
    if k_final == 0: return []
        
    seletor = SelectKBest(score_func=f_classif, k=k_final)
    seletor.fit(X_sem_corr, y_train)
    features_selecionadas = X_sem_corr.columns[seletor.get_support()].tolist()
    return features_selecionadas


def definir_modelos_e_grids(y_train):
    ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    return {
        'Logistic Regression': (
            LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE),
            {'clf__C': [0.01, 0.1, 1.0]}
        ),
        'Random Forest': (
            RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE),
            {'clf__n_estimators': [100, 300], 'clf__max_depth': [3, 5, 7]}
        ),
        'XGBoost': (
            XGBClassifier(scale_pos_weight=ratio, eval_metric='aucpr', random_state=RANDOM_STATE, verbosity=0),
            {'clf__n_estimators': [100, 200], 'clf__max_depth': [3, 5], 'clf__learning_rate': [0.01, 0.05]}
        ),
        'SVM RBF': (
            SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=RANDOM_STATE),
            {'clf__C': [0.1, 1.0, 10.0]}
        ),
    }


def avaliar_cv_com_grid(X_train, y_train, modelos_grids, usar_smote, n_splits=N_SPLITS):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=False)
    scoring = {
        'f1':        make_scorer(f1_score, zero_division=0),
        'pr_auc':    'average_precision',
        'roc_auc':   'roc_auc',
        'recall':    make_scorer(recall_score, zero_division=0),
        'precision': make_scorer(precision_score, zero_division=0),
    }
    resultados, melhores_estimadores = [], {}

    # O SMOTE precisa dos vizinhos DENTRO de cada fold, não no treino inteiro:
    # com poucos positivos um fold pode ficar com 2 ou 3 falhas e o k estourar.
    n_falhas_treino = int(sum(y_train == 1))
    n_falhas_fold   = int(np.floor(n_falhas_treino * (n_splits - 1) / n_splits))
    k_vizinhos      = max(1, min(3, n_falhas_fold - 1))

    if usar_smote and n_falhas_fold < 2:
        print(f"  -> AVISO: {n_falhas_treino} falhas no treino ({n_falhas_fold} por fold) — "
              f"poucas para o SMOTE. Rodando sem oversampling.")
        usar_smote = False


    for nome, (modelo, grid) in modelos_grids.items():
        if usar_smote:
            pipe = ImbPipeline([
                ('imputer', SimpleImputer(strategy='median')), 
                ('scaler', StandardScaler()), 
                ('smote', SMOTE(k_neighbors=k_vizinhos, random_state=RANDOM_STATE)),
                ('clf', modelo)
            ])
        else:
            pipe = SklearnPipeline([
                ('imputer', SimpleImputer(strategy='median')), 
                ('scaler', StandardScaler()), 
                ('clf', modelo)
            ])
            
        gs = GridSearchCV(pipe, param_grid=grid, cv=cv, scoring=scoring, refit='pr_auc', n_jobs=-1)
        gs.fit(X_train, y_train)
        
        melhores_estimadores[nome] = gs.best_estimator_
        idx_best = gs.best_index_
        res_cv = gs.cv_results_
        
        resultados.append({
            'Modelo':    nome,
            'F1':        round(res_cv['mean_test_f1'][idx_best], 4),
            'PR_AUC':    round(res_cv['mean_test_pr_auc'][idx_best], 4),
            'ROC_AUC':   round(res_cv['mean_test_roc_auc'][idx_best], 4),
            'Recall':    round(res_cv['mean_test_recall'][idx_best], 4),
            'Precision': round(res_cv['mean_test_precision'][idx_best], 4),
            'Melhor_Params': str(gs.best_params_)
        })
        
    return pd.DataFrame(resultados).sort_values('PR_AUC', ascending=False), melhores_estimadores


def avaliar_teste(melhor_pipe, X_test, y_test, best_thr=0.4):
    probs_test = melhor_pipe.predict_proba(X_test)[:, 1]
    y_pred_test = (probs_test >= best_thr).astype(int)
    return melhor_pipe, best_thr, probs_test, y_pred_test


def plotar_resultados(resultados_por_tratamento, modo, df_cv_consolidado):
    tratamentos = list(resultados_por_tratamento.keys())
    n = len(tratamentos)

    fig_cm = make_subplots(
        rows=1, cols=n,
        subplot_titles=[f"{modo} — {t}" for t in tratamentos]
    )

    for col_idx, tratamento in enumerate(tratamentos, start=1):
        r = resultados_por_tratamento[tratamento]
        cm = confusion_matrix(r["y_test"], r["y_pred_test"])

        labels = ["Falso Positivo", "Contaminação"]
        fig_cm.add_trace(go.Heatmap(
            z=cm, x=labels, y=labels, colorscale="Blues",
            showscale=(col_idx == n), text=cm, texttemplate="%{text}", textfont=dict(size=14)
        ), row=1, col=col_idx)

        fig_cm.update_xaxes(title_text="Predito",  row=1, col=col_idx)
        fig_cm.update_yaxes(title_text="Real",     row=1, col=col_idx)

    fig_cm.update_layout(height=400, template="plotly_white", title=f"Matriz de Confusão — {modo} (Pipeline Completo)")
    fig_cm.show()


def rodar_classificacao(modo, df_med, df_ev, janelas, tratamento_alvo="",
                        stats_funcs=None, n_splits=N_SPLITS, test_size=TEST_SIZE,
                        k_features=10, best_thr=0.4, plotar=True,
                        map_medicoes_unif=None, contexto=None):
    """Compara as quatro abordagens (base / SMOTE / feature selection / ambos).

    `map_margem`, `map_baseline` e `campanhas` ligam as features de contexto vindas da
    etapa não supervisionada (ver `_features_contexto`). Quando não são passados, o
    conjunto de features é o antigo — útil para medir o ganho das novas.

    Retorna (df_relatorio, resultados_para_plot).
    """
    if stats_funcs is None:
        stats_funcs = STATS_FUNCS

    print(f"\nIniciando testes comparativos no {modo} usando tratamento {tratamento_alvo}\n")

    df_feat, cols_feat_brutas = extrair_features(
        modo, df_med, df_ev, janelas, stats_funcs,
        map_medicoes_unif=map_medicoes_unif, contexto=contexto)
    print(f"Features por evento: {len(cols_feat_brutas)}  |  eventos: {len(df_feat)}")

    abordagens = [
        {"nome": "Base (Sem SMOTE, Sem FS)",  "smote": False, "fs": False},
        {"nome": "Apenas SMOTE",              "smote": True,  "fs": False},
        {"nome": "Apenas Feature Selection",  "smote": False, "fs": True},
        {"nome": "SMOTE + Feature Selection", "smote": True,  "fs": True},
    ]

    relatorio_final      = []
    resultados_para_plot = {}

    if df_feat.empty or len(df_feat) < n_splits + 2:
        print("Dados insuficientes.")
        return pd.DataFrame(), {}

    for abordagem in abordagens:
        nome_ab = abordagem['nome']
        print(f"\n{'='*55}")
        print(f"Executando: {nome_ab}")
        print(f"{'='*55}")

        X_train_b, X_test_b, y_train, y_test, _, _ = dividir_dados(
            df_feat, cols_feat_brutas, test_size=test_size
        )

        if abordagem['fs']:
            cols_ativas = selecionar_features(X_train_b, y_train, cols_feat_brutas, k_features=k_features)
            if not cols_ativas:
                print(f"  -> AVISO: Nenhuma feature sobreviveu à seleção em {nome_ab}. Pulando.")
                continue
        else:
            cols_ativas = cols_feat_brutas

        X_train, X_test = X_train_b[cols_ativas], X_test_b[cols_ativas]

        modelos_grids = definir_modelos_e_grids(y_train)
        df_cv, melhores_estimadores = avaliar_cv_com_grid(
            X_train, y_train, modelos_grids, usar_smote=abordagem['smote'], n_splits=n_splits
        )

        melhor_nome = df_cv.iloc[0]['Modelo']
        melhor_pipe = melhores_estimadores[melhor_nome]

        pipe, thr, probs, y_pred = avaliar_teste(melhor_pipe, X_test, y_test, best_thr=best_thr)

        relatorio_final.append({
            'Abordagem':          nome_ab,
            'Melhor Modelo':      melhor_nome,
            'Features Usadas':    len(cols_ativas),
            'PR_AUC (Treino CV)': df_cv.iloc[0]['PR_AUC'],
            'Recall (Teste)':     recall_score(y_test, y_pred, zero_division=0),
            'Precisão (Teste)':   precision_score(y_test, y_pred, zero_division=0),
            'F1 (Teste)':         f1_score(y_test, y_pred, zero_division=0),
        })

        resultados_para_plot[nome_ab] = {'y_test': y_test, 'y_pred_test': y_pred}

        print(f"  Melhor Modelo: {melhor_nome}")
        print(f"  Relatório no Teste:")
        print(classification_report(y_test, y_pred, labels=[0, 1], zero_division=0))

    df_relatorio = pd.DataFrame(relatorio_final)

    if not df_relatorio.empty:
        print("\n" + "=" * 80)
        print("RELATÓRIO COMPARATIVO DE DESEMPENHO NO TESTE".center(80))
        print("=" * 80)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(df_relatorio.to_string(index=False))

        if plotar:
            plotar_resultados(resultados_para_plot, f"{modo} ({tratamento_alvo})", None)

    return df_relatorio, resultados_para_plot


def rodar_classificacao_por_tratamento(modo, configs, df_eventos, janelas, stats_funcs=None,
                                       test_size=TEST_SIZE, usar_smote=False, best_thr=0.4,
                                       incluir_crystallizer=False, plotar=True,
                                       map_medicoes_unif=None, contexto=None):
    """Compara os tratamentos de outlier dentro de um mesmo modo (um GridSearch por tratamento).

    Diferente de `rodar_classificacao`, que varre as quatro abordagens (SMOTE/FS) em um
    único tratamento, aqui varremos os tratamentos com uma abordagem fixa.

    configs : {nome_tratamento: df_medicoes}
    Retorna (resultados_por_tratamento, df_cv_consolidado).
    """
    if stats_funcs is None:
        stats_funcs = STATS_FUNCS

    resultados_por_tratamento = {}
    df_cv_todos = []

    for tratamento, df_med in configs.items():
        print(f"\n{'='*55}")
        print(f"{modo} | {tratamento}")
        print(f"{'='*55}")

        # `map_medicoes_unif` recorta a janela de cada evento na série do PRÓPRIO reator.
        # Sem ele (como era antes), a janela era recortada sobre o dataframe concatenado
        # dos três reatores: 182 amostras em vez das 65 do reator do evento, com max/p90
        # vindo de outro equipamento. Passe sempre o mapa no modo unificado.
        df_feat, cols_feat = extrair_features(
            crystallizer=modo,
            df_medicoes=df_med,
            df_eventos=df_eventos,
            janelas=janelas,
            stats_funcs=stats_funcs,
            map_medicoes_unif=map_medicoes_unif,
            contexto=contexto,
        )
        print(f"  Features por evento: {len(cols_feat)}")

        X_train, X_test, y_train, y_test, X_full, y_full = dividir_dados(
            df_feat, cols_feat, test_size=test_size, incluir_crystallizer=incluir_crystallizer
        )

        print(f"  Treino : {X_train.shape[0]} amostras "
              f"(Real=1: {y_train.sum()}  Real=0: {(y_train==0).sum()})")
        print(f"  Teste  : {X_test.shape[0]} amostras  "
              f"(Real=1: {y_test.sum()}  Real=0: {(y_test==0).sum()})")

        # Validação cruzada com GridSearch no treino
        modelos_grids = definir_modelos_e_grids(y_train)
        df_cv, melhores_estimadores = avaliar_cv_com_grid(
            X_train, y_train, modelos_grids, usar_smote=usar_smote
        )
        df_cv.insert(0, 'Tratamento', tratamento)
        df_cv_todos.append(df_cv)

        print("\n  CV (treino):")
        print(df_cv[['Modelo', 'F1', 'PR_AUC', 'ROC_AUC', 'Recall']].to_string(index=False))

        # Teste final usando o melhor estimador do GridSearch
        melhor_nome = df_cv.iloc[0]['Modelo']
        melhor_pipe = melhores_estimadores[melhor_nome]

        pipe, thr, probs_test, y_pred_test = avaliar_teste(
            melhor_pipe, X_test, y_test, best_thr=best_thr
        )

        resultados_por_tratamento[tratamento] = {
            'pipe':        pipe,
            'threshold':   thr,
            'probs_test':  probs_test,
            'y_test':      y_test,
            'y_pred_test': y_pred_test,
            'melhor':      melhor_nome,
            'X_train':     X_train,
            'X_test':      X_test,
            'y_train':     y_train,
        }

        print(f"\n  Melhor modelo : {melhor_nome}  |  threshold: {thr:.3f}")
        print(f"\n  Relatório no Teste:")
        print(classification_report(
            y_test, y_pred_test,
            labels=[0, 1],
            target_names=['Falso Positivo', 'Contaminação Real'],
            zero_division=0
        ))

    df_cv_consolidado = pd.concat(df_cv_todos, ignore_index=True)

    if plotar:
        plotar_resultados(resultados_por_tratamento, modo, df_cv_consolidado)

    return resultados_por_tratamento, df_cv_consolidado

# =============================================================================
# IsolationForest sobre janelas deslizantes (detecção de anomalia)
# =============================================================================
#
# O que existia aqui antes (otimizar_avaliar_iforest / rodar_iforest /
# plotar_resultados_iforest) foi removido. Ele treinava sobre as 59 linhas da tabela
# de eventos, com JANELAS = [6, 3, 1] e 4 estatísticas. Três problemas:
#
#   1. 59 linhas não sustentam um grid de 108 combinações — o "melhor F2 de treino"
#      que escolhia o modelo é ruído de seleção, e a matriz de confusão saía de 6 a 12
#      linhas de teste;
#   2. detectar anomalia SOBRE a tabela de eventos é circular: a tabela já foi montada
#      selecionando ultrapassagens, então "anômalo" e "evento" são quase sinônimos por
#      construção — o modelo não tinha como errar nem como aprender;
#   3. não havia comparação com taxa base, então não dava para saber se o alarme
#      significava alguma coisa.
#
# Agora o IsolationForest roda onde faz sentido: sobre TODA a operação (as janelas
# deslizantes), sem rótulo nenhum no ajuste, e é avaliado por lift contra a taxa base
# e pela mesma convenção de alarme dos demais detectores (agrupamento de 15 dias,
# acerto se a falha vem em até 15 dias).

def rodar_iforest_janelas(df_base, df_falhas, map_falhas=None, colunas=None,
                          contaminacao_grid=(0.01, 0.02, 0.05, 0.10), n_estimators=300,
                          random_state=RANDOM_STATE, verbose=True):
    """IsolationForest não supervisionado sobre as janelas deslizantes.

    O ajuste usa apenas `df_base` (a operação inteira, sem rótulo). As âncoras de falha
    entram só na avaliação: qual fração delas o modelo marca como anômala, contra a
    fração do tempo que ele marca no geral (= lift).
    """
    colunas = colunas or [c for c in COLUNAS_REGIME if c in df_base.columns]
    X = np.log1p(df_base[colunas].values.clip(min=0))
    escala = RobustScaler().fit(X)
    Xs = escala.transform(X)
    Xf = escala.transform(np.log1p(df_falhas[colunas].values.clip(min=0)))

    linhas, modelos, scores = [], {}, {}
    for cont in contaminacao_grid:
        mod = IsolationForest(n_estimators=n_estimators, contamination=cont,
                              random_state=random_state, n_jobs=-1).fit(Xs)
        anom_base = mod.predict(Xs) == -1
        anom_falha = mod.predict(Xf) == -1
        p_base = float(anom_base.mean())
        p_falha = float(anom_falha.mean())

        linha = {"contaminacao": cont,
                 "taxa_base": round(p_base, 4),
                 "falhas_sinalizadas": f"{int(anom_falha.sum())}/{len(anom_falha)}",
                 "%_falhas": round(100 * p_falha, 1),
                 "lift": round(p_falha / p_base, 2) if p_base else np.nan}

        if map_falhas is not None:
            alarmes = {}
            for cryst in df_base["Crystallizer"].unique():
                m = anom_base & (df_base["Crystallizer"] == cryst).values
                alarmes[cryst] = list(df_base.loc[m, "TIMESTAMP"])
            linha.update({k: v for k, v in
                          _metricas_alarmes(alarmes, map_falhas,
                                            rotulo=f"iforest c={cont}").items()
                          if k != "regra"})

        linhas.append(linha)
        modelos[cont] = mod
        scores[cont] = {"base": mod.score_samples(Xs), "falha": mod.score_samples(Xf)}

    tabela = pd.DataFrame(linhas)
    if verbose:
        print(tabela.to_string(index=False))
        print("\nLeitura: lift 1 significa que o modelo marca a janela que antecede falha "
              "com a mesma frequência com que marca uma janela qualquer — ou seja, o alarme "
              "não carrega informação. Compare a linha de F2 com a tabela de detectores do "
              "Baseline antes de considerar este caminho.")
    return {"tabela": tabela, "modelos": modelos, "escala": escala, "colunas": colunas,
            "scores": scores, "df_base": df_base, "df_falhas": df_falhas}


def plotar_iforest_janelas(res, contaminacao=None, titulo="IsolationForest — score de anomalia"):
    """Distribuição do score de anomalia: operação inteira vs janelas que antecedem falha."""
    contaminacao = contaminacao or list(res["scores"].keys())[0]
    s = res["scores"][contaminacao]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=s["base"], name="Todas as janelas", histnorm="probability",
                               opacity=0.65, nbinsx=60))
    fig.add_trace(go.Histogram(x=s["falha"], name="Janelas antes de falha",
                               histnorm="probability", opacity=0.65, nbinsx=30))
    fig.update_layout(title=f"{titulo} (contaminação {contaminacao})",
                      xaxis_title="score_samples (mais negativo = mais anômalo)",
                      yaxis_title="proporção", barmode="overlay",
                      template="plotly_white", height=450)
    return fig

# =============================================================================
# Cartas de controle
# =============================================================================

def calcular_ewma(serie, media_baseline, std_baseline, lambd=0.2, L=3):
    """
    Calcula a carta EWMA com baseline fixo.
    
    lambd : fator de suavização (0 < λ ≤ 1) — menor = mais suave
    L     : multiplicador do limite de controle — menor = mais sensível
    """
    n = len(serie)
    ewma = np.zeros(n)
    ucl  = np.zeros(n)
    
    ewma[0] = media_baseline

    for i in range(1, n):
        ewma[i] = lambd * serie.iloc[i] + (1 - lambd) * ewma[i - 1]
        # Variância assintótica da EWMA
        sigma_ewma = std_baseline * np.sqrt(lambd / (2 - lambd) * (1 - (1 - lambd) ** (2 * (i + 1))))
        ucl[i] = media_baseline + L * sigma_ewma

    alarmes = ewma > ucl

    df_ewma = pd.DataFrame({
        'Valor'  : serie.values,
        'EWMA'   : ewma,
        'UCL'    : ucl,
        'Alarme' : alarmes
    }, index=serie.index)

    return df_ewma


def otimizar_ewma(serie, df_eventos, media_hist, std_hist, param_grid):
    """
    Busca os melhores hiperparâmetros para a carta EWMA baseando-se no maior F2-Score.
    """
    melhor_f2 = -1
    melhores_params = {}
    
    chaves = param_grid.keys()
    combinacoes = list(itertools.product(*param_grid.values()))
    
    print(f"Iniciando otimização com {len(combinacoes)} combinações...")
    
    for config in combinacoes:
        params = dict(zip(chaves, config))
        
        # 1. Gera a carta de controle com os parâmetros atuais
        df_ewma_temp = calcular_ewma(
            serie, media_hist, std_hist, 
            lambd=params['lambd'], L=params['L']
        )
        
        # 2. Avalia a carta usando a janela de dias atual
        y_true = []
        y_pred = []
        janela = params['janela_dias']
        
        for _, evento in df_eventos.iterrows():
            ts = evento["TIMESTAMP"]
            classe = int(evento["Real"])
            
            inicio = ts - pd.Timedelta(days=janela)
            mask = (df_ewma_temp.index >= inicio) & (df_ewma_temp.index < ts)
            slice_controle = df_ewma_temp[mask]
            
            teve_alarme = 1 if slice_controle['Alarme'].any() else 0
            
            y_true.append(classe)
            y_pred.append(teve_alarme)
            
        # 3. Calcula F2
        f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        
        # 4. Atualiza os melhores parâmetros
        if f2 > melhor_f2:
            melhor_f2 = f2
            melhores_params = params
            
    print(f"\nMelhor F2-Score encontrado: {melhor_f2:.4f}")
    print(f"Melhores parâmetros: {melhores_params}")
    
    return melhores_params, melhor_f2


def plotar_carta_ewma(df_ewma, df_eventos, titulo="Carta EWMA", mostrar_falsos=True):
    """Visualização da carta EWMA com a lógica de shapes para as linhas de eventos."""
    fig = go.Figure()

    # Estatística de Controle (EWMA)
    fig.add_trace(go.Scatter(x=df_ewma.index, y=df_ewma['EWMA'], 
                             mode='lines', name='EWMA', line=dict(color='black')))
    
    # Limite de Controle (UCL)
    fig.add_trace(go.Scatter(x=df_ewma.index, y=df_ewma['UCL'], 
                             mode='lines', name='Limite (UCL)', 
                             line=dict(color='red', dash='dash')))
    
    # Pontos de Alarme
    alarmes = df_ewma[df_ewma['Alarme']]
    fig.add_trace(go.Scatter(x=alarmes.index, y=alarmes['EWMA'], 
                             mode='markers', name='Alarme EWMA', 
                             marker=dict(color='red', size=8, symbol='x')))

    # Adição dos eventos com a mesma lógica do seu plot_crystallizer
    for _, row in df_eventos.iterrows():
        if not mostrar_falsos and row["Real"] == 0:
            continue

        cor = "red" if row["Real"] == 1 else "blue"

        fig.add_shape(
            type="line",
            x0=str(row["TIMESTAMP"]),
            x1=str(row["TIMESTAMP"]),
            y0=0, y1=1,
            yref="paper",
            line=dict(color=cor, width=1.5, dash="dash")
        )

    fig.update_layout(
        title=titulo, 
        template="plotly_white", 
        height=450,
        hovermode='x unified'
    )
    # convenção do módulo: a função devolve a fig e quem chama decide se dá .show()
    return fig


def avaliar_carta_controle(df_controle, df_eventos, nome_carta, janela_dias=3):
    """
    Avalia a carta de controle verificando se ela disparou na janela de tempo antes do evento.
    """
    y_true = []
    y_pred = []
    
    for _, evento in df_eventos.iterrows():
        ts = evento["TIMESTAMP"]
        classe = int(evento["Real"])
        
        # Define a janela de busca para o alarme (ex: últimos 3 dias antes do evento)
        inicio = ts - pd.Timedelta(days=janela_dias)
        
        # Filtra a carta de controle nessa janela
        mask = (df_controle.index >= inicio) & (df_controle.index < ts)
        slice_controle = df_controle[mask]
        
        # Se houve QUALQUER alarme nessa janela, prevemos 1 (Anomalia)
        teve_alarme = 1 if slice_controle['Alarme'].any() else 0
        
        y_true.append(classe)
        y_pred.append(teve_alarme)
        
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    
    print(f"\n{'='*55}")
    print(f"Avaliação da Carta: {nome_carta} (Janela de {janela_dias} dias)")
    print(f"{'='*55}")
    print(f"F2-Score: {f2:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_true, y_pred, labels=[0, 1], 
                                target_names=['Operação Normal', 'Contaminação Detectada'], 
                                zero_division=0))
    return y_true, y_pred, f2


def calcular_cusum_dinamico(serie, janela_baseline=30, k=0.5, h=4):
    """
    Calcula o CUSUM adaptativo. A média e desvio padrão são atualizados 
    continuamente usando uma janela móvel do passado.
    """
    n = len(serie)
    c_pos = np.zeros(n)
    limite_h = np.zeros(n)
    alarmes = np.zeros(n, dtype=bool)
    
    # Média e std móveis. O shift(1) garante que o valor de "hoje" não contamine o baseline de "hoje"
    media_movel = serie.shift(1).rolling(window=janela_baseline, min_periods=janela_baseline//2).mean()
    std_movel = serie.shift(1).rolling(window=janela_baseline, min_periods=janela_baseline//2).std()
    
    for i in range(1, n):
        mu = media_movel.iloc[i]
        sigma = std_movel.iloc[i]
        
        # Pula se não houver histórico suficiente ou se o desvio for zero absoluto
        if pd.isna(mu) or pd.isna(sigma) or sigma == 0:
            continue
            
        # Calcula os limites dinâmicos para o dia 'i'
        K_val = k * sigma
        H_val = h * sigma
        limite_h[i] = H_val
        
        # Acumula o desvio positivo
        c_pos[i] = max(0, c_pos[i-1] + serie.iloc[i] - mu - K_val)
        
        # Dispara o alarme e reseta a soma (crucial para evitar fadiga de alarme)
        if c_pos[i] > H_val:
            alarmes[i] = True
            c_pos[i] = 0 
            
    df_cusum = pd.DataFrame({
        'Valor': serie.values,
        'CUSUM_Positivo': c_pos,
        'Limite_H': limite_h,
        'Alarme': alarmes
    }, index=serie.index)
    
    return df_cusum


def otimizar_cusum_dinamico(serie, df_eventos, param_grid):
    """Grid Search manual para encontrar a melhor parametrização do CUSUM Dinâmico."""
    melhor_f2 = -1
    melhores_params = {}
    
    chaves = param_grid.keys()
    combinacoes = list(itertools.product(*param_grid.values()))
    
    print(f"Iniciando otimização do CUSUM Dinâmico com {len(combinacoes)} combinações...")
    
    for config in combinacoes:
        params = dict(zip(chaves, config))
        
        df_cusum_temp = calcular_cusum_dinamico(
            serie, 
            janela_baseline=params['janela_baseline'], 
            k=params['k'], 
            h=params['h']
        )
        
        y_true = []
        y_pred = []
        janela = params['janela_dias']
        
        for _, evento in df_eventos.iterrows():
            ts = evento["TIMESTAMP"]
            classe = int(evento["Real"])
            
            inicio = ts - pd.Timedelta(days=janela)
            mask = (df_cusum_temp.index >= inicio) & (df_cusum_temp.index < ts)
            slice_controle = df_cusum_temp[mask]
            
            teve_alarme = 1 if slice_controle['Alarme'].any() else 0
            
            y_true.append(classe)
            y_pred.append(teve_alarme)
            
        f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        
        if f2 > melhor_f2:
            melhor_f2 = f2
            melhores_params = params
            
    print(f"\nMelhor F2-Score encontrado: {melhor_f2:.4f}")
    print(f"Melhores parâmetros: {melhores_params}")
    
    return melhores_params, melhor_f2


def plotar_carta_cusum(df_cusum, df_eventos, titulo="Carta CUSUM Dinâmico", mostrar_falsos=True):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df_cusum.index, y=df_cusum['CUSUM_Positivo'], 
                             mode='lines', name='CUSUM Positivo', line=dict(color='black')))
    
    fig.add_trace(go.Scatter(x=df_cusum.index, y=df_cusum['Limite_H'], 
                             mode='lines', name='Limite Dinâmico (H)', 
                             line=dict(color='red', dash='dash')))
    
    alarmes = df_cusum[df_cusum['Alarme']]
    fig.add_trace(go.Scatter(x=alarmes.index, y=alarmes['CUSUM_Positivo'], 
                             mode='markers', name='Alarme', 
                             marker=dict(color='red', size=8, symbol='x')))

    for _, row in df_eventos.iterrows():
        if not mostrar_falsos and row["Real"] == 0:
            continue

        cor = "red" if row["Real"] == 1 else "blue"
        texto = row["EVENTO"] if "EVENTO" in df_eventos.columns else ("Contaminação Real" if row["Real"] == 1 else "Evento Normal")

        fig.add_shape(
            type="line", x0=str(row["TIMESTAMP"]), x1=str(row["TIMESTAMP"]),
            y0=0, y1=1, yref="paper", line=dict(color=cor, width=1.5, dash="dash")
        )
        fig.add_annotation(
            x=str(row["TIMESTAMP"]), y=1, yref="paper", text=texto,
            showarrow=False, textangle=-90, yanchor="top", font=dict(color=cor)
        )

    fig.update_layout(title=titulo, template="plotly_white", height=450, hovermode='x unified')
    # convenção do módulo: a função devolve a fig e quem chama decide se dá .show()
    return fig


# =============================================================================
# Protocolo de avaliação, validação fora da amostra e auditoria de dado
# =============================================================================
#
# Esta seção existe por causa de um defeito medido no relatório supervisionado: o corte
# temporal do `dividir_dados` corta pelo quantil dos POSITIVOS, e a proporção de classes
# inverte entre treino e teste (no unificado: treino 24 pos / 6 neg, teste 7 pos / 21 neg).
# Um modelo treinado com 80% de positivos prevê positivo quase sempre — e é o que os
# relatórios mostravam: recall 1.00 em 11 das 12 combinações, com precisão igual à taxa
# base. Nada podia ser concluído dali.

def calcular_ewma_rolante(serie, lambd=0.2, L=3, janela_baseline=500, min_periodos=100):
    """EWMA com baseline ROLANTE, no lugar do baseline escolhido à mão.

    O baseline fixo de 2012-2015 usado antes fica sistematicamente frouxo depois de 2020
    (mediana anual ~2.7 -> ~1.8 ppm), e o grid compensava isso ajustando ruído. Aqui a
    média e a dispersão vêm de uma janela móvel do passado (`shift(1)` garante que o ponto
    de hoje não entra no próprio limite), e a dispersão é robusta (IQR/1.349) porque a
    série tem excursões de três ordens de grandeza.

    Devolve Valor, EWMA, UCL, Z (quantos sigmas acima do baseline) e Alarme.
    """
    serie = serie.dropna()
    ewma = serie.ewm(alpha=lambd, adjust=False).mean()
    passado = serie.shift(1).rolling(janela_baseline, min_periods=min_periodos)
    mu = passado.median()
    escala = (passado.quantile(0.75) - passado.quantile(0.25)) / 1.349
    sigma_ewma = escala * np.sqrt(lambd / (2 - lambd))
    z = (ewma - mu) / sigma_ewma.replace(0, np.nan)
    ucl = mu + L * sigma_ewma
    return pd.DataFrame({"Valor": serie, "EWMA": ewma, "UCL": ucl,
                         "Z": z, "Alarme": ewma > ucl})


# -----------------------------------------------------------------------------
# Validação temporal do supervisionado
# -----------------------------------------------------------------------------

def cortes_temporais(timestamps, n_cortes=4, frac_min=0.45, frac_max=0.9):
    """Pontos de corte para validação de janela expansiva."""
    ts = pd.Series(pd.to_datetime(timestamps)).sort_values()
    fracoes = np.linspace(frac_min, frac_max, n_cortes)
    return [ts.quantile(f) for f in fracoes]


def modelos_padrao_cv(random_state=RANDOM_STATE):
    """Modelos usados na CV temporal — sem grid interno, de propósito.

    Com ~30 eventos de treino por fold, um GridSearchCV dentro do fold escolhe
    hiperparâmetro por ruído. Aqui os modelos são fixos e balanceados por classe; o que
    se mede é a diferença entre conjuntos de features e entre representações do problema,
    não o ajuste fino.
    """
    return {
        "LogReg": LogisticRegression(max_iter=2000, class_weight="balanced",
                                     random_state=random_state),
        "RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                               class_weight="balanced_subsample",
                                               random_state=random_state, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                                 subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                                 random_state=random_state, n_jobs=-1),
    }


def avaliar_cv_temporal(df_feat, colunas, n_cortes=4, modelos=None, limiar=0.4,
                        coluna_ts="TIMESTAMP_Evento", alvo="Real", verbose=True):
    """Validação temporal de janela expansiva: treina no passado, testa no bloco seguinte.

    Reporta a PREVALÊNCIA de cada fold junto com a métrica — sem isso um F2 alto pode ser
    só efeito de o bloco de teste ter muitos positivos. F2 é a métrica-alvo do estudo
    (evento raro, recall importa mais que precisão); F1 aparecia no relatório antigo por
    engano.
    """
    modelos = modelos or modelos_padrao_cv()
    df = df_feat.sort_values(coluna_ts).reset_index(drop=True)
    cortes = cortes_temporais(df[coluna_ts], n_cortes)

    linhas = []
    for i, corte in enumerate(cortes):
        fim = cortes[i + 1] if i + 1 < len(cortes) else df[coluna_ts].max() + pd.Timedelta(days=1)
        treino = df[df[coluna_ts] < corte]
        teste = df[(df[coluna_ts] >= corte) & (df[coluna_ts] <= fim)]
        if len(teste) < 3 or treino[alvo].nunique() < 2 or teste[alvo].nunique() < 2:
            if verbose:
                print(f"  fold {i+1} ({corte:%Y-%m}) descartado: "
                      f"{len(treino)} treino / {len(teste)} teste, classes insuficientes")
            continue

        X_tr, y_tr = treino[colunas], treino[alvo]
        X_te, y_te = teste[colunas], teste[alvo]
        for nome, modelo in modelos.items():
            pipe = SklearnPipeline([("imputer", SimpleImputer(strategy="median")),
                                    ("scaler", StandardScaler()),
                                    ("clf", modelo)])
            pipe.fit(X_tr, y_tr)
            prob = pipe.predict_proba(X_te)[:, 1]
            y_pred = (prob >= limiar).astype(int)
            linhas.append({
                "fold": i + 1, "corte": corte.date(), "modelo": nome,
                "n_treino": len(treino), "prev_treino": round(y_tr.mean(), 2),
                "n_teste": len(teste), "prev_teste": round(y_te.mean(), 2),
                "F2": round(fbeta_score(y_te, y_pred, beta=2, zero_division=0), 3),
                "precisao": round(precision_score(y_te, y_pred, zero_division=0), 3),
                "recall": round(recall_score(y_te, y_pred, zero_division=0), 3),
                "PR_AUC": round(average_precision_score(y_te, prob), 3) if y_te.nunique() > 1 else np.nan,
            })

    df_cv = pd.DataFrame(linhas)
    if df_cv.empty:
        print("Nenhum fold utilizável — amostra pequena demais para validação temporal.")
        return df_cv, pd.DataFrame()

    resumo = (df_cv.groupby("modelo")
              .agg(folds=("F2", "size"), F2_medio=("F2", "mean"), F2_dp=("F2", "std"),
                   F2_min=("F2", "min"), F2_max=("F2", "max"),
                   precisao=("precisao", "mean"), recall=("recall", "mean"),
                   PR_AUC=("PR_AUC", "mean"))
              .round(3).sort_values("F2_medio", ascending=False))

    if verbose:
        print(df_cv.to_string(index=False))
        print("\nResumo por modelo (média entre folds):")
        print(resumo.to_string())
        print("\nLeitura: compare F2_dp com a diferença entre modelos. Se o desvio entre "
              "folds for maior que a diferença, os modelos são indistinguíveis nesta amostra.")
    return df_cv, resumo


# -----------------------------------------------------------------------------
# Modelo avaliado como detector (a única comparação justa com a regra vigente)
# -----------------------------------------------------------------------------

def metricas_alarmes_periodo(alarmes, map_falhas, periodos=None, janela_alarme=15,
                             agrupar_dias=15, rotulo=""):
    """Igual a `_metricas_alarmes`, mas restrito a intervalos de tempo.

    Necessário para comparar modelo e regra na MESMA fatia de tempo: um detector avaliado
    na série inteira e um modelo avaliado só no período de teste não são comparáveis.
    """
    def dentro(d):
        return periodos is None or any(ini <= d <= fim for ini, fim in periodos)

    alarmes = {c: [d for d in _agrupar_alarmes(v, agrupar_dias) if dentro(d)]
               for c, v in alarmes.items()}
    falhas = {c: df[df["TIMESTAMP"].apply(dentro)] for c, df in map_falhas.items()}
    total = sum(len(df) for df in falhas.values())

    vp = fp = 0
    for cryst, df_ev in falhas.items():
        for ts in df_ev["TIMESTAMP"]:
            if any(0 <= (ts - a).days <= janela_alarme for a in alarmes.get(cryst, [])):
                vp += 1
    for cryst, lista in alarmes.items():
        ts_falhas = falhas.get(cryst, pd.DataFrame({"TIMESTAMP": []}))["TIMESTAMP"]
        for a in lista:
            if not any(0 <= (ts - a).days <= janela_alarme for ts in ts_falhas):
                fp += 1
    precisao = vp / (vp + fp) if vp + fp else 0.0
    recall = vp / total if total else 0.0
    f2 = 5 * precisao * recall / (4 * precisao + recall) if precisao + recall else 0.0
    return {"regra": rotulo, "falhas_no_periodo": total, "VP": vp, "FP": fp,
            "precisao": round(precisao, 3), "recall": round(recall, 3), "F2": round(f2, 3)}


def treinar_detector_janelas(df_janelas, map_falhas, colunas=None, n_cortes=4,
                             quantis_alarme=(0.90, 0.95, 0.98, 0.99), modelo=None,
                             random_state=RANDOM_STATE, margem=None, map_medicoes=None,
                             verbose=True):
    """Supervisionado sobre as janelas deslizantes, avaliado como DETECTOR.

    Por que trocar o objeto de estudo: a tabela de eventos tem ~60 linhas e uma classe
    negativa SINTETIZADA por um limiar arbitrário (10 ou 5 ppm) — o resultado depende de
    uma escolha nossa. As janelas deslizantes são ~5000, com taxa base real (3%) e rótulo
    que não depende de limiar nenhum.

    O modelo é treinado no passado de cada corte e pontua o bloco seguinte; as janelas
    acima do quantil viram alarme (agrupado em 15 dias) e são medidas contra as falhas
    daquele bloco — exatamente a régua dos outros detectores. Quando `margem` e
    `map_medicoes` são passados, a regra composta é medida NO MESMO período de teste.
    """
    colunas = colunas or [c for c in COLUNAS_MODELO_JANELA if c in df_janelas.columns]
    modelo = modelo or RandomForestClassifier(
        n_estimators=400, min_samples_leaf=5, class_weight="balanced_subsample",
        random_state=random_state, n_jobs=-1)

    df = df_janelas.sort_values("TIMESTAMP").reset_index(drop=True)
    cortes = cortes_temporais(df["TIMESTAMP"], n_cortes)

    scores_teste, periodos = [], []
    for i, corte in enumerate(cortes):
        fim = cortes[i + 1] if i + 1 < len(cortes) else df["TIMESTAMP"].max() + pd.Timedelta(days=1)
        treino = df[df["TIMESTAMP"] < corte]
        teste = df[(df["TIMESTAMP"] >= corte) & (df["TIMESTAMP"] <= fim)]
        if treino["pre_falha"].sum() < 5 or teste["pre_falha"].sum() < 1:
            continue
        pipe = SklearnPipeline([("imputer", SimpleImputer(strategy="median")),
                                ("scaler", StandardScaler()),
                                ("clf", modelo)])
        pipe.fit(treino[colunas], treino["pre_falha"])
        prob = pipe.predict_proba(teste[colunas])[:, 1]
        bloco = teste[["Crystallizer", "TIMESTAMP"]].copy()
        bloco["score"] = prob
        bloco["fold"] = i + 1
        # limiar vem do TREINO, nunca do teste
        bloco["quantis_treino"] = None
        q_treino = {q: float(np.quantile(pipe.predict_proba(treino[colunas])[:, 1], q))
                    for q in quantis_alarme}
        for q, val in q_treino.items():
            bloco[f"acima_q{int(q*100)}"] = bloco["score"] >= val
        scores_teste.append(bloco)
        periodos.append((corte, fim))

    if not scores_teste:
        print("Nenhum fold utilizável.")
        return {}

    scores = pd.concat(scores_teste, ignore_index=True)

    linhas = []
    for q in quantis_alarme:
        col = f"acima_q{int(q*100)}"
        alarmes = {c: list(g.loc[g[col], "TIMESTAMP"]) for c, g in scores.groupby("Crystallizer")}
        linhas.append(metricas_alarmes_periodo(alarmes, map_falhas, periodos,
                                               rotulo=f"modelo (quantil {q})"))
    if margem is not None and map_medicoes is not None:
        for lmax, lmed, lmar in ((20, 3.0, 0.6), (20, 3.5, 0.9)):
            al = alarmes_regra_composta(map_medicoes, margem, lmax, lmed, lmar)
            linhas.append(metricas_alarmes_periodo(
                al, map_falhas, periodos,
                rotulo=f"regra composta ({lmax}/{lmed}/{lmar})"))
        diario_max = serie_diaria(map_medicoes, "max")
        al = {c: list(diario_max[c].dropna()[diario_max[c].dropna() > 5].index)
              for c in map_medicoes}
        linhas.append(metricas_alarmes_periodo(al, map_falhas, periodos,
                                               rotulo="regra vigente (max > 5)"))

    tabela = pd.DataFrame(linhas).sort_values("F2", ascending=False).reset_index(drop=True)
    if verbose:
        print(f"Folds usados: {len(periodos)}  |  janelas pontuadas fora da amostra: {len(scores)}")
        print(tabela.to_string(index=False))
        print("\nTodas as linhas acima são medidas NO MESMO período de teste — é a única "
              "comparação justa entre modelo e regra.")
    return {"tabela": tabela, "scores": scores, "periodos": periodos, "colunas": colunas}


# -----------------------------------------------------------------------------
# Regra composta: ajuste e validação fora da amostra
# -----------------------------------------------------------------------------

def alarmes_regra_composta(map_medicoes, margem, limite_max=20, limite_mediana=3.0,
                           limite_margem=0.6, coluna=COLUNA_FE):
    """Datas de alarme da regra composta, por reator (sem agrupar)."""
    diario_max = serie_diaria(map_medicoes, "max", coluna)
    diario_med = serie_diaria(map_medicoes, "median", coluna)
    alarmes = {}
    for cryst in map_medicoes:
        s_max = diario_max[cryst].dropna()
        s_med = diario_med[cryst].dropna()
        s_mar = margem[cryst].dropna()
        impulso = set(s_max[s_max > limite_max].index)
        sustentado = {d for d in s_med[s_med > limite_mediana].index
                      if d in s_mar.index and s_mar.loc[d] > limite_margem}
        alarmes[cryst] = sorted(impulso | sustentado)
    return alarmes


GRADE_REGRA_COMPOSTA = {
    "limite_max": [10, 15, 20, 30],
    "limite_mediana": [2.5, 3.0, 3.5, 4.0],
    "limite_margem": [0.3, 0.6, 0.9, 1.2],
}


def otimizar_regra_composta(map_medicoes, map_falhas, margem, grade=None, periodos=None,
                            coluna=COLUNA_FE, verbose=False):
    """Escolhe os três limiares por F2 dentro de `periodos` (o período de ajuste)."""
    grade = grade or GRADE_REGRA_COMPOSTA
    melhor = None
    for lmax in grade["limite_max"]:
        for lmed in grade["limite_mediana"]:
            for lmar in grade["limite_margem"]:
                al = alarmes_regra_composta(map_medicoes, margem, lmax, lmed, lmar, coluna)
                m = metricas_alarmes_periodo(al, map_falhas, periodos,
                                             rotulo=f"{lmax}/{lmed}/{lmar}")
                if melhor is None or m["F2"] > melhor[1]["F2"]:
                    melhor = ((lmax, lmed, lmar), m)
    if verbose:
        print(f"Melhores limiares no período de ajuste: {melhor[0]}  ->  F2 {melhor[1]['F2']}")
    return melhor


def validar_regra_composta(map_medicoes, map_falhas, margem=None, data_corte="2019-01-01",
                           limiares_fixos=(20, 3.0, 0.6), n_bootstrap=500,
                           random_state=RANDOM_STATE, coluna=COLUNA_FE, verbose=True):
    """Três validações da regra composta, porque os limiares foram escolhidos in-sample.

      1. HOLDOUT TEMPORAL — ajusta os limiares só com as falhas anteriores a `data_corte`
         e mede no período seguinte. Responde "a regra ajustada no passado funciona no
         futuro?", que é a pergunta operacional;
      2. LEAVE-ONE-REACTOR-OUT — ajusta em dois reatores e mede no terceiro. Responde "os
         limiares são do processo ou de um equipamento?";
      3. BOOTSTRAP das falhas — reamostra as 31 falhas com reposição e devolve o intervalo
         do F2 da regra fixa. Captura a incerteza que domina aqui (poucos positivos); os
         falsos positivos ficam fixos, então o intervalo é do lado do recall.
    """
    if margem is None:
        _, _, margem = calcular_margem_cross_reator(map_medicoes, coluna, verbose=False)
    corte = pd.Timestamp(data_corte)
    inicio = min(df["TIMESTAMP"].min() for df in map_medicoes.values())
    fim = max(df["TIMESTAMP"].max() for df in map_medicoes.values())
    per_ajuste = [(inicio, corte)]
    per_teste = [(corte, fim)]

    # 1. holdout temporal
    (lmax, lmed, lmar), m_ajuste = otimizar_regra_composta(
        map_medicoes, map_falhas, margem, periodos=per_ajuste, coluna=coluna)
    al_ajustada = alarmes_regra_composta(map_medicoes, margem, lmax, lmed, lmar, coluna)
    al_fixa = alarmes_regra_composta(map_medicoes, margem, *limiares_fixos, coluna=coluna)
    holdout = pd.DataFrame([
        {**metricas_alarmes_periodo(al_ajustada, map_falhas, per_ajuste,
                                    rotulo=f"ajustada {lmax}/{lmed}/{lmar}"), "periodo": "ajuste"},
        {**metricas_alarmes_periodo(al_ajustada, map_falhas, per_teste,
                                    rotulo=f"ajustada {lmax}/{lmed}/{lmar}"), "periodo": "teste"},
        {**metricas_alarmes_periodo(al_fixa, map_falhas, per_teste,
                                    rotulo=f"fixa {limiares_fixos}"), "periodo": "teste"},
        {**metricas_alarmes_periodo(
            {c: list(serie_diaria(map_medicoes, "max", coluna)[c].dropna()[
                serie_diaria(map_medicoes, "max", coluna)[c].dropna() > 5].index)
             for c in map_medicoes}, map_falhas, per_teste,
            rotulo="regra vigente (max > 5)"), "periodo": "teste"},
    ])

    # 2. leave-one-reactor-out
    linhas_loro = []
    for alvo in map_medicoes:
        outros_falhas = {c: df for c, df in map_falhas.items() if c != alvo}
        (a, b, c_), _ = otimizar_regra_composta(map_medicoes, outros_falhas, margem, coluna=coluna)
        al = alarmes_regra_composta(map_medicoes, margem, a, b, c_, coluna)
        m = metricas_alarmes_periodo({alvo: al[alvo]}, {alvo: map_falhas[alvo]},
                                     rotulo=f"{alvo}: limiares {a}/{b}/{c_} (dos outros dois)")
        linhas_loro.append(m)
    loro = pd.DataFrame(linhas_loro)

    # 3. bootstrap das falhas
    rng = np.random.default_rng(random_state)
    agrupados = {c: _agrupar_alarmes(v) for c, v in al_fixa.items()}
    pares = [(c, ts) for c, df in map_falhas.items() for ts in df["TIMESTAMP"]]
    acertos = np.array([1 if any(0 <= (ts - a).days <= 15 for a in agrupados.get(c, [])) else 0
                        for c, ts in pares])
    fp_total = sum(
        1 for c, lista in agrupados.items() for a in lista
        if not any(0 <= (ts - a).days <= 15 for ts in map_falhas[c]["TIMESTAMP"]))
    f2s = []
    for _ in range(n_bootstrap):
        amostra = rng.choice(acertos, size=len(acertos), replace=True)
        vp = int(amostra.sum())
        prec = vp / (vp + fp_total) if vp + fp_total else 0
        rec = vp / len(amostra)
        f2s.append(5 * prec * rec / (4 * prec + rec) if prec + rec else 0)
    ic = (float(np.percentile(f2s, 2.5)), float(np.percentile(f2s, 97.5)))

    if verbose:
        print("1) HOLDOUT TEMPORAL — ajuste até "
              f"{corte:%d/%m/%Y}, teste depois")
        print(holdout.to_string(index=False))
        print("\n2) LEAVE-ONE-REACTOR-OUT — limiares vindos dos outros dois reatores")
        print(loro.to_string(index=False))
        print(f"\n3) BOOTSTRAP ({n_bootstrap} reamostragens das falhas) — regra fixa "
              f"{limiares_fixos}")
        print(f"   F2 = {np.mean(f2s):.3f}  IC95% [{ic[0]:.3f}, {ic[1]:.3f}]  "
              f"(FP fixos em {fp_total})")

    return {"holdout": holdout, "loro": loro, "bootstrap_f2": f2s, "ic95": ic,
            "limiares_ajustados": (lmax, lmed, lmar)}


def lead_time_regra(map_medicoes, map_falhas, margem=None, limiares=(20, 3.0, 0.6),
                    dias_max=60, coluna=COLUNA_FE, verbose=True):
    """Antecedência do primeiro alarme antes de cada falha — evento a evento.

    É o número que decide o projeto. Reportado como DISTRIBUIÇÃO, não como média: nos
    limiares confiáveis do estudo a mediana é 0 dia, o que significa que a leitura alta é
    o gatilho da parada, não um precursor. Se continuar 0, o produto não é predição — é
    confirmação mais barata, com menos alarme falso.
    """
    if margem is None:
        _, _, margem = calcular_margem_cross_reator(map_medicoes, coluna, verbose=False)
    alarmes = {c: _agrupar_alarmes(v)
               for c, v in alarmes_regra_composta(map_medicoes, margem, *limiares, coluna=coluna).items()}

    linhas = []
    for cryst, df_ev in map_falhas.items():
        for ts in df_ev["TIMESTAMP"]:
            anteriores = [a for a in alarmes.get(cryst, [])
                          if 0 <= (ts - a).days <= dias_max]
            linhas.append({
                "Crystallizer": cryst, "falha": ts,
                "alarmou": bool(anteriores),
                "lead_dias": float((ts - min(anteriores)).days) if anteriores else np.nan,
                "n_alarmes_60d": len(anteriores),
            })
    df = pd.DataFrame(linhas).sort_values(["Crystallizer", "falha"])
    leads = df["lead_dias"].dropna()
    if verbose:
        print(f"Falhas com alarme em até {dias_max} d: {len(leads)} de {len(df)}")
        if len(leads):
            print(f"Lead time (dias): mediana {leads.median():.1f} | "
                  f"p25 {leads.quantile(.25):.1f} | p75 {leads.quantile(.75):.1f} | "
                  f"máx {leads.max():.1f}")
            print(f"  com lead >= 1 dia : {(leads >= 1).sum()} de {len(df)} falhas")
            print(f"  com lead >= 7 dias: {(leads >= 7).sum()} de {len(df)} falhas")
    return df


# -----------------------------------------------------------------------------
# Auditoria de dado e ponto de operação
# -----------------------------------------------------------------------------

def auditar_janelas_eventos(map_medicoes, map_eventos, janela_dias=15, min_amostras=8,
                            coluna=COLUNA_FE, verbose=True):
    """Suporte da janela de cada evento — o viés que a tabela de eventos ainda esconde.

    Nas janelas deslizantes o suporte insuficiente é filtrado; na tabela de eventos um
    evento com 2 amostras em 15 dias ainda entra calado e vira estatística de 2 pontos.
    """
    linhas = []
    for cryst, df_ev in map_eventos.items():
        df_med = map_medicoes[cryst]
        for _, ev in df_ev.iterrows():
            ts = ev["TIMESTAMP"]
            jan = df_med[(df_med["TIMESTAMP"] >= ts - pd.Timedelta(days=janela_dias)) &
                         (df_med["TIMESTAMP"] < ts)]
            t = jan["TIMESTAMP"].sort_values()
            lacuna = float(t.diff().dt.total_seconds().max() / 86400) if len(t) > 1 else np.nan
            linhas.append({
                "Crystallizer": cryst, "TIMESTAMP": ts, "Real": int(ev.get("Real", 1)),
                "n_amostras": len(jan), "maior_lacuna_dias": round(lacuna, 1) if lacuna == lacuna else np.nan,
                "suficiente": len(jan) >= min_amostras,
            })
    df = pd.DataFrame(linhas).sort_values(["Crystallizer", "TIMESTAMP"])
    if verbose:
        ruins = df[~df["suficiente"]]
        print(f"Eventos avaliados: {len(df)}  |  com menos de {min_amostras} amostras "
              f"na janela de {janela_dias} d: {len(ruins)}")
        if len(ruins):
            print(ruins.to_string(index=False))
        print(f"\nMediana de amostras por janela: {df['n_amostras'].median():.0f}  "
              f"(mín {df['n_amostras'].min()}, máx {df['n_amostras'].max()})")
    return df


def ficha_eventos_para_validacao(df_inspecoes, map_medicoes, janela_dias=15,
                                 incluir_descartados=True, coluna=COLUNA_FE):
    """Planilha para a planta confirmar data e natureza de cada evento.

    Todo resultado que depende de rótulo repousa nestes timestamps, e há divergência
    conhecida entre o deck da equipe Bayer e a planilha de inspeção em pelo menos três
    eventos. Esta ficha põe lado a lado: a data da planilha, a data reancorada, o
    deslocamento aplicado, o motivo do deslocamento e o suporte de amostra — para a planta
    validar linha a linha.
    """
    cols = ["Crystallizer", "Inicio", "DataAncoradaManual", "TS_Ajustado", "AncoraManual",
            "Deslocado", "DiasDeslocado", "LacunaDias", "LacunaDePlanta", "Selecionado",
            "Falha", "Emergencia", "TrocaDoReator", "FerroCitado", "TipoFalha",
            "DescobertoEmParada", "Corrigido", "ModoFalha", "FerroPlausivel",
            "ModoCorrigido", "Ocorrimento"]
    disponiveis = [c for c in cols if c in df_inspecoes.columns]
    df = df_inspecoes[df_inspecoes["NoPeriodo"]][disponiveis].copy() \
        if "NoPeriodo" in df_inspecoes.columns else df_inspecoes[disponiveis].copy()

    if not incluir_descartados:
        df = df[df["Selecionado"]]

    n_amostras = []
    for _, r in df.iterrows():
        ts = r.get("TS_Ajustado")
        med = map_medicoes.get(r["Crystallizer"])
        if pd.isna(ts) or med is None:
            n_amostras.append(np.nan)
            continue
        jan = med[(med["TIMESTAMP"] >= ts - pd.Timedelta(days=janela_dias)) &
                  (med["TIMESTAMP"] < ts)]
        n_amostras.append(len(jan))
    df["n_amostras_janela"] = n_amostras
    df["status"] = np.where(df.get("Selecionado", False), "usado como falha",
                            np.where(df.get("LacunaDePlanta", False),
                                     "descartado: parada de planta", "descartado: não é falha"))
    if "Ocorrimento" in df.columns:
        df["Ocorrimento"] = df["Ocorrimento"].astype(str).str.replace(r"\s+", " ", regex=True).str.slice(0, 90)
    return df.sort_values(["Crystallizer", "TS_Ajustado"]).reset_index(drop=True)


def custo_esperado(df_detectores, custo_fp, custo_fn, n_falhas=None, coluna_vp="VP",
                   coluna_fp="FP"):
    """Custo esperado de cada regra, dado o custo de um alarme falso e de uma falha perdida.

    Enquanto o histórico de condenação indevida e o custo de parada não planejada não
    chegarem da planta, o ponto de operação é escolhido pelo F2 — que embute uma razão
    arbitrária (recall vale 4x a precisão). Com os dois custos, a escolha vira aritmética.
    """
    df = df_detectores.copy()
    n_falhas = n_falhas or df.get("falhas_no_periodo", pd.Series([31] * len(df))).max()
    df["FN"] = n_falhas - df[coluna_vp]
    df["custo_total"] = df[coluna_fp] * custo_fp + df["FN"] * custo_fn
    return df.sort_values("custo_total").reset_index(drop=True)


def sensibilidade_custo(df_detectores, razoes=(1, 5, 10, 25, 50, 100), n_falhas=None):
    """Qual regra vence para cada razão custo(falha perdida)/custo(alarme falso)."""
    linhas = []
    for r in razoes:
        d = custo_esperado(df_detectores, custo_fp=1, custo_fn=r, n_falhas=n_falhas)
        linhas.append({"custo_FN/custo_FP": r, "regra_vencedora": d.iloc[0]["regra"],
                       "custo_relativo": round(d.iloc[0]["custo_total"], 1),
                       "VP": int(d.iloc[0]["VP"]), "FP": int(d.iloc[0]["FP"])})
    return pd.DataFrame(linhas)


def testar_tendencia(map_medicoes, coluna=COLUNA_FE, verbose=True):
    """Quantifica o drift: Mann-Kendall (via tau de Kendall) + inclinação de Sen + KW por ano.

    O drift era visível nos gráficos mas nunca medido — e ele é o motivo de o limite fixo
    de 5 ppm significar coisas diferentes em 2013 e em 2025. ADF não é usado aqui: a
    pergunta é de tendência monotônica, não de raiz unitária (e statsmodels não está
    instalado neste ambiente).
    """
    linhas = []
    for cryst, df in map_medicoes.items():
        s = df.set_index("TIMESTAMP")[coluna].resample("D").median().dropna()
        t = np.arange(len(s), dtype=float)
        tau, p = kendalltau(t, s.values)

        mensal = s.resample("MS").median().dropna()
        y = mensal.values
        x = np.arange(len(y), dtype=float)
        inclinacoes = [(y[j] - y[i]) / (x[j] - x[i])
                       for i in range(len(y)) for j in range(i + 1, len(y))]
        sen_mes = float(np.median(inclinacoes)) if inclinacoes else np.nan

        anos = s.groupby(s.index.year)
        h, p_kw = kruskal(*[g.values for _, g in anos if len(g) > 30])
        linhas.append({"Crystallizer": cryst, "tau_kendall": round(tau, 3),
                       "p_mann_kendall": f"{p:.2e}",
                       "sen_ppm_por_ano": round(sen_mes * 12, 4),
                       "KW_entre_anos_H": round(h, 1), "KW_p": f"{p_kw:.2e}",
                       "mediana_2011_2015": round(float(s[:"2015"].median()), 2),
                       "mediana_2021_2026": round(float(s["2021":].median()), 2)})
    df = pd.DataFrame(linhas)
    if verbose:
        print(df.to_string(index=False))
        print("\nLeitura: tau negativo com p minúsculo = tendência de queda confirmada. "
              "A série NÃO é estacionária, o que justifica limite móvel e features "
              "relativas em vez de patamar fixo.")
    return df


# =============================================================================
# Escopo por reator: ajuste unificado, alarme e prestação de contas separados
# =============================================================================
#
# A decisão de escopo foi medida, não arbitrada (a seção "Resultados por reator" do
# notebook reproduz):
#
#   - as DISTRIBUIÇÕES de ferro são estatisticamente iguais entre os três reatores
#     (Kruskal-Wallis eta² = 0.0008; mediana 2.2/2.3/2.3 ppm; p99 = 4.3 nos três), o que
#     justifica unificar carga, limpeza e EDA;
#   - a RELAÇÃO ferro -> falha é diferente em cada um: mediana diária > 3.5 tem lift 6.17
#     no C3 e 1.42 no C1; margem > 1.2 tem lift 12.24 no C1 e 2.58 no C2. Cada reator
#     responde a uma estatística diferente;
#   - TREINAR separado não ajuda (o C3, onde o sinal existe, tem só 6 falhas e aprende
#     melhor com os outros dois do que consigo mesmo), mas CALIBRAR o limiar separado
#     ajuda muito (no C3, mesmo recall com 45% menos alarme falso);
#   - a melhor feature (margem cross-reator) só existe com os três juntos — separar
#     completamente nem sequer é possível.
#
# Daí a regra de trabalho do estudo: **ajuste unificado, alarme e relatório por reator.**

def comparar_escopo_treino(df_janelas, colunas=None, n_cortes=3, modelo=None,
                           random_state=RANDOM_STATE, verbose=True):
    """Treinar unificado, só no próprio reator ou nos outros dois? Medido fora da amostra.

    Para cada corte temporal, treina os três escopos e pontua as janelas do bloco seguinte
    de cada reator. A métrica é PR-AUC dividida pela taxa base do próprio reator (`lift_PR`),
    porque a taxa base é diferente em cada um e o PR-AUC cru não seria comparável.
    `lift_PR` = 1 significa que o modelo não ordena melhor que o acaso.
    """
    colunas = colunas or [c for c in COLUNAS_MODELO_JANELA if c in df_janelas.columns]
    modelo = modelo or RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
        random_state=random_state, n_jobs=-1)

    df = df_janelas.sort_values("TIMESTAMP").reset_index(drop=True)
    cortes = cortes_temporais(df["TIMESTAMP"], n_cortes)
    reatores = sorted(df["Crystallizer"].unique())
    acumulado = {e: {c: {"y": [], "s": []} for c in reatores}
                 for e in ["unificado", "proprio", "outros"]}

    for i, corte in enumerate(cortes):
        fim = cortes[i + 1] if i + 1 < len(cortes) else df["TIMESTAMP"].max() + pd.Timedelta(days=1)
        treino = df[df["TIMESTAMP"] < corte]
        teste = df[(df["TIMESTAMP"] >= corte) & (df["TIMESTAMP"] <= fim)]
        if treino["pre_falha"].sum() < 5 or teste["pre_falha"].sum() < 1:
            continue
        for cryst in reatores:
            teste_c = teste[teste["Crystallizer"] == cryst]
            if len(teste_c) < 20 or teste_c["pre_falha"].sum() < 1:
                continue
            escopos = {"unificado": treino,
                       "proprio": treino[treino["Crystallizer"] == cryst],
                       "outros": treino[treino["Crystallizer"] != cryst]}
            for nome, tr in escopos.items():
                if tr["pre_falha"].sum() < 3:
                    continue
                pipe = SklearnPipeline([("imputer", SimpleImputer(strategy="median")),
                                        ("scaler", StandardScaler()),
                                        ("clf", modelo)])
                pipe.fit(tr[colunas], tr["pre_falha"])
                acumulado[nome][cryst]["y"].extend(teste_c["pre_falha"].tolist())
                acumulado[nome][cryst]["s"].extend(pipe.predict_proba(teste_c[colunas])[:, 1].tolist())

    linhas = []
    for cryst in reatores:
        for nome in ["unificado", "proprio", "outros"]:
            y = np.array(acumulado[nome][cryst]["y"])
            s = np.array(acumulado[nome][cryst]["s"])
            if len(y) < 20 or y.sum() == 0:
                linhas.append({"reator": cryst, "escopo_treino": nome, "n_teste": len(y),
                               "positivos": int(y.sum()), "PR_AUC": np.nan,
                               "taxa_base": np.nan, "lift_PR": np.nan})
                continue
            ap = average_precision_score(y, s)
            linhas.append({"reator": cryst, "escopo_treino": nome, "n_teste": len(y),
                           "positivos": int(y.sum()), "PR_AUC": round(ap, 3),
                           "taxa_base": round(float(y.mean()), 3),
                           "lift_PR": round(ap / y.mean(), 2)})
    tabela = pd.DataFrame(linhas)
    if verbose:
        print(tabela.to_string(index=False))
        print("\nLeitura: nenhum escopo vence em todos os reatores, e 'próprio' não vence em "
              "nenhum — o C3, o único com sinal forte, aprende melhor com os OUTROS dois do que "
              "consigo mesmo. Treinar por reator não se sustenta; o ajuste fica unificado.")
    return tabela


def calibrar_limiares_por_reator(map_medicoes, map_falhas, margem=None, grade=None,
                                 coluna=COLUNA_FE, verbose=True):
    """Um limiar por reator para a regra composta, mantendo a MESMA estrutura de regra.

    O que muda entre reatores não é a lógica do alarme, é o ponto de corte: o C3 dispara
    com nível, o C1 com margem, e o C2 pede um corte mais permissivo. Cada reator é
    otimizado contando apenas os seus próprios alarmes e as suas próprias falhas.

    ATENÇÃO: com 6 a 14 falhas por reator, o limiar próprio é ajustado in-sample e é um
    teto otimista. Ele deve ser lido como "quanto se ganharia se o limiar fosse do
    equipamento", não como desempenho esperado — e revisto quando a planta trouxer mais
    eventos rotulados.
    """
    grade = grade or GRADE_REGRA_COMPOSTA
    if margem is None:
        _, _, margem = calcular_margem_cross_reator(map_medicoes, coluna, verbose=False)

    # limiar global: o que maximiza F2 somando os três reatores
    melhor_global, f2_global = None, -1
    for lmax in grade["limite_max"]:
        for lmed in grade["limite_mediana"]:
            for lmar in grade["limite_margem"]:
                m = metricas_alarmes_periodo(
                    alarmes_regra_composta(map_medicoes, margem, lmax, lmed, lmar, coluna),
                    map_falhas)
                if m["F2"] > f2_global:
                    f2_global, melhor_global = m["F2"], (lmax, lmed, lmar)

    linhas, limiares = [], {}
    for cryst in map_medicoes:
        um_reator = {cryst: map_medicoes[cryst]}
        uma_falha = {cryst: map_falhas[cryst]}
        m_glob = metricas_alarmes_periodo(
            alarmes_regra_composta(um_reator, margem, *melhor_global, coluna=coluna), uma_falha)

        melhor_loc, m_loc, f2_loc = None, None, -1
        for lmax in grade["limite_max"]:
            for lmed in grade["limite_mediana"]:
                for lmar in grade["limite_margem"]:
                    m = metricas_alarmes_periodo(
                        alarmes_regra_composta(um_reator, margem, lmax, lmed, lmar, coluna),
                        uma_falha)
                    if m["F2"] > f2_loc:
                        f2_loc, melhor_loc, m_loc = m["F2"], (lmax, lmed, lmar), m
        limiares[cryst] = melhor_loc
        linhas.append({"reator": cryst, "escopo": "limiar global", "limiares": str(melhor_global),
                       "VP": m_glob["VP"], "FP": m_glob["FP"], "precisao": m_glob["precisao"],
                       "recall": m_glob["recall"], "F2": m_glob["F2"]})
        linhas.append({"reator": cryst, "escopo": "limiar calibrado", "limiares": str(melhor_loc),
                       "VP": m_loc["VP"], "FP": m_loc["FP"], "precisao": m_loc["precisao"],
                       "recall": m_loc["recall"], "F2": m_loc["F2"]})

    tabela = pd.DataFrame(linhas)
    if verbose:
        print(f"Limiar global ótimo (max/mediana/margem): {melhor_global}  "
              f"— F2 somando os três reatores: {f2_global:.3f}\n")
        print(tabela.to_string(index=False))
    return limiares, tabela, melhor_global


def avaliar_regra_calibrada(map_medicoes, map_falhas, limiares_por_reator, margem=None,
                            limiares_globais=(20, 3.0, 0.6), periodos=None,
                            coluna=COLUNA_FE, verbose=True):
    """Desempenho total da regra com limiar por reator, contra o limiar único e a regra vigente."""
    if margem is None:
        _, _, margem = calcular_margem_cross_reator(map_medicoes, coluna, verbose=False)

    al_calibrada = {}
    for cryst in map_medicoes:
        lm = limiares_por_reator[cryst]
        al_calibrada[cryst] = alarmes_regra_composta(
            {cryst: map_medicoes[cryst]}, margem, *lm, coluna=coluna)[cryst]

    diario_max = serie_diaria(map_medicoes, "max", coluna)
    linhas = [
        metricas_alarmes_periodo(al_calibrada, map_falhas, periodos,
                                 rotulo="regra composta com limiar POR REATOR"),
        metricas_alarmes_periodo(
            alarmes_regra_composta(map_medicoes, margem, *limiares_globais, coluna=coluna),
            map_falhas, periodos, rotulo=f"regra composta com limiar único {limiares_globais}"),
        metricas_alarmes_periodo(
            {c: list(diario_max[c].dropna()[diario_max[c].dropna() > 5].index) for c in map_medicoes},
            map_falhas, periodos, rotulo="regra vigente (max diário > 5 ppm)"),
    ]
    tabela = pd.DataFrame(linhas)
    if verbose:
        print(tabela.to_string(index=False))
    return tabela, al_calibrada


def resumo_por_reator(map_medicoes, map_falhas, df_janelas=None, margem=None,
                      limiares_por_reator=None, df_lead=None, contexto=None,
                      coluna=COLUNA_FE, verbose=True):
    """Uma linha por reator com tudo o que a equipe precisa ver junto.

    Deliberadamente NÃO devolve um número agregado: a média entre reatores esconde que a
    regra funciona no C3 (recall 0.83) e falha no C2 (0.07), que é o fato mais importante
    para a decisão de implantação.
    """
    if margem is None:
        _, _, margem = calcular_margem_cross_reator(map_medicoes, coluna, verbose=False)
    diario_med = serie_diaria(map_medicoes, "median", coluna)

    linhas = []
    for cryst in sorted(map_medicoes):
        med = map_medicoes[cryst]
        falhas = map_falhas[cryst]
        linha = {
            "reator": cryst,
            "amostras": len(med),
            "periodo": f"{med['TIMESTAMP'].min():%Y}-{med['TIMESTAMP'].max():%Y}",
            "mediana_ppm": round(float(med[coluna].median()), 2),
            "p99_ppm": round(float(med[coluna].quantile(0.99)), 2),
            "falhas": len(falhas),
        }
        if df_janelas is not None:
            j = df_janelas[df_janelas["Crystallizer"] == cryst]
            linha["janelas"] = len(j)
            linha["taxa_base_pre_falha"] = round(float(j["pre_falha"].mean()), 4)

        # a qual estatística este reator responde
        s_med, s_mar = diario_med[cryst].dropna(), margem[cryst].dropna()
        alarmes_med = {cryst: list(s_med[s_med > 3.5].index)}
        alarmes_mar = {cryst: list(s_mar[s_mar > 0.6].index)}
        f2_med = metricas_alarmes_periodo(alarmes_med, {cryst: falhas})["F2"]
        f2_mar = metricas_alarmes_periodo(alarmes_mar, {cryst: falhas})["F2"]
        linha["F2_mediana>3.5"] = f2_med
        linha["F2_margem>0.6"] = f2_mar
        linha["responde_a"] = "nível" if f2_med > f2_mar else ("margem" if f2_mar > f2_med else "—")

        if limiares_por_reator is not None:
            lm = limiares_por_reator[cryst]
            m = metricas_alarmes_periodo(
                alarmes_regra_composta({cryst: med}, margem, *lm, coluna=coluna),
                {cryst: falhas})
            linha.update({"limiar_calibrado": str(lm), "VP": m["VP"], "FP": m["FP"],
                          "precisao": m["precisao"], "recall": m["recall"], "F2": m["F2"]})

        if df_lead is not None:
            d = df_lead[df_lead["Crystallizer"] == cryst]
            alarmadas = d[d["alarmou"]]
            linha["falhas_com_alarme_60d"] = f"{len(alarmadas)}/{len(d)}"
            linha["lead_mediano_dias"] = (round(float(alarmadas["lead_dias"].median()), 1)
                                          if len(alarmadas) else np.nan)

        if contexto is not None and contexto.get("campanhas"):
            idades = [idade_campanha(contexto["campanhas"], cryst, ts)
                      for ts in falhas["TIMESTAMP"]]
            idades = [i for i in idades if i == i]
            linha["idade_campanha_mediana"] = round(float(np.median(idades)), 0) if idades else np.nan

        linhas.append(linha)

    tabela = pd.DataFrame(linhas)
    if verbose:
        print(tabela.to_string(index=False))
        print("\nSem linha 'total' de propósito: a média entre reatores esconde que a regra "
              "funciona em um e falha em outro, que é justamente o que decide a implantação.")
    return tabela


# =============================================================================
# Otimização da regra de detecção: grade ampla, platô e validação walk-forward
# =============================================================================
#
# Motivo desta seção: a calibração original usava uma grade de 4x4x4 = 64 combinações
# e DOIS dos três reatores escolheram o valor da BORDA (limite_max = 30, o teto da
# grade). Quando o ótimo encosta na borda, ele não é ótimo — é o limite da busca. Além
# disso, escolher o argmax de F2 sobre 9 a 16 falhas por reator é otimização sobre
# ruído: o número que sai é um teto otimista, não desempenho esperado.
#
# O que esta seção acrescenta:
#   1. grade AMPLA e mais fina, incluindo o valor "desligado" (`inf` no limite de max ou
#      de mediana desliga aquele ramo; `-inf` na margem desliga a condição de margem) —
#      assim a própria busca descobre se um ramo da regra é peso morto;
#   2. avaliação RÁPIDA (arrays de dias, sem pandas no laço), porque a grade cresce de
#      64 para ~900 combinações por reator;
#   3. análise de PLATÔ: em vez do pico isolado, escolhe-se o ponto mais estável entre
#      os limiares que empatam dentro da tolerância — o pico isolado é a assinatura
#      clássica de sobreajuste;
#   4. validação WALK-FORWARD: calibra no passado, mede no bloco seguinte. É a única
#      estimativa honesta do procedimento "calibrar limiar com o histórico".

GRADE_REGRA_AMPLA = {
    "limite_max":     [5, 7, 10, 12, 15, 20, 25, 30, 40, 60, np.inf],
    "limite_mediana": [2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 4.0, 5.0, np.inf],
    "limite_margem":  [-np.inf, 0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.2, 1.6],
}


def _dias_int(indice):
    """Índice de datas -> número inteiro de dias (para aritmética rápida)."""
    return indice.values.astype("datetime64[D]").astype(np.int64)


def preparar_arrays_regra(map_medicoes, margem, coluna=COLUNA_FE):
    """Pré-calcula, por reator, os arrays diários usados na busca em grade.

    Sem isso cada combinação recalcularia `serie_diaria` — o que torna uma grade de
    900 combinações inviável.
    """
    diario_max = serie_diaria(map_medicoes, "max", coluna)
    diario_med = serie_diaria(map_medicoes, "median", coluna)
    arrays = {}
    for cryst in map_medicoes:
        s_max = diario_max[cryst].dropna()
        s_med = diario_med[cryst].dropna()
        s_mar = margem[cryst].dropna()
        # alinha os três no mesmo eixo de dias
        idx = s_max.index.union(s_med.index).union(s_mar.index)
        arrays[cryst] = {
            "dias": _dias_int(idx),
            "max": s_max.reindex(idx).values.astype(float),
            "med": s_med.reindex(idx).values.astype(float),
            "mar": s_mar.reindex(idx).values.astype(float),
        }
    return arrays


def _metricas_dias(dias_alarme, dias_falha, janela_alarme=15, agrupar_dias=15,
                   dias_neutros=None):
    """VP/FP/precisão/recall/F2 a partir de vetores de dias (inteiros).

    Replica exatamente a convenção de `_metricas_alarmes`: alarmes separados por até
    `agrupar_dias` do alarme ANTERIOR contam como um só (agrupamento por silêncio), e
    o acerto vale se a falha ocorre em até `janela_alarme` dias depois do alarme.
    """
    total = len(dias_falha)
    if len(dias_alarme) == 0:
        return {"VP": 0, "FP": 0, "precisao": 0.0, "recall": 0.0, "F2": 0.0}

    a = np.sort(dias_alarme)
    novo = np.empty(len(a), dtype=bool)
    novo[0] = True
    if len(a) > 1:
        novo[1:] = np.diff(a) > agrupar_dias
    a = a[novo]

    f = np.sort(dias_falha)
    if total:
        lo = np.searchsorted(a, f - janela_alarme, side="left")
        hi = np.searchsorted(a, f, side="right")
        vp = int((hi > lo).sum())
    else:
        vp = 0

    lo = np.searchsorted(f, a, side="left")
    hi = np.searchsorted(f, a + janela_alarme, side="right")
    sem_acerto = hi <= lo
    # alarme perto de uma falha fora do escopo não é falso positivo: ele acertou algo
    # real, apenas de um modo de falha que não está sendo avaliado
    if dias_neutros is not None and len(dias_neutros):
        n = np.sort(np.asarray(dias_neutros))
        lo_n = np.searchsorted(n, a, side="left")
        hi_n = np.searchsorted(n, a + janela_alarme, side="right")
        sem_acerto = sem_acerto & (hi_n <= lo_n)
    fp = int(sem_acerto.sum())

    precisao = vp / (vp + fp) if vp + fp else 0.0
    recall = vp / total if total else 0.0
    f2 = 5 * precisao * recall / (4 * precisao + recall) if precisao + recall else 0.0
    return {"VP": vp, "FP": fp, "precisao": round(precisao, 4),
            "recall": round(recall, 4), "F2": round(f2, 4)}


def _dias_alarme_regra(arr, lmax, lmed, lmar, mascara=None):
    """Dias em que a regra composta dispara, dado o dicionário de arrays de um reator."""
    disp = (arr["max"] > lmax) | ((arr["med"] > lmed) & (arr["mar"] > lmar))
    disp = np.nan_to_num(disp, nan=False)
    if mascara is not None:
        disp = disp & mascara
    return arr["dias"][disp]


def _mascara_periodos(dias, periodos):
    if periodos is None:
        return None
    m = np.zeros(len(dias), dtype=bool)
    for ini, fim in periodos:
        m |= (dias >= _dias_int(pd.DatetimeIndex([ini]))[0]) & \
             (dias <= _dias_int(pd.DatetimeIndex([fim]))[0])
    return m


def buscar_grade_regra(arr, dias_falha, grade=None, janela_alarme=15, agrupar_dias=15,
                       mascara=None):
    """Varre a grade inteira para UM reator e devolve um DataFrame com todas as métricas."""
    grade = grade or GRADE_REGRA_AMPLA
    linhas = []
    for lmax in grade["limite_max"]:
        for lmed in grade["limite_mediana"]:
            for lmar in grade["limite_margem"]:
                d = _dias_alarme_regra(arr, lmax, lmed, lmar, mascara)
                m = _metricas_dias(d, dias_falha, janela_alarme, agrupar_dias)
                linhas.append({"limite_max": lmax, "limite_mediana": lmed,
                               "limite_margem": lmar, **m})
    return pd.DataFrame(linhas)


def _escolha_estavel(df_grade, grade, metrica="F2", tolerancia=0.02):
    """Entre os limiares que empatam dentro da tolerância, escolhe o mais ESTÁVEL.

    Estabilidade = média da métrica na vizinhança do ponto na grade (±1 posição em cada
    eixo). Um pico isolado cercado de valores ruins é sobreajuste; um ponto no meio de
    um platô sobrevive a pequenas mudanças de limiar — e é o que se implanta.
    """
    eixos = ["limite_max", "limite_mediana", "limite_margem"]
    pos = {e: {v: i for i, v in enumerate(grade[e])} for e in eixos}
    chave = {}
    for _, r in df_grade.iterrows():
        chave[(pos["limite_max"][r["limite_max"]],
               pos["limite_mediana"][r["limite_mediana"]],
               pos["limite_margem"][r["limite_margem"]])] = r[metrica]

    melhor = df_grade[metrica].max()
    plato = df_grade[df_grade[metrica] >= melhor - tolerancia].copy()

    vizinhanca = []
    for _, r in plato.iterrows():
        i = (pos["limite_max"][r["limite_max"]],
             pos["limite_mediana"][r["limite_mediana"]],
             pos["limite_margem"][r["limite_margem"]])
        vals = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    v = chave.get((i[0] + di, i[1] + dj, i[2] + dk))
                    if v is not None:
                        vals.append(v)
        vizinhanca.append(np.mean(vals))
    plato["estabilidade"] = vizinhanca
    plato = plato.sort_values(["estabilidade", metrica, "precisao"], ascending=False)
    return plato


def otimizar_regra_por_reator(map_medicoes, map_falhas, margem=None, grade=None,
                              metrica="F2", tolerancia_plato=0.02, janela_alarme=15,
                              agrupar_dias=15, coluna=COLUNA_FE, verbose=True):
    """Busca em grade ampla, por reator, com análise de platô.

    Devolve, para cada reator: o argmax da métrica, a escolha ESTÁVEL (topo do platô) e
    a grade completa para inspeção da superfície. A escolha estável é a recomendada
    para implantação — ver `_escolha_estavel`.
    """
    grade = grade or GRADE_REGRA_AMPLA
    if margem is None:
        _, _, margem = calcular_margem_cross_reator(map_medicoes, coluna, verbose=False)
    arrays = preparar_arrays_regra(map_medicoes, margem, coluna)

    resultado = {}
    for cryst in map_medicoes:
        dias_falha = _dias_int(pd.DatetimeIndex(map_falhas[cryst]["TIMESTAMP"]))
        g = buscar_grade_regra(arrays[cryst], dias_falha, grade, janela_alarme, agrupar_dias)
        argmax = g.loc[g[metrica].idxmax()]
        plato = _escolha_estavel(g, grade, metrica, tolerancia_plato)
        estavel = plato.iloc[0]
        resultado[cryst] = {"grade": g, "argmax": argmax, "plato": plato,
                            "estavel": estavel, "n_plato": len(plato),
                            "n_falhas": len(dias_falha)}

    if verbose:
        linhas = []
        for cryst, r in resultado.items():
            a, e = r["argmax"], r["estavel"]
            linhas.append({
                "reator": cryst, "falhas": r["n_falhas"],
                "argmax (max/med/margem)": f"{a.limite_max}/{a.limite_mediana}/{a.limite_margem}",
                f"{metrica}_argmax": a[metrica],
                "escolha estável": f"{e.limite_max}/{e.limite_mediana}/{e.limite_margem}",
                f"{metrica}_estável": e[metrica], "VP": int(e.VP), "FP": int(e.FP),
                "combos no platô": r["n_plato"],
            })
        print(pd.DataFrame(linhas).to_string(index=False))
        print(f"\nGrade: {len(grade['limite_max'])}x{len(grade['limite_mediana'])}"
              f"x{len(grade['limite_margem'])} = "
              f"{len(grade['limite_max']) * len(grade['limite_mediana']) * len(grade['limite_margem'])}"
              f" combinações por reator.")
        print("`inf` no limite de max ou de mediana significa RAMO DESLIGADO; -inf na margem "
              "significa condição de margem sempre verdadeira.")
        print("A 'escolha estável' é o topo do platô (melhor média na vizinhança da grade), "
              "não o pico isolado — é ela que deve ir para implantação.")
    return resultado


def limiares_da_otimizacao(resultado, usar="estavel"):
    """Extrai {reator: (lmax, lmed, lmar)} do resultado de `otimizar_regra_por_reator`."""
    return {c: (r[usar]["limite_max"], r[usar]["limite_mediana"], r[usar]["limite_margem"])
            for c, r in resultado.items()}


def sensibilidade_limiares(resultado, cryst, metrica="F2", verbose=True):
    """Como a métrica responde a cada limiar isoladamente, fixando os outros dois no ótimo.

    Responde à pergunta prática: "se eu mexer um pouco neste número, quanto perco?"
    """
    r = resultado[cryst]
    e = r["estavel"]
    g = r["grade"]
    saida = {}
    for eixo, fixos in [("limite_max", ["limite_mediana", "limite_margem"]),
                        ("limite_mediana", ["limite_max", "limite_margem"]),
                        ("limite_margem", ["limite_max", "limite_mediana"])]:
        sel = g
        for f in fixos:
            sel = sel[sel[f] == e[f]]
        saida[eixo] = sel[[eixo, "VP", "FP", "precisao", "recall", metrica]] \
            .sort_values(eixo).reset_index(drop=True)
    if verbose:
        print(f"{cryst} — limiares no ponto estável: max>{e.limite_max}, "
              f"mediana>{e.limite_mediana}, margem>{e.limite_margem} "
              f"({metrica} {e[metrica]:.3f})")
        for eixo, tab in saida.items():
            print(f"\n  variando {eixo} (os outros dois fixos):")
            print("   " + tab.to_string(index=False).replace("\n", "\n   "))
    return saida


REFERENCIAS_REGRA = {
    # a regra vigente cabe na mesma família: só o ramo do máximo, em 5 ppm
    "regra vigente (max > 5)":     (5, np.inf, -np.inf),
    "composta fixa (20/3.0/0.6)":  (20, 3.0, 0.6),
    "simplificada (mediana>2.75 E margem>0.6)": (np.inf, 2.75, 0.6),
    "margem fixa (> 0.8)":         (np.inf, -np.inf, 0.8),
}


def validar_calibracao_walkforward(map_medicoes, map_falhas, margem=None, grade=None,
                                   n_blocos=3, metrica="F2", janela_alarme=15,
                                   agrupar_dias=15, referencias=None,
                                   coluna=COLUNA_FE, verbose=True):
    """Calibra no passado, mede no bloco seguinte — a estimativa honesta do procedimento.

    O F2 in-sample de uma grade de centenas de combinações sobre ~10 falhas por reator é
    um teto, não uma previsão. Aqui o limiar é escolhido usando SÓ o passado de cada
    corte e aplicado ao bloco seguinte, nunca visto. O agregado dos blocos é o que se
    pode prometer para a operação.

    Devolve a tabela por reator/bloco e o consolidado por reator.
    """
    grade = grade or GRADE_REGRA_AMPLA
    if margem is None:
        _, _, margem = calcular_margem_cross_reator(map_medicoes, coluna, verbose=False)
    arrays = preparar_arrays_regra(map_medicoes, margem, coluna)

    linhas = []
    for cryst in map_medicoes:
        ts_falhas = pd.DatetimeIndex(map_falhas[cryst]["TIMESTAMP"]).sort_values()
        if len(ts_falhas) < n_blocos + 1:
            continue
        arr = arrays[cryst]
        dias_todos = arr["dias"]
        cortes = [ts_falhas[int(len(ts_falhas) * (i + 1) / (n_blocos + 1))]
                  for i in range(n_blocos)]

        for i, corte in enumerate(cortes):
            fim = cortes[i + 1] if i + 1 < len(cortes) else pd.Timestamp(
                dias_todos.max(), unit="D") + pd.Timedelta(days=1)
            d_corte = _dias_int(pd.DatetimeIndex([corte]))[0]
            d_fim = _dias_int(pd.DatetimeIndex([fim]))[0]

            m_treino = dias_todos < d_corte
            m_teste = (dias_todos >= d_corte) & (dias_todos <= d_fim)
            df_f = _dias_int(ts_falhas)
            f_treino = df_f[df_f < d_corte]
            f_teste = df_f[(df_f >= d_corte) & (df_f <= d_fim)]
            if len(f_treino) < 2 or len(f_teste) < 1:
                continue

            g = buscar_grade_regra(arr, f_treino, grade, janela_alarme, agrupar_dias,
                                   mascara=m_treino)
            plato = _escolha_estavel(g, grade, metrica)
            e = plato.iloc[0]
            d_alarme = _dias_alarme_regra(arr, e.limite_max, e.limite_mediana,
                                          e.limite_margem, m_teste)
            m_out = _metricas_dias(d_alarme, f_teste, janela_alarme, agrupar_dias)
            linhas.append({"reator": cryst, "bloco": i + 1, "corte": corte.date(),
                           "regra": "calibrada no passado",
                           "limiares": f"{e.limite_max}/{e.limite_mediana}/{e.limite_margem}",
                           f"{metrica}_treino": e[metrica], "falhas_teste": len(f_teste),
                           **m_out})

            # mesmas fatias de teste, regras de referência (limiar fixo, sem calibração)
            for nome, (a, b, c) in (referencias or REFERENCIAS_REGRA).items():
                d_ref = _dias_alarme_regra(arr, a, b, c, m_teste)
                m_ref = _metricas_dias(d_ref, f_teste, janela_alarme, agrupar_dias)
                linhas.append({"reator": cryst, "bloco": i + 1, "corte": corte.date(),
                               "regra": nome, "limiares": f"{a}/{b}/{c}",
                               f"{metrica}_treino": np.nan, "falhas_teste": len(f_teste),
                               **m_ref})

    df = pd.DataFrame(linhas)
    if df.empty:
        print("Amostra insuficiente para walk-forward.")
        return df, pd.DataFrame()

    def _consolida(chaves):
        r = (df.groupby(chaves)
             .agg(blocos=("bloco", "size"), falhas_teste=("falhas_teste", "sum"),
                  VP=("VP", "sum"), FP=("FP", "sum"),
                  F2_treino_medio=(f"{metrica}_treino", "mean"))
             .reset_index())
        r["precisao"] = (r["VP"] / (r["VP"] + r["FP"]).replace(0, np.nan)).fillna(0).round(3)
        r["recall"] = (r["VP"] / r["falhas_teste"].replace(0, np.nan)).fillna(0).round(3)
        r["F2_fora_da_amostra"] = (
            5 * r["precisao"] * r["recall"] /
            (4 * r["precisao"] + r["recall"]).replace(0, np.nan)).fillna(0).round(3)
        return r

    resumo = _consolida(["reator", "regra"])
    resumo["otimismo"] = (resumo["F2_treino_medio"] - resumo["F2_fora_da_amostra"]).round(3)
    consolidado = _consolida(["regra"]).sort_values("F2_fora_da_amostra", ascending=False)

    if verbose:
        print(df.to_string(index=False))
        print("\nPor reator e regra (VP/FP somados sobre os blocos de teste):")
        print(resumo.to_string(index=False))
        print("\nConsolidado nos três reatores — a comparação que decide:")
        print(consolidado.to_string(index=False))
        print("\n`otimismo` = F2 in-sample menos F2 fora da amostra: o quanto a calibração "
              "promete a mais do que entrega. As linhas de referência não são calibradas, "
              "então são medidas nas MESMAS fatias de teste sem nenhuma vantagem informacional.")
    return df, resumo, consolidado


def ablacao_ramos_regra(map_medicoes, map_falhas, limiares_por_reator, margem=None,
                        janela_alarme=15, agrupar_dias=15, coluna=COLUNA_FE, verbose=True):
    """Qual ramo da regra carrega o resultado em cada reator?

    Compara a regra completa com cada ramo isolado, usando os limiares calibrados. Se um
    ramo sozinho empata com a regra completa, o outro é peso morto naquele equipamento —
    e simplificar a regra é ganho de implantação, não perda.
    """
    if margem is None:
        _, _, margem = calcular_margem_cross_reator(map_medicoes, coluna, verbose=False)
    arrays = preparar_arrays_regra(map_medicoes, margem, coluna)

    linhas = []
    for cryst in map_medicoes:
        lmax, lmed, lmar = limiares_por_reator[cryst]
        arr = arrays[cryst]
        dias_falha = _dias_int(pd.DatetimeIndex(map_falhas[cryst]["TIMESTAMP"]))
        variantes = {
            "regra completa": (lmax, lmed, lmar),
            "só o ramo do máximo": (lmax, np.inf, lmar),
            "só o ramo mediana+margem": (np.inf, lmed, lmar),
            "só mediana (sem margem)": (np.inf, lmed, -np.inf),
        }
        for nome, (a, b, c) in variantes.items():
            d = _dias_alarme_regra(arr, a, b, c)
            m = _metricas_dias(d, dias_falha, janela_alarme, agrupar_dias)
            linhas.append({"reator": cryst, "variante": nome,
                           "limiares": f"{a}/{b}/{c}", **m})
    df = pd.DataFrame(linhas)
    if verbose:
        print(df.to_string(index=False))
    return df


def _dias_falha_por_reator(map_falhas):
    return {c: _dias_int(pd.DatetimeIndex(df["TIMESTAMP"])) for c, df in map_falhas.items()}


def comparar_estrategias_calibracao(map_medicoes, map_falhas, margem=None, grade=None,
                                    base=(20, 3.0, 0.6), n_blocos=3, metrica="F2",
                                    janela_alarme=15, agrupar_dias=15, coluna=COLUNA_FE,
                                    verbose=True):
    """Quantos graus de liberdade esta base sustenta na calibração dos limiares?

    Todas as estratégias são medidas por WALK-FORWARD com os MESMOS cortes de tempo
    (definidos sobre as falhas dos três reatores juntos), então a comparação é limpa:

      - `fixa`                  — nenhum parâmetro estimado do histórico (o limiar `base`);
      - `só margem (por reator)`— 1 parâmetro por reator, os outros dois fixos;
      - `só mediana (por reator)`— idem, na mediana;
      - `3 params (global)`     — 3 parâmetros estimados, mas com os três reatores juntos;
      - `3 params (por reator)` — 3 parâmetros por reator = 9 no total.

    Quanto mais parâmetros, melhor o ajuste no passado e — se a amostra não sustentar —
    pior o desempenho no futuro. Esta função mede exatamente essa troca.
    """
    grade = grade or GRADE_REGRA_AMPLA
    if margem is None:
        _, _, margem = calcular_margem_cross_reator(map_medicoes, coluna, verbose=False)
    arrays = preparar_arrays_regra(map_medicoes, margem, coluna)
    dias_falha = _dias_falha_por_reator(map_falhas)

    todas = np.sort(np.concatenate([v for v in dias_falha.values()]))
    cortes = [todas[int(len(todas) * (i + 1) / (n_blocos + 1))] for i in range(n_blocos)]

    def _melhor(grade_local, mascaras, dias_treino, pooled):
        """Escolhe limiares maximizando a métrica no treino (pooled ou por reator)."""
        melhor, valor = None, -1
        for lmax in grade_local["limite_max"]:
            for lmed in grade_local["limite_mediana"]:
                for lmar in grade_local["limite_margem"]:
                    vp = fp = tot = 0
                    for c in pooled:
                        d = _dias_alarme_regra(arrays[c], lmax, lmed, lmar, mascaras[c])
                        m = _metricas_dias(d, dias_treino[c], janela_alarme, agrupar_dias)
                        vp += m["VP"]; fp += m["FP"]; tot += len(dias_treino[c])
                    p = vp / (vp + fp) if vp + fp else 0.0
                    r = vp / tot if tot else 0.0
                    f2 = 5 * p * r / (4 * p + r) if p + r else 0.0
                    if f2 > valor:
                        valor, melhor = f2, (lmax, lmed, lmar)
        return melhor, valor

    g_margem = {"limite_max": [base[0]], "limite_mediana": [base[1]],
                "limite_margem": grade["limite_margem"]}
    g_mediana = {"limite_max": [base[0]], "limite_mediana": grade["limite_mediana"],
                 "limite_margem": [base[2]]}

    estrategias = [
        ("fixa (0 parâmetros)", None, None),
        ("só margem (1 por reator)", g_margem, "reator"),
        ("só mediana (1 por reator)", g_mediana, "reator"),
        ("3 params (global)", grade, "global"),
        ("3 params (por reator)", grade, "reator"),
    ]

    linhas = []
    for nome, g, escopo in estrategias:
        for i, corte in enumerate(cortes):
            fim = cortes[i + 1] if i + 1 < len(cortes) else max(
                a["dias"].max() for a in arrays.values()) + 1
            m_treino = {c: arrays[c]["dias"] < corte for c in arrays}
            m_teste = {c: (arrays[c]["dias"] >= corte) & (arrays[c]["dias"] <= fim) for c in arrays}
            f_treino = {c: v[v < corte] for c, v in dias_falha.items()}
            f_teste = {c: v[(v >= corte) & (v <= fim)] for c, v in dias_falha.items()}
            if sum(len(v) for v in f_treino.values()) < 4 or sum(len(v) for v in f_teste.values()) < 1:
                continue

            if escopo is None:
                lim = {c: base for c in arrays}
            elif escopo == "global":
                lm, _ = _melhor(g, m_treino, f_treino, list(arrays))
                lim = {c: lm for c in arrays}
            else:
                lim = {}
                for c in arrays:
                    if len(f_treino[c]) < 2:
                        lim[c] = base
                        continue
                    lm, _ = _melhor(g, {c: m_treino[c]}, {c: f_treino[c]}, [c])
                    lim[c] = lm

            vp = fp = tot = 0
            for c in arrays:
                d = _dias_alarme_regra(arrays[c], *lim[c], m_teste[c])
                m = _metricas_dias(d, f_teste[c], janela_alarme, agrupar_dias)
                vp += m["VP"]; fp += m["FP"]; tot += len(f_teste[c])
            linhas.append({"estrategia": nome, "bloco": i + 1, "falhas_teste": tot,
                           "VP": vp, "FP": fp})

    df = pd.DataFrame(linhas)
    resumo = df.groupby("estrategia").agg(blocos=("bloco", "size"),
                                          falhas_teste=("falhas_teste", "sum"),
                                          VP=("VP", "sum"), FP=("FP", "sum")).reset_index()
    resumo["precisao"] = (resumo["VP"] / (resumo["VP"] + resumo["FP"]).replace(0, np.nan)).fillna(0).round(3)
    resumo["recall"] = (resumo["VP"] / resumo["falhas_teste"]).round(3)
    resumo["F2_fora_da_amostra"] = (
        5 * resumo["precisao"] * resumo["recall"] /
        (4 * resumo["precisao"] + resumo["recall"]).replace(0, np.nan)).fillna(0).round(3)
    resumo["n_parametros"] = resumo["estrategia"].map(
        {"fixa (0 parâmetros)": 0, "só margem (1 por reator)": 3, "só mediana (1 por reator)": 3,
         "3 params (global)": 3, "3 params (por reator)": 9})
    resumo = resumo.sort_values("F2_fora_da_amostra", ascending=False)

    if verbose:
        print(resumo.to_string(index=False))
        print("\nLeitura: se as estratégias com mais parâmetros ficam ABAIXO da fixa, a amostra "
              "não sustenta a calibração — o limiar estimado está decorando o passado. Note que "
              "a linha 'fixa' leva vantagem embutida (o limiar base foi escolhido olhando a série "
              "inteira), então ela é o piso otimista da comparação, não um teto.")
    return df, resumo


def comparar_limiares_por_reator(map_medicoes, map_falhas, conjuntos, margem=None,
                                 janela_alarme=15, agrupar_dias=15, coluna=COLUNA_FE,
                                 verbose=True):
    """Compara conjuntos de limiares lado a lado, reator a reator.

    `conjuntos` : {nome: {reator: (limite_max, limite_mediana, limite_margem)}}

    Serve para responder "o que está em uso é melhor ou pior que o que a busca encontrou?"
    sem precisar mexer nas funções internas de avaliação.
    """
    if margem is None:
        _, _, margem = calcular_margem_cross_reator(map_medicoes, coluna, verbose=False)
    arrays = preparar_arrays_regra(map_medicoes, margem, coluna)

    linhas = []
    for nome, limiares in conjuntos.items():
        for cryst in map_medicoes:
            lm = limiares[cryst]
            dias_falha = _dias_int(pd.DatetimeIndex(map_falhas[cryst]["TIMESTAMP"]))
            m = _metricas_dias(_dias_alarme_regra(arrays[cryst], *lm),
                               dias_falha, janela_alarme, agrupar_dias)
            linhas.append({"reator": cryst, "conjunto": nome,
                           "limiares": "/".join(str(x) for x in lm),
                           "falhas": len(dias_falha), **m})
    df = pd.DataFrame(linhas).sort_values(["reator", "conjunto"]).reset_index(drop=True)
    if verbose:
        print(df.to_string(index=False))
    return df


# =============================================================================
# Modo de falha: quais eventos o ferro tem como enxergar
# =============================================================================
#
# O ferro só sobe no licor quando o revestimento vitrificado rompe e expõe aço carbono
# à solução. Falha de agitador sem exposição de aço, vazamento externo por junta/selo e
# vazamento em plug de reparo (área minúscula) não têm por que mover a medição — exigir
# que o detector as preveja é medir contra um denominador impossível.
#
# `classificar_modo_falha` faz uma classificação POR TEXTO, que é um proxy: a
# classificação definitiva tem de vir da planta (é o pedido nº 1 da entrega). Ela existe
# para responder "quanto do desempenho medido é limitação do método e quanto é limitação
# física do sinal?" — e a resposta muda o que se pode prometer.


# ordem importa: a primeira regra que casar define o modo. A busca é feita em
# OCORRIMENTO + OBSERVAÇÕES (o que aconteceu) e NÃO em SERVIÇOS EXECUTADOS (o que foi
# feito) — instalar um plug é o reparo de um furo, não o modo de falha. Ignorar essa
# distinção classificava 21 das 36 falhas como "plug", inclusive as duas paradas que a
# planilha atribui ao ferro.
REGRAS_MODO_FALHA = [
    ("plug/reparo",
     r"reinstala[çc][ãa]o de plug|substitui[çc][ãa]o do plug|"
     r"(plug|reparo|luva)[^.]{0,60}(infiltra|danificad|vazamento|solt)|infiltra[çc]"),
    ("revestimento/casco",
     r"furo (no|do|passante)|furo n[oa] (costado|tampo|a[çc]o|revestimento|reator)|"
     r"quebra do (vidro|revestimento)|quebra/furo|falha do revestimento|"
     r"perda de material|at[ée] a parte met[áa]lica|poro passante|perfura"),
    ("eixo/agitador",
     r"eixo|agitador|h[ée]lice|p[áa] superior|p[áa] do|parafus|baffle"),
    ("bocal/tampa/selo",
     r"bocal|tampa|\bbv\b|selo|junta|domo|deep ?pipe"),
    ("desgaste/constatação em inspeção",
     r"desgaste|poro|concavidade|espessura|rugosidade|dano|trinca"),
]

# modos em que a física permite ferro no licor: aço exposto à solução
MODOS_COM_FERRO = ("revestimento/casco", "eixo/agitador")

# Correções do modo de falha, quando o texto do OCORRIMENTO descreve o gatilho da parada
# (a análise de ferro) e não o dano — que aparece no registro da inspeção subsequente.
# Mesmo padrão de CORRECOES_INSPECAO: (reator, data ancorada, modo, fonte).
CORRECOES_MODO_FALHA = [
    ("C3", "2013-11-23", "revestimento/casco",
     "planilha: parada de emergência por ferro > 200 ppm; reator substituído em 30/11 com furo confirmado"),
    ("C3", "2018-08-02", "revestimento/casco",
     "planilha: 'quebra do revestimento vitríficado seguido de furo no costado ... perda de material'"),
]


def classificar_modo_falha(df_inspecoes, coluna_saida="ModoFalha"):
    """Classifica cada apontamento por modo de falha a partir do texto livre.

    **É um proxy.** A classificação definitiva tem de vir da planta — é o pedido nº 1 da
    entrega, e a coluna existe justamente para receber a resposta deles. Aqui ela serve
    para uma pergunta específica: *quanto do desempenho medido é limitação do método e
    quanto é limitação física do sinal?*

    Colunas novas:
      `ModoFalha`      — plug/reparo | revestimento/casco | eixo/agitador |
                         bocal/tampa/selo | desgaste/constatação em inspeção | indefinido
      `FerroPlausivel` — o modo permite, fisicamente, ferro no licor (aço exposto)?
    """
    df = df_inspecoes.copy()
    # só o que ACONTECEU, não o que foi feito
    texto = (df["Ocorrimento"].fillna("") + " " + df["Observacoes"].fillna("")).str.lower()

    modo = pd.Series("indefinido", index=df.index)
    for nome, padrao in REGRAS_MODO_FALHA:
        casa = texto.str.contains(padrao, regex=True) & (modo == "indefinido")
        modo[casa] = nome

    df[coluna_saida] = modo
    df["ModoCorrigido"] = False
    for cryst, data, valor, _fonte in CORRECOES_MODO_FALHA:
        m = (df["Crystallizer"] == cryst) &             (pd.to_datetime(df["TS_Ajustado"]).dt.strftime("%Y-%m-%d") == data)
        df.loc[m, coluna_saida] = valor
        df.loc[m, "ModoCorrigido"] = True
    df["FerroPlausivel"] = df[coluna_saida].isin(MODOS_COM_FERRO)
    return df


def avaliar_regra_por_subconjunto(map_medicoes, df_inspecoes, limiares, coluna_grupo,
                                  margem=None, neutralizar=True, janela_alarme=15,
                                  agrupar_dias=15, coluna=COLUNA_FE, verbose=True):
    """Desempenho da regra quando o denominador é restrito a um subconjunto de falhas.

    Responde à pergunta "e se a planta só cobrar do detector as falhas que o ferro tem
    como enxergar?". Cada linha usa o MESMO alarme; o que muda é quais falhas contam.

    `neutralizar=True`: um alarme que cai perto de uma falha fora do subconjunto **não**
    conta como falso positivo — ele acertou algo real, apenas fora do escopo avaliado.
    A coluna `FP_estrito` mostra a leitura pessimista (a falha excluída é ignorada e o
    alarme vira falso positivo). A diferença entre as duas é, em si, informativa.
    """
    if margem is None:
        _, _, margem = calcular_margem_cross_reator(map_medicoes, coluna, verbose=False)
    arrays = preparar_arrays_regra(map_medicoes, margem, coluna)
    sel = df_inspecoes[df_inspecoes["Selecionado"]]

    grupos = {"TODAS as falhas": sel}
    for valor in sorted(sel[coluna_grupo].astype(str).unique()):
        grupos[f"só: {valor}"] = sel[sel[coluna_grupo].astype(str) == valor]

    linhas = []
    for nome, g in grupos.items():
        if g.empty:
            continue
        vp = fp = fp_estrito = tot = 0
        for cryst in map_medicoes:
            alvo = _dias_int(pd.DatetimeIndex(g[g["Crystallizer"] == cryst]["TS_Ajustado"]))
            fora = _dias_int(pd.DatetimeIndex(
                sel[(sel["Crystallizer"] == cryst) & (~sel.index.isin(g.index))]["TS_Ajustado"]))
            d = _dias_alarme_regra(arrays[cryst], *limiares)
            m = _metricas_dias(d, alvo, janela_alarme, agrupar_dias,
                               dias_neutros=fora if neutralizar else None)
            m_est = _metricas_dias(d, alvo, janela_alarme, agrupar_dias)
            vp += m["VP"]; fp += m["FP"]; fp_estrito += m_est["FP"]; tot += len(alvo)

        def _m(v, f, t):
            p = v / (v + f) if v + f else 0.0
            r = v / t if t else 0.0
            return round(p, 3), round(r, 3), (round(5 * p * r / (4 * p + r), 3) if p + r else 0.0)

        p, r, f2 = _m(vp, fp, tot)
        _, _, f2e = _m(vp, fp_estrito, tot)
        linhas.append({"subconjunto": nome, "n_falhas": tot, "VP": vp, "FP": fp,
                       "precisao": p, "recall": r, "F2": f2,
                       "FP_estrito": fp_estrito, "F2_estrito": f2e})

    df = pd.DataFrame(linhas)
    if verbose:
        print(df.to_string(index=False))
        print("\nFP / F2 : alarme perto de falha excluída é NEUTRO (leitura operacional).")
        print("FP_estrito / F2_estrito : essa falha é ignorada e o alarme vira falso positivo.")
    return df
