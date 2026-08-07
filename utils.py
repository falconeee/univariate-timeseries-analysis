"""Funções auxiliares do estudo de contaminação por ferro nos crystallizers.

Concentra tudo o que o notebook `Crystallizer #1#2#3.ipynb` repetia célula a célula:
carga e limpeza, tratamentos de outlier, recortes por janela, gráficos e os
pipelines de clusterização, classificação, IsolationForest e cartas de controle.

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

from scipy.stats import skew, kurtosis, mannwhitneyu, ks_2samp, linregress

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, BisectingKMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (confusion_matrix, classification_report, adjusted_rand_score,
                             normalized_mutual_info_score, make_scorer, f1_score, precision_score,
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

# Mesmo conjunto, usado pela clusterização (mantém as features absolutas)
STATS_FUNCS_CLUSTERING = dict(STATS_FUNCS)

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
    # limite_ppm=None desliga o corte fixo — use `limpar_excursoes`, que preserva
    # as excursões reais e descarta só as leituras altas isoladas.
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


def _normalizar_cabecalho(valor):
    """Normaliza o nome da coluna: sem acento, sem plural, minusculo."""
    if valor is None:
        return ""
    txt = unicodedata.normalize("NFKD", str(valor)).encode("ascii", "ignore").decode()
    txt = re.sub(r"[^a-z ]", " ", txt.lower())
    txt = re.sub(r"\s+", " ", txt).strip()
    # as abas alternam entre singular e plural
    return {"tipos de inspecao": "tipo de inspecao", "observacao": "observacoes"}.get(txt, txt)


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
            linhas.append({
                "Crystallizer": crystallizer,
                "Tag":          ws.title,
                "Ano":          ano_corrente,
                "DataTexto":    "" if reg.get("data") is None else str(reg["data"]),
                "Inicio":       inicio,
                "Fim":          fim,
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
    df["Emergencia"]    = texto.str.contains(r"emerg|n[ãa]o programad", regex=True)
    df["Programada"]    = texto.str.contains(r"programad|preventiv|tempo de opera|tempo de vida|vida [uú]til", regex=True)
    df["Vazamento"]     = texto.str.contains(r"vazamento|furo|trinca|quebra|perfura", regex=True)
    df["TrocaDoReator"] = texto.str.contains(
        r"substitui[çc][ãa]o do reator|equipamento novo|reator novo|troca do reator|"
        r"substitui[çc][ãa]o do equipamento|equipamento substitu[íi]do|pelo spare", regex=True)
    df["FerroCitado"]   = texto.str.contains("ferro")
    df["Falha"]         = df["Emergencia"] | df["Vazamento"] | df["TrocaDoReator"]

    return df.sort_values(["Crystallizer", "Inicio"]).reset_index(drop=True)


def ajustar_eventos_para_lacunas(df_eventos, map_medicoes, dias_lacuna=2,
                                 coluna_ts="Inicio", incluir_ultima_medicao=True):
    """Reancora os eventos que caem dentro de uma lacuna de amostragem.

    Quando o reator para, a amostragem para junto: o apontamento da planilha
    costuma cair no meio do período sem medição. Nesses casos o evento é
    deslocado para a última medição anterior à lacuna, de modo que a janela
    `[ts - dias, ts)` cubra os dados que realmente antecedem a parada.

    map_medicoes           : {crystallizer: df_medicoes}
    dias_lacuna            : intervalo entre medições consecutivas a partir do
                             qual o trecho é considerado lacuna
    incluir_ultima_medicao : se True, ancora um segundo depois da última medição,
                             para que ela entre na janela (é a amostra que
                             costuma disparar a parada)

    Acrescenta as colunas TS_Ajustado, Deslocado, DiasDeslocado e LacunaDias.
    """
    df = df_eventos.copy()
    ajustados, deslocados, dias_desl, lacunas = [], [], [], []

    for _, ev in df.iterrows():
        ts = ev[coluna_ts]
        med = map_medicoes.get(ev["Crystallizer"])

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

    df["TS_Ajustado"]   = ajustados
    df["Deslocado"]     = deslocados
    df["DiasDeslocado"] = dias_desl
    df["LacunaDias"]    = lacunas
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


def limpar_excursoes(df_medicoes, teto_implausivel=1000, coluna=COLUNA_FE, verbose=True, **kwargs):
    """Remove erros de laboratório preservando as excursões reais.

    Duas camadas, no lugar do corte fixo de 100 ppm:

      1. teto de implausibilidade — valores acima de `teto_implausivel` ppm são
         fisicamente impossíveis na operação (na base inteira só existe 1 amostra
         acima de 1000 ppm; a segunda maior é 999);
      2. `classificar_amostras_altas` — descarta apenas as leituras altas isoladas.

    Retorna (df_limpo, df_descartado).
    """
    df = df_medicoes.copy()

    acima_teto = df[coluna] > teto_implausivel
    implausiveis = df[acima_teto].copy()
    implausiveis["Motivo"] = "acima do teto de implausibilidade"
    df = df[~acima_teto]

    classificado = classificar_amostras_altas(df, coluna=coluna, **kwargs)
    erros = classificado[classificado["ErroLab"]].copy()
    erros["Motivo"] = "leitura alta isolada"
    df_limpo = classificado[~classificado["ErroLab"]].drop(
        columns=["Alta", "Corroborada", "SeguidaDeParada", "ExcursaoReal", "ErroLab"]
    ).reset_index(drop=True)

    descartado = pd.concat([implausiveis, erros[list(implausiveis.columns)]], ignore_index=True)
    descartado = descartado.sort_values("TIMESTAMP").reset_index(drop=True)

    if verbose:
        n_real = int(classificado["ExcursaoReal"].sum())
        print(f"Amostras acima do teto de {teto_implausivel} ppm : {len(implausiveis)}")
        print(f"Leituras altas isoladas (erro de laboratório): {len(erros)}")
        print(f"Excursões reais preservadas                 : {n_real}")

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
      2. marca os que caem numa parada de planta;
      3. mantém apenas os apontamentos que são falha de equipamento;
      4. funde apontamentos do mesmo reator a menos de `fundir_dias` dias — a
         planilha frequentemente registra a mesma parada em duas linhas.

    Retorna a tabela completa com a coluna `Selecionado` marcando as falhas que
    devem ser usadas como positivos.
    """
    if paradas_planta is None:
        paradas_planta = identificar_paradas_de_planta(map_medicoes)

    df = ajustar_eventos_para_lacunas(inspecoes, map_medicoes, dias_lacuna=dias_lacuna)

    inicio_serie = min(m["TIMESTAMP"].min() for m in map_medicoes.values())
    df["NoPeriodo"] = df["Inicio"] >= inicio_serie
    df["LacunaDePlanta"] = [
        any(ini <= ts <= fim for ini, fim in paradas_planta) if pd.notna(ts) else False
        for ts in df["Inicio"]
    ]

    candidatos = df["NoPeriodo"] & df["Falha"] & ~df["LacunaDePlanta"]

    # funde apontamentos consecutivos do mesmo reator
    df["Selecionado"] = False
    for cryst in df["Crystallizer"].unique():
        sel = df[candidatos & (df["Crystallizer"] == cryst)].sort_values("TS_Ajustado")
        ultimo = None
        for i, linha in sel.iterrows():
            if ultimo is None or (linha["TS_Ajustado"] - ultimo).days > fundir_dias:
                df.loc[i, "Selecionado"] = True
            ultimo = linha["TS_Ajustado"]

    if verbose:
        print(f"Apontamentos na planilha            : {len(df)}")
        print(f"  dentro do período da série        : {int(df['NoPeriodo'].sum())}")
        print(f"  reancorados por cair em lacuna    : {int((df['NoPeriodo'] & df['Deslocado']).sum())}")
        print(f"  em parada de planta (descartados) : {int((df['NoPeriodo'] & df['LacunaDePlanta']).sum())}")
        print(f"  falhas selecionadas               : {int(df['Selecionado'].sum())}")
        print()
        print(df[df["Selecionado"]].groupby("Crystallizer").agg(
            falhas=("TS_Ajustado", "size"),
            emergencia=("Emergencia", "sum"),
            troca_reator=("TrocaDoReator", "sum"),
            cita_ferro=("FerroCitado", "sum"),
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
# Clusterização (não supervisionada)
# =============================================================================

def extrair_features_clustering(nome_c, df_medicoes, df_eventos, janelas, stats_funcs):
    dataset_linhas = []
    for _, evento in df_eventos.iterrows():
        ts     = evento["TIMESTAMP"]
        classe = int(evento["Real"])
        features = {
            'Crystallizer':     nome_c,
            'TIMESTAMP_Evento': ts,
            'Real':             classe,
        }
        for dias in janelas:
            y_ppm = valores_na_janela(df_medicoes, ts, dias)
            if len(y_ppm) < 2:
                for nome_stat in stats_funcs:
                    features[f"{nome_stat}_{dias}d"] = np.nan
                continue
            for nome_stat, func in stats_funcs.items():
                features[f"{nome_stat}_{dias}d"] = float(func(y_ppm))

        if not any(not np.isnan(features.get(f"media_{d}d", np.nan))
                   for d in janelas):
            continue
        dataset_linhas.append(features)

    colunas_meta = ['Crystallizer', 'TIMESTAMP_Evento', 'Real']
    df = pd.DataFrame(dataset_linhas)
    colunas_features = sorted(
        [c for c in df.columns if c not in colunas_meta],
        key=lambda c: (c.split('_')[0], int(c.split('_')[-1].replace('d', '')))
    )
    return df[colunas_meta + colunas_features], colunas_features


def pipeline_clustering(df_feat, colunas_features, label_dataset):
    # Imputação
    imputer  = SimpleImputer(strategy='median')
    X_imp    = pd.DataFrame(
        imputer.fit_transform(df_feat[colunas_features]),
        columns=colunas_features, index=df_feat.index
    )

    # Scaling
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    # Grid search
    resultados = []
    for n_clusters in range(2, 8):
        modelos = {
            'Hierarchical (ward)':     AgglomerativeClustering(n_clusters=n_clusters, linkage='ward'),
            'Hierarchical (complete)': AgglomerativeClustering(n_clusters=n_clusters, linkage='complete'),
            'Hierarchical (average)':  AgglomerativeClustering(n_clusters=n_clusters, linkage='average'),
            'GMM':                     GaussianMixture(n_components=n_clusters, random_state=42, n_init=10),
            'Bisecting KMeans':        BisectingKMeans(n_clusters=n_clusters, random_state=42, n_init=10),
            'Spectral':                SpectralClustering(n_clusters=n_clusters,
                                                          affinity='nearest_neighbors',
                                                          n_neighbors=10, random_state=42),
        }
        for nome, modelo in modelos.items():
            labels = modelo.fit_predict(X_scaled)
            ari    = adjusted_rand_score(df_feat['Real'], labels)
            nmi    = normalized_mutual_info_score(df_feat['Real'], labels)
            resultados.append({
                'Dataset': label_dataset, 'Modelo': nome,
                'n_clusters': n_clusters,
                'ARI': round(ari, 4), 'NMI': round(nmi, 4),
            })

    df_grid  = pd.DataFrame(resultados).sort_values('ARI', ascending=False)
    melhor   = df_grid.iloc[0]
    melhor_n = int(melhor['n_clusters'])
    melhor_mod = melhor['Modelo']

    # Clusterização final
    modelos_final = {
        'Hierarchical (ward)':     AgglomerativeClustering(n_clusters=melhor_n, linkage='ward'),
        'Hierarchical (complete)': AgglomerativeClustering(n_clusters=melhor_n, linkage='complete'),
        'Hierarchical (average)':  AgglomerativeClustering(n_clusters=melhor_n, linkage='average'),
        'GMM':                     GaussianMixture(n_components=melhor_n, random_state=42, n_init=10),
        'Bisecting KMeans':        BisectingKMeans(n_clusters=melhor_n, random_state=42, n_init=10),
        'Spectral':                SpectralClustering(n_clusters=melhor_n,
                                                      affinity='nearest_neighbors',
                                                      n_neighbors=10, random_state=42),
    }
    df_feat = df_feat.copy()
    df_feat['Cluster'] = modelos_final[melhor_mod].fit_predict(X_scaled)

    ari_f = adjusted_rand_score(df_feat['Real'], df_feat['Cluster'])
    nmi_f = normalized_mutual_info_score(df_feat['Real'], df_feat['Cluster'])

    return {
        'df_features': df_feat,
        'X_scaled':    X_scaled,
        'df_grid':     df_grid,
        'melhor_mod':  melhor_mod,
        'melhor_n':    melhor_n,
        'ari':         ari_f,
        'nmi':         nmi_f,
    }


def rodar_clusterizacao(crystallizer, configs, janelas, stats_funcs=None):
    """Roda o pipeline de clusterização em todos os tratamentos de um crystallizer.

    configs : {nome_tratamento: (df_medicoes, df_eventos)}
    Retorna {nome_tratamento: resultado do pipeline_clustering}.
    """
    if stats_funcs is None:
        stats_funcs = STATS_FUNCS_CLUSTERING

    resultados_por_dataset = {}

    for nome_dataset, (df_med, df_ev) in configs.items():
        df_feat, cols_feat = extrair_features_clustering(
            crystallizer, df_med, df_ev, janelas, stats_funcs
        )
        res = pipeline_clustering(df_feat, cols_feat, nome_dataset)
        resultados_por_dataset[nome_dataset] = res

        print(f"\n{'='*55}")
        print(f"{crystallizer} — {nome_dataset}")
        print(f"  Melhor modelo : {res['melhor_mod']}  (k={res['melhor_n']})")
        print(f"  ARI           : {res['ari']:.4f}")
        print(f"  NMI           : {res['nmi']:.4f}")
        print("\n  Tabela de contingência:")
        print(pd.crosstab(
            res['df_features']['Cluster'], res['df_features']['Real'],
            rownames=['Cluster'], colnames=['Real'],
            margins=True, margins_name='Total'
        ))

    return resultados_por_dataset


def plotar_clusterizacao_pca(resultados_por_dataset, titulo):
    """PCA 2D dos clusters encontrados vs o label real, um par de subplots por tratamento."""
    n_datasets     = len(resultados_por_dataset)
    nomes_datasets = list(resultados_por_dataset.keys())

    fig = make_subplots(
        rows=n_datasets, cols=2,
        subplot_titles=[
            t
            for nd in nomes_datasets
            for t in [
                f"{nd} — Clusters ({resultados_por_dataset[nd]['melhor_mod']}, "
                f"k={resultados_por_dataset[nd]['melhor_n']})",
                f"{nd} — Label Real"
            ]
        ],
        vertical_spacing=0.06
    )

    cores_cluster = px.colors.qualitative.Plotly
    cores_real    = {'0': '#4878CF', '1': '#D65F5F'}
    nomes_real    = {'0': 'Falso Positivo', '1': 'Contaminação Real'}
    simbolos      = {'0': 'circle', '1': 'diamond'}

    for row_idx, nome_dataset in enumerate(nomes_datasets, start=1):
        res      = resultados_por_dataset[nome_dataset]
        df_feat  = res['df_features']
        X_scaled = res['X_scaled']

        pca     = PCA(n_components=2, random_state=RANDOM_STATE)
        X_pca   = pca.fit_transform(X_scaled)
        var_pc1 = pca.explained_variance_ratio_[0] * 100
        var_pc2 = pca.explained_variance_ratio_[1] * 100

        df_plot = pd.DataFrame({
            'PC1':     X_pca[:, 0],
            'PC2':     X_pca[:, 1],
            'Cluster': df_feat['Cluster'].astype(str),
            'Real':    df_feat['Real'].astype(str),
        })

        # Subplot esquerdo — clusters
        for cluster_id in sorted(df_plot['Cluster'].unique()):
            mask = df_plot['Cluster'] == cluster_id
            fig.add_trace(go.Scatter(
                x=df_plot.loc[mask, 'PC1'], y=df_plot.loc[mask, 'PC2'],
                mode='markers',
                name=f'Cluster {cluster_id}',
                marker=dict(
                    size=8,
                    color=cores_cluster[int(cluster_id) % len(cores_cluster)],
                    opacity=0.8, line=dict(width=0.5, color='white')
                ),
                legendgroup=f'cluster_{cluster_id}',
                showlegend=(row_idx == 1)
            ), row=row_idx, col=1)

        # Subplot direito — label real
        for real_id in ['0', '1']:
            mask = df_plot['Real'] == real_id
            fig.add_trace(go.Scatter(
                x=df_plot.loc[mask, 'PC1'], y=df_plot.loc[mask, 'PC2'],
                mode='markers',
                name=nomes_real[real_id],
                marker=dict(
                    size=8, color=cores_real[real_id],
                    symbol=simbolos[real_id], opacity=0.85,
                    line=dict(width=0.5, color='white')
                ),
                legendgroup=f'real_{real_id}',
                showlegend=(row_idx == 1)
            ), row=row_idx, col=2)

        for col in [1, 2]:
            fig.update_xaxes(title_text=f'PC1 ({var_pc1:.1f}%)', row=row_idx, col=col)
            fig.update_yaxes(title_text=f'PC2 ({var_pc2:.1f}%)', row=row_idx, col=col)

    fig.update_layout(
        title=f'PCA 2D — {titulo}',
        template='plotly_white',
        height=500 * n_datasets,
        legend=dict(groupclick='toggleitem')
    )
    return fig

# =============================================================================
# Classificação (supervisionada)
# =============================================================================

def extrair_features(crystallizer, df_medicoes, df_eventos, janelas, stats_funcs, map_medicoes_unif=None):
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
                        k_features=10, best_thr=0.4, plotar=True):
    """Compara as quatro abordagens (base / SMOTE / feature selection / ambos).

    Retorna (df_relatorio, resultados_para_plot).
    """
    if stats_funcs is None:
        stats_funcs = STATS_FUNCS

    print(f"\nIniciando testes comparativos no {modo} usando tratamento {tratamento_alvo}\n")

    df_feat, cols_feat_brutas = extrair_features(modo, df_med, df_ev, janelas, stats_funcs)

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
                                       incluir_crystallizer=False, plotar=True):
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
        print(f"{modo} | {tratamento} (Absolutas + Relativas Misturadas)")
        print(f"{'='*55}")

        # map_medicoes_unif=None força a mistura temporal entre os reatores
        df_feat, cols_feat = extrair_features(
            crystallizer=modo,
            df_medicoes=df_med,
            df_eventos=df_eventos,
            janelas=janelas,
            stats_funcs=stats_funcs,
            map_medicoes_unif=None
        )

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
# IsolationForest
# =============================================================================

def otimizar_avaliar_iforest(X_train, y_train, X_test, y_test, param_grid):
    """
    Testa várias combinações de hiperparâmetros usando as anomalias do CONJUNTO DE TREINO 
    para escolher o melhor modelo, mantendo o conjunto de teste fora.
    """
    X_train_normal = X_train[y_train == 0]
    
    melhor_score = -1
    melhores_params = {}
    melhor_pipe = None
    
    # Gera todas as combinações possíveis
    chaves = param_grid.keys()
    combinacoes = list(itertools.product(*param_grid.values())) # Transformado em lista por segurança
    
    for config in combinacoes:
        params = dict(zip(chaves, config))
        
        pipe = SklearnPipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('iforest', IsolationForest(
                n_estimators=params['n_estimators'], 
                max_samples=params['max_samples'],
                max_features=params['max_features'],
                contamination=params['contamination'], 
                random_state=RANDOM_STATE, 
                n_jobs=-1
            ))
        ])
        
        # Treina estritamente nos dados normais do passado
        pipe.fit(X_train_normal)
        
        # AVALIAÇÃO NO TREINO: O modelo tenta prever o próprio treino (que contém anomalias que ele não viu no fit)
        preds_train = pipe.predict(X_train)
        y_pred_train = np.where(preds_train == -1, 1, 0)
        
        # O score que define o melhor modelo é o do treino
        score_train = fbeta_score(y_train, y_pred_train, beta=2, pos_label=1, zero_division=0)
        
        if score_train > melhor_score:
            melhor_score = score_train
            melhores_params = params
            melhor_pipe = pipe
            
    # APÓS escolher o melhor modelo é feita uma ÚNICA predição no conjunto de Teste
    preds_test = melhor_pipe.predict(X_test)
    melhor_y_pred_test = np.where(preds_test == -1, 1, 0)
            
    return melhor_pipe, melhor_y_pred_test, melhores_params, melhor_score


def plotar_resultados_iforest(y_test, y_pred, tratamento_nome):
    """Gera apenas a Matriz de Confusão."""
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Prev: Normal', 'Prev: Anomalia'],
        y=['Real: Normal', 'Real: Contaminação'],
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        showscale=False
    ))

    fig.update_layout(
        title=f"Matriz de Confusão — Isolation Forest — {tratamento_nome}",
        template="plotly_white",
        height=400,
        width=500
    )
    
    fig.show()


def rodar_iforest(modo, configs, janelas, param_grid, stats_funcs=None, plotar=True):
    """Roda o IsolationForest otimizado em todos os tratamentos.

    configs : {nome_tratamento: (df_medicoes, df_eventos)}
    Retorna {nome_tratamento: dict com o pipe, as predições e os melhores parâmetros}.
    """
    if stats_funcs is None:
        stats_funcs = STATS_FUNCS

    resultados = {}

    for tratamento, (df_med, df_ev) in configs.items():
        print(f"\n{'='*55}")
        print(f"{modo} | {tratamento} (Isolation Forest - Otimizado)")
        print(f"{'='*55}")

        df_feat, cols_feat = extrair_features(modo, df_med, df_ev, janelas, stats_funcs)

        # Feature de salto abrupto (variação percentual)
        if 'max_1d' in df_feat.columns and 'media_6d' in df_feat.columns:
            df_feat['salto_abrupto'] = (df_feat['max_1d'] - df_feat['media_6d']) / (df_feat['media_6d'] + 0.001)
            cols_feat.append('salto_abrupto')

        # Divisão temporal
        X_train, X_test, y_train, y_test, _, _ = dividir_dados(
            df_feat, cols_feat, incluir_crystallizer=False
        )

        print(f"  Treino : {X_train.shape[0]} amostras (usando apenas as normais)")
        print(f"  Teste  : {X_test.shape[0]} amostras  (Real=1: {y_test.sum()}  Real=0: {(y_test==0).sum()})")

        pipe_if, y_pred_if, melhores_params, melhor_score = otimizar_avaliar_iforest(
            X_train, y_train, X_test, y_test, param_grid=param_grid
        )

        print(f"\n  Melhores Parâmetros Encontrados:")
        for k, v in melhores_params.items():
            print(f"  - {k}: {v}")
        print(f"  - F2-Score: {melhor_score:.4f}")

        print(f"\n  Relatório no Teste:")
        print(classification_report(
            y_test, y_pred_if,
            labels=[0, 1],
            target_names=['Operação Normal', 'Contaminação Detectada'],
            zero_division=0
        ))

        if plotar:
            plotar_resultados_iforest(y_test, y_pred_if, tratamento)

        resultados[tratamento] = {
            'pipe': pipe_if, 'y_test': y_test, 'y_pred': y_pred_if,
            'params': melhores_params, 'f2': melhor_score,
        }

    return resultados

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
    fig.show()


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
    fig.show()
