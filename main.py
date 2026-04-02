import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.base import clone
from scipy.stats import uniform, randint
import warnings

warnings.filterwarnings('ignore')

# Configuração da Página
st.set_page_config(page_title="Análise De Dados no Futebol", layout="wide", page_icon="📈")

# =====================================================================
# CSS CUSTOMIZADO
# =====================================================================
st.markdown("""
<style>
    .stApp, .main {
        background-color: #0d1117 !important;
        color: #ffffff !important;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #0d1117 !important;
        border-right: 1px solid #30363d !important;
    }
    section[data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }

    div[data-baseweb="select"] > div, 
    div[data-baseweb="base-input"] > input {
        background-color: #161b22 !important;
        color: #ffffff !important;
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
    }
    
    /* Ajuste para os textos dos inputs (Número de apostas, Quantidade de usuários, etc) */
    label[data-testid="stWidgetLabel"] p {
        color: #ffffff !important;
        font-size: 16px !important;
        font-weight: 600 !important;
    }

    /* Ajuste para os botões (como o Iniciar Simulação) */
    div[data-testid="stButton"] > button {
        background-color: #3498db !important;
        color: #ffffff !important;
        border: 1px solid #2980b9 !important;
        font-weight: bold !important;
    }
    div[data-testid="stButton"] > button p {
        color: #ffffff !important;
        font-size: 16px !important;
    }
    div[data-testid="stButton"] > button:hover {
        background-color: #2980b9 !important;
        border-color: #ffffff !important;
    }

    span[data-baseweb="tag"] {
        background-color: #3498db !important;
        color: #ffffff !important;
        border: none !important;
    }
    div[data-baseweb="popover"] > div, 
    div[data-baseweb="popover"] ul,
    ul[role="listbox"],
    div[role="listbox"] {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
    }
    li[role="option"] {
        background-color: #161b22 !important;
        color: #ffffff !important;
    }
    li[role="option"]:hover, li[aria-selected="true"] {
        background-color: #21262d !important;
        color: #3498db !important;
    }

    [data-testid="stMetric"] {
        background-color: #161b22 !important;
        border: 1px solid #21262d !important;
        border-left: 4px solid #3498db !important;
        padding: 15px 20px !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.4) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(52, 152, 219, 0.25) !important;
        border-left: 4px solid #00ff88 !important;
    }
    [data-testid="stMetricLabel"] * {
        color: #8b949e !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 26px !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #161b22;
        border-radius: 6px 6px 0px 0px;
        padding: 10px 20px;
        border: 1px solid #21262d;
        border-bottom: none;
        font-weight: 600;
        color: #8b949e;
    }
    .stTabs [aria-selected="true"] {
        background-color: #21262d !important;
        border-top: 3px solid #3498db !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

plt.style.use('dark_background')
plt.rcParams.update({
    "figure.facecolor": "none",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "savefig.facecolor": "#0d1117",
    "text.color": "#ffffff",
    "axes.labelcolor": "#8b949e",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d"
})

col_logo, col_title = st.columns([1, 15])
with col_logo:
    st.markdown("<h1 style='text-align: left;'>⚡</h1>", unsafe_allow_html=True)
with col_title:
    st.title("Sports AI & Predictive Analytics")
    st.markdown("Painel de inteligência quantitativa para análise tática, estatística e de valor esperado (EV).")

# =====================================================================
# FUNÇÕES GERAIS E CACHE
# =====================================================================

@st.cache_data
def carregar_dados():
    try:
        df_br = pd.read_csv("Campeonato_Brasileiro_Com_Clima.csv", encoding="utf-8")
        df_pl = pd.read_csv("Campeonato_Premier_League_Com_Clima.csv", encoding="utf-8")
    except FileNotFoundError as e:
        st.error(f"ERRO AO CARREGAR ARQUIVOS: {e}")
        st.stop()

    df_br["Liga"] = "Brasileirão Série A"
    df_pl["Liga"] = "Premier League"
    df = pd.concat([df_br, df_pl], ignore_index=True)

    colunas_numericas = [
        "Gols_Mandante", "Gols_Visitante", "Odd_Mandante", "Odd_Empate", "Odd_Visitante",
        "Finalizacoes_Mandante", "Finalizacoes_Visitante", "No_Alvo_Mandante", "No_Alvo_Visitante",
        "Escanteios_Mandante", "Escanteios_Visitante", "Amarelos_Mandante", "Amarelos_Visitante", 
        "Faltas_Mandante", "Faltas_Visitante", "Defesas_Goleiro_Mandante", "Defesas_Goleiro_Visitante",
        "Temperatura_C", "Umidade_Relativa_%", "Velocidade_Vento_kmh"
    ]
    for col in colunas_numericas:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '.')
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Publico" in df.columns:
        df["Publico"] = df["Publico"].astype(str).str.replace(r"[^\d]", "", regex=True)
        df["Publico"] = pd.to_numeric(df["Publico"], errors="coerce")
    
    if "Posse_Mandante" in df.columns:
        df["Posse_Mandante"] = pd.to_numeric(df["Posse_Mandante"].astype(str).str.replace("%", ""), errors="coerce")
        df["Posse_Visitante"] = pd.to_numeric(df["Posse_Visitante"].astype(str).str.replace("%", ""), errors="coerce")

    def classificar_resultado(row):
        gm, gv = row["Gols_Mandante"], row["Gols_Visitante"]
        if pd.isna(gm) or pd.isna(gv): return np.nan
        if gm > gv: return "Mandante"
        elif gm == gv: return "Empate"
        else: return "Visitante"

    df["Resultado"] = df.apply(classificar_resultado, axis=1)
    df["Temporada"] = df["Temporada"].astype(str).str.extract(r'(\d{4})')[0]
    
    df["Gols_Total"] = df["Gols_Mandante"] + df["Gols_Visitante"]
    
    for prefixo, mand, vis in [
        ("Escanteios", "Escanteios_Mandante", "Escanteios_Visitante"),
        ("Amarelos", "Amarelos_Mandante", "Amarelos_Visitante"),
        ("No_Alvo", "No_Alvo_Mandante", "No_Alvo_Visitante"),
        ("Defesas", "Defesas_Goleiro_Mandante", "Defesas_Goleiro_Visitante")
    ]:
        if mand in df.columns and vis in df.columns:
            df[f"{prefixo}_Total"] = df[mand] + df[vis]
        else:
            df[f"{prefixo}_Total"] = np.nan
    
    df["Over_1_5"] = (df["Gols_Total"] > 1).astype(int)
    df["Over_2_5"] = (df["Gols_Total"] > 2).astype(int)
    df["Ambas_Marcam"] = ((df["Gols_Mandante"] > 0) & (df["Gols_Visitante"] > 0)).astype(int)
    df["Vit_Mandante"] = (df["Resultado"] == "Mandante").astype(int)
    df["Empate_Res"] = (df["Resultado"] == "Empate").astype(int)
    df["Vit_Visitante"] = (df["Resultado"] == "Visitante").astype(int)
    
    if 'Data_Hora' in df.columns:
        df['Data_Hora'] = pd.to_datetime(df['Data_Hora'], dayfirst=True, errors='coerce')
        df = df.sort_values(by=['Liga', 'Data_Hora']).reset_index(drop=True)
        df['Mes'] = df['Data_Hora'].dt.month.fillna(0).astype(int)

    return df

@st.cache_resource(show_spinner=False)
def treinar_e_avaliar_modelo(df_base, liga_nome):
    df_liga = df_base[df_base['Liga'] == liga_nome].copy()
    
    janelas = [3, 5]
    df_features = df_liga.copy()

    df_m = df_liga[['Data_Hora', 'Mandante', 'Gols_Mandante', 'Gols_Visitante', 'Finalizacoes_Mandante', 'Finalizacoes_Visitante']].copy()
    df_m.columns = ['Data_Hora', 'Equipe', 'Gols_Feitos', 'Gols_Sofridos', 'Fin_Feitas', 'Fin_Sofridas']
    df_v = df_liga[['Data_Hora', 'Visitante', 'Gols_Visitante', 'Gols_Mandante', 'Finalizacoes_Visitante', 'Finalizacoes_Mandante']].copy()
    df_v.columns = ['Data_Hora', 'Equipe', 'Gols_Feitos', 'Gols_Sofridos', 'Fin_Feitas', 'Fin_Sofridas']

    df_times = pd.concat([df_m, df_v]).sort_values(by=['Equipe', 'Data_Hora']).reset_index(drop=True)

    cols_calc = ['Gols_Feitos', 'Gols_Sofridos', 'Fin_Feitas', 'Fin_Sofridas']
    for j in janelas:
        for col in cols_calc:
            df_times[f'Media{j}_{col}'] = df_times.groupby('Equipe')[col].transform(lambda x: x.shift(1).rolling(window=j, min_periods=1).mean())

    cols_merge = ['Data_Hora', 'Equipe'] + [c for c in df_times.columns if 'Media' in c]
    df_times_stats = df_times[cols_merge]

    df_features = df_features.merge(df_times_stats, left_on=['Data_Hora', 'Mandante'], right_on=['Data_Hora', 'Equipe'], how='left').drop(columns=['Equipe'])
    df_features.rename(columns={c: f'Mandante_{c}' for c in df_times_stats.columns if 'Media' in c}, inplace=True)
    
    df_features = df_features.merge(df_times_stats, left_on=['Data_Hora', 'Visitante'], right_on=['Data_Hora', 'Equipe'], how='left').drop(columns=['Equipe'])
    df_features.rename(columns={c: f'Visitante_{c}' for c in df_times_stats.columns if 'Media' in c}, inplace=True)

    for j in janelas:
        df_features[f'Diff{j}_Ataque'] = df_features[f'Mandante_Media{j}_Gols_Feitos'] - df_features[f'Visitante_Media{j}_Gols_Feitos']
        df_features[f'Diff{j}_Defesa'] = df_features[f'Mandante_Media{j}_Gols_Sofridos'] - df_features[f'Visitante_Media{j}_Gols_Sofridos']
        df_features[f'Diff{j}_Volume'] = df_features[f'Mandante_Media{j}_Fin_Feitas'] - df_features[f'Visitante_Media{j}_Fin_Feitas']

    df_features['Alvo_Vitoria'] = (df_features['Gols_Mandante'] > df_features['Gols_Visitante']).astype(int)

    # =========================================================
    # SELEÇÃO DINÂMICA DO ARQUIVO PKL (E ASSEGURAR CARREGAMENTO)
    # =========================================================
    if liga_nome == "Premier League":
        arquivo_pkl = "melhor_modelo_pipeline_premier.pkl"
    else:
        arquivo_pkl = "melhor_modelo_pipeline.pkl"

    if not os.path.exists(arquivo_pkl):
        st.error(f"O arquivo {arquivo_pkl} não foi encontrado para a liga {liga_nome}.")
        st.stop()
        
    dado_carregado = joblib.load(arquivo_pkl)
    pipeline_melhor = dado_carregado['pipeline']
    melhor_limiar = dado_carregado['LIMIAR_VALOR']
    colunas_treino = dado_carregado['colunas_treino']
    nome_melhor = dado_carregado['nome_modelo']
    best_params = dado_carregado['best_params']

    cols_essenciais = colunas_treino + ['Odd_Mandante', 'Alvo_Vitoria', 'Mandante', 'Visitante', 'Data_Hora']

    # REPLICAÇÃO EXATA DO CORTE DE DADOS DO CÓDIGO PADRÃO:
    df_modelo = df_features.dropna(subset=cols_essenciais).copy()
    df_modelo = df_modelo[df_modelo['Odd_Mandante'] > 1.01]
    df_modelo = df_modelo.sort_values(by='Data_Hora').reset_index(drop=True)

    n = len(df_modelo)
    idx_treino = int(n * 0.60)
    idx_val = int(n * 0.80)

    X = df_modelo[colunas_treino]
    y = df_modelo['Alvo_Vitoria']

    X_treino, y_treino = X.iloc[:idx_treino], y.iloc[:idx_treino]
    X_val, y_val = X.iloc[idx_treino:idx_val], y.iloc[idx_treino:idx_val]
    X_teste, y_teste = X.iloc[idx_val:], y.iloc[idx_val:]

    # =========================================================
    # APLICAÇÃO DO WALK-FORWARD NO TESTE (EXATAMENTE COMO O BASE)
    # =========================================================
    X_hist_atual = pd.concat([X_treino, X_val]).copy()
    y_hist_atual = pd.concat([y_treino, y_val]).copy()

    df_teste_wf = df_modelo.iloc[idx_val:].copy()
    df_teste_wf['Data_Apenas'] = df_teste_wf['Data_Hora'].dt.date
    datas_teste = df_teste_wf['Data_Apenas'].unique()

    modelo_producao = clone(pipeline_melhor)
    probs_teste_wf = []

    for data_atual in datas_teste:
        jogos_dia_idx = df_teste_wf[df_teste_wf['Data_Apenas'] == data_atual].index
        X_dia = X_teste.loc[jogos_dia_idx]
        
        modelo_producao.fit(X_hist_atual, y_hist_atual)
        probs_dia = modelo_producao.predict_proba(X_dia)[:, 1]
        probs_teste_wf.extend(probs_dia)
        
        X_hist_atual = pd.concat([X_hist_atual, X_dia], ignore_index=True)
        y_hist_atual = pd.concat([y_hist_atual, y_teste.loc[jogos_dia_idx]], ignore_index=True)

    probs_teste = np.array(probs_teste_wf)

    # DataFrame de avaliação do teste
    df_teste = df_modelo.iloc[idx_val:].copy()
    df_teste['Prob_Modelo'] = probs_teste
    df_teste['Odd_Justa_Modelo'] = 1 / df_teste['Prob_Modelo']
    df_teste['EV_Porcentagem'] = (df_teste['Odd_Mandante'] / df_teste['Odd_Justa_Modelo']) - 1
    
    df_apostas = df_teste[df_teste['EV_Porcentagem'] > melhor_limiar].copy()
    df_apostas['Resultado_Aposta'] = np.where(df_apostas['Alvo_Vitoria'] == 1, df_apostas['Odd_Mandante'] - 1, -1)
    df_apostas['Lucro_Acumulado'] = df_apostas['Resultado_Aposta'].cumsum()

    # =========================================================
    # JOGOS FUTUROS (SEM RESULTADO)
    # =========================================================
    df_futuro = df_features[df_features['Gols_Mandante'].isna()].copy()
    df_futuro_final = pd.DataFrame()
    if not df_futuro.empty:
        df_futuro = df_futuro.dropna(subset=colunas_treino)
        if not df_futuro.empty:
            modelo_producao.fit(X_hist_atual, y_hist_atual) # Treina com todo o histórico acumulado
            probs_fut = modelo_producao.predict_proba(df_futuro[colunas_treino])[:, 1]
            df_futuro['Prob_Modelo_Num'] = probs_fut
            df_futuro['Odd_Justa'] = 1 / df_futuro['Prob_Modelo_Num']
            df_futuro['EV (%)'] = ((df_futuro['Odd_Mandante'] / df_futuro['Odd_Justa']) - 1) * 100
            df_futuro_final = df_futuro

    return nome_melhor, melhor_limiar, df_apostas, df_futuro_final, best_params, pipeline_melhor, colunas_treino

df_geral = carregar_dados()

# ---------------------------------------------------------------------
# BARRA LATERAL (SIDEBAR) & CONTROLE DE ACESSIBILIDADE
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Configurações Gerais")
    modo_daltonico = st.toggle("👁️ Modo de Acessibilidade (Cores Seguras)", value=False)
    
    if modo_daltonico:
        COR_POS = "#0072B2" 
        COR_NEG = "#D55E00" 
        COR_MAN = "#56B4E9" 
        COR_VIS = "#E69F00" 
        COR_EMP = "#F0E442" 
        TXT_ENTRAR = "SIM"
        TXT_NAO_ENTRAR = "NÃO"
    else:
        COR_POS = "#00ff88" 
        COR_NEG = "#ff4d4d" 
        COR_MAN = "#3498db" 
        COR_VIS = "#e74c3c" 
        COR_EMP = "#8b949e" 
        TXT_ENTRAR = "SIM"
        TXT_NAO_ENTRAR = "NÃO"

    st.markdown("### 📅 Filtros de Tempo e Ligas")
    
    anos_disponiveis = sorted(df_geral["Temporada"].dropna().unique(), reverse=True)
    anos_sel = st.multiselect("Anos:", anos_disponiveis, default=anos_disponiveis)
    
    meses_disponiveis = sorted([m for m in df_geral["Mes"].unique() if m != 0])
    nomes_meses = {1:"Jan", 2:"Fev", 3:"Mar", 4:"Abr", 5:"Mai", 6:"Jun", 7:"Jul", 8:"Ago", 9:"Set", 10:"Out", 11:"Nov", 12:"Dez"}
    meses_formatados = [f"{m} - {nomes_meses[m]}" for m in meses_disponiveis]
    meses_sel_str = st.multiselect("Meses:", meses_formatados, default=meses_formatados)
    meses_sel = [int(m.split(" - ")[0]) for m in meses_sel_str]

    st.markdown("### 📊 Filtro de Odds e Probabilidades")
    st.markdown("<small>Dica: Você pode ativar e cruzar múltiplos filtros ao mesmo tempo.</small>", unsafe_allow_html=True)
    
    modo_entrada_odd = st.radio("Método de Entrada:", ["🎚️ Usar Sliders", "⌨️ Digitar Manualmente"], horizontal=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    filtro_mandante_ativo = st.checkbox("✅ Filtrar Odd Casa (Mandante)", value=True)
    odd_m_min, odd_m_max = 1.01, 25.0
    if filtro_mandante_ativo:
        if modo_entrada_odd == "🎚️ Usar Sliders":
            odd_m_min, odd_m_max = st.slider("Range Casa:", min_value=1.01, max_value=25.0, value=(1.5, 3.0), step=0.05, key="sl_m")
        else:
            col1, col2 = st.columns(2)
            odd_m_min = col1.number_input("Min Casa:", min_value=1.01, max_value=25.0, value=1.50, step=0.05, key="num_m_min")
            odd_m_max = col2.number_input("Max Casa:", min_value=1.01, max_value=25.0, value=3.00, step=0.05, key="num_m_max")
        
        prob_m_max = (1 / odd_m_min) * 100
        prob_m_min = (1 / odd_m_max) * 100
        st.markdown(f"<div style='text-align: right; color: {COR_MAN}; font-size: 0.85em;'>Probabilidade Implícita: <b>{prob_m_min:.1f}% a {prob_m_max:.1f}%</b></div>", unsafe_allow_html=True)

    filtro_empate_ativo = st.checkbox("✅ Filtrar Odd Empate", value=False)
    odd_e_min, odd_e_max = 1.01, 25.0
    if filtro_empate_ativo:
        if modo_entrada_odd == "🎚️ Usar Sliders":
            odd_e_min, odd_e_max = st.slider("Range Empate:", min_value=1.01, max_value=25.0, value=(3.0, 4.0), step=0.05, key="sl_e")
        else:
            col1, col2 = st.columns(2)
            odd_e_min = col1.number_input("Min Empate:", min_value=1.01, max_value=25.0, value=3.00, step=0.05, key="num_e_min")
            odd_e_max = col2.number_input("Max Empate:", min_value=1.01, max_value=25.0, value=4.00, step=0.05, key="num_e_max")
        
        prob_e_max = (1 / odd_e_min) * 100
        prob_e_min = (1 / odd_e_max) * 100
        st.markdown(f"<div style='text-align: right; color: {COR_EMP}; font-size: 0.85em;'>Probabilidade Implícita: <b>{prob_e_min:.1f}% a {prob_e_max:.1f}%</b></div>", unsafe_allow_html=True)

    filtro_visitante_ativo = st.checkbox("✅ Filtrar Odd Fora (Visitante)", value=False)
    odd_v_min, odd_v_max = 1.01, 25.0
    if filtro_visitante_ativo:
        if modo_entrada_odd == "🎚️ Usar Sliders":
            odd_v_min, odd_v_max = st.slider("Range Fora:", min_value=1.01, max_value=25.0, value=(2.0, 5.0), step=0.05, key="sl_v")
        else:
            col1, col2 = st.columns(2)
            odd_v_min = col1.number_input("Min Fora:", min_value=1.01, max_value=25.0, value=2.00, step=0.05, key="num_v_min")
            odd_v_max = col2.number_input("Max Fora:", min_value=1.01, max_value=25.0, value=5.00, step=0.05, key="num_v_max")
        
        prob_v_max = (1 / odd_v_min) * 100
        prob_v_min = (1 / odd_v_max) * 100
        st.markdown(f"<div style='text-align: right; color: {COR_VIS}; font-size: 0.85em;'>Probabilidade Implícita: <b>{prob_v_min:.1f}% a {prob_v_max:.1f}%</b></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# ABAS PRINCIPAIS
# ---------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Estatísticas e Tática", "🤖 Inteligência (Brasileirão)", "🤖 Inteligência (Premier League)", "🎲 Simulador de Apostas", "⚖️ Eficiência das Odds"])

# ---------------------------------------------------------------------
# TAB 1: ESTATÍSTICAS E GRÁFICOS
# ---------------------------------------------------------------------
with tab1:
    with st.expander("🎯 **Refinar Competição e Equipes (Clique para abrir)**", expanded=False):
        col_f1, col_f2 = st.columns([1, 2])
        with col_f1:
            ligas = df_geral["Liga"].unique()
            ligas_sel = st.multiselect("Competições:", ligas, default=ligas)
        
        df_liga_filter = df_geral[
            (df_geral["Liga"].isin(ligas_sel)) & 
            (df_geral["Temporada"].isin(anos_sel)) &
            (df_geral["Mes"].isin(meses_sel))
        ].copy()
        
        if filtro_mandante_ativo:
            df_liga_filter = df_liga_filter[(df_liga_filter["Odd_Mandante"] >= odd_m_min) & (df_liga_filter["Odd_Mandante"] <= odd_m_max)]
        if filtro_empate_ativo:
            df_liga_filter = df_liga_filter[(df_liga_filter["Odd_Empate"] >= odd_e_min) & (df_liga_filter["Odd_Empate"] <= odd_e_max)]
        if filtro_visitante_ativo:
            df_liga_filter = df_liga_filter[(df_liga_filter["Odd_Visitante"] >= odd_v_min) & (df_liga_filter["Odd_Visitante"] <= odd_v_max)]
        
        times_disponiveis = sorted(list(set(df_liga_filter["Mandante"].unique()) | set(df_liga_filter["Visitante"].unique())))
        
        with col_f2:
            selecionar_todos = st.toggle("Selecionar todas as equipas do filtro", value=True)
            if selecionar_todos:
                times_sel = times_disponiveis
            else:
                times_sel = st.multiselect("Equipas:", times_disponiveis, default=[times_disponiveis[0]] if times_disponiveis else [])
                
    df_subset = df_liga_filter[(df_liga_filter["Mandante"].isin(times_sel)) | (df_liga_filter["Visitante"].isin(times_sel))].copy()

    if df_subset.empty:
        st.warning("Nenhum dado encontrado com a combinação de filtros atuais.")
    else:
        total_jogos = len(df_subset)
        media_gols = df_subset["Gols_Total"].mean()
        btts_pct = df_subset["Ambas_Marcam"].mean() * 100
        vit_mandante_pct = df_subset["Vit_Mandante"].mean() * 100
        
        media_publico = df_subset["Publico"].mean()
        media_escanteios = df_subset["Escanteios_Total"].mean()
        media_amarelos = df_subset["Amarelos_Total"].mean()
        media_no_alvo = df_subset["No_Alvo_Total"].mean()
        media_defesas = df_subset["Defesas_Total"].mean()
        
        odd_m_media = df_subset["Odd_Mandante"].mean()
        odd_e_media = df_subset["Odd_Empate"].mean()
        odd_v_media = df_subset["Odd_Visitante"].mean()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric(label="🏟️ Partidas Filtradas", value=f"{total_jogos}")
        k2.metric(label="⚽ Gols por Jogo (Média)", value=f"{media_gols:.2f}")
        pub_val = f"{media_publico:,.0f}".replace(',', '.') if pd.notna(media_publico) else "N/A"
        k3.metric(label="👥 Público Médio", value=pub_val)
        k4.metric(label="🏠 Win Rate Mandante", value=f"{vit_mandante_pct:.1f}%")
        
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        
        k5, k6, k7, k8 = st.columns(4)
        val_esc = f"{media_escanteios:.1f}" if pd.notna(media_escanteios) else "N/A"
        val_ama = f"{media_amarelos:.1f}" if pd.notna(media_amarelos) else "N/A"
        val_alv = f"{media_no_alvo:.1f}" if pd.notna(media_no_alvo) else "N/A"
        val_def = f"{media_defesas:.1f}" if pd.notna(media_defesas) else "N/A"
        
        k5.metric(label="🚩 Escanteios / Jogo", value=val_esc)
        k6.metric(label="🟨 Cartões Amarelos / Jogo", value=val_ama)
        k7.metric(label="🎯 Finalizações no Alvo / Jogo", value=val_alv)
        k8.metric(label="🧤 Defesas Goleiro / Jogo", value=val_def)
        
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        
        k9, k10, k11, k12 = st.columns(4)
        val_om = f"{odd_m_media:.2f}" if pd.notna(odd_m_media) else "N/A"
        val_oe = f"{odd_e_media:.2f}" if pd.notna(odd_e_media) else "N/A"
        val_ov = f"{odd_v_media:.2f}" if pd.notna(odd_v_media) else "N/A"
        
        k9.metric(label="📊 Odd Mandante (Média)", value=val_om)
        k10.metric(label="📊 Odd Empate (Média)", value=val_oe)
        k11.metric(label="📊 Odd Visitante (Média)", value=val_ov)
        k12.metric(label="🔥 Ambas Marcam (BTTS)", value=f"{btts_pct:.1f}%")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        colA, colB = st.columns([2, 2])
        
        with colA:
            st.markdown("#### ⚖️ Comparativo de Força (Casa vs Fora)")
            
            metricas = ['Posse de Bola (%)', 'Finalizações', 'Finalizações no Alvo', 'Escanteios', 'Faltas']
            
            max_posse = max(df_subset['Posse_Mandante'].max() if 'Posse_Mandante' in df_subset else 100,
                            df_subset['Posse_Visitante'].max() if 'Posse_Visitante' in df_subset else 100)
            max_fin = max(df_subset['Finalizacoes_Mandante'].max(), df_subset['Finalizacoes_Visitante'].max())
            max_alvo = max(df_subset['No_Alvo_Mandante'].max(), df_subset['No_Alvo_Visitante'].max())
            max_esc = max(df_subset['Escanteios_Mandante'].max(), df_subset['Escanteios_Visitante'].max())
            max_fal = max(df_subset['Faltas_Mandante'].max(), df_subset['Faltas_Visitante'].max())
            
            tetos = [
                max_posse if pd.notna(max_posse) and max_posse > 0 else 100,
                max_fin if pd.notna(max_fin) and max_fin > 0 else 30,
                max_alvo if pd.notna(max_alvo) and max_alvo > 0 else 15,
                max_esc if pd.notna(max_esc) and max_esc > 0 else 15,
                max_fal if pd.notna(max_fal) and max_fal > 0 else 30
            ]

            med_man = [
                df_subset['Posse_Mandante'].mean() if 'Posse_Mandante' in df_subset else 0, 
                df_subset['Finalizacoes_Mandante'].mean(), 
                df_subset['No_Alvo_Mandante'].mean() if 'No_Alvo_Mandante' in df_subset else 0, 
                df_subset['Escanteios_Mandante'].mean() if 'Escanteios_Mandante' in df_subset else 0, 
                df_subset['Faltas_Mandante'].mean() if 'Faltas_Mandante' in df_subset else 0
            ]
            med_vis = [
                df_subset['Posse_Visitante'].mean() if 'Posse_Visitante' in df_subset else 0, 
                df_subset['Finalizacoes_Visitante'].mean(), 
                df_subset['No_Alvo_Visitante'].mean() if 'No_Alvo_Visitante' in df_subset else 0, 
                df_subset['Escanteios_Visitante'].mean() if 'Escanteios_Visitante' in df_subset else 0, 
                df_subset['Faltas_Visitante'].mean() if 'Faltas_Visitante' in df_subset else 0
            ]
            
            med_man = [x if pd.notna(x) else 0 for x in med_man]
            med_vis = [x if pd.notna(x) else 0 for x in med_vis]

            norm_man = [m / t if t > 0 else 0 for m, t in zip(med_man, tetos)]
            norm_vis = [v / t if t > 0 else 0 for v, t in zip(med_vis, tetos)]

            hover_man = [f"{m}: {v:.1f}" for m, v in zip(metricas, med_man)]
            hover_vis = [f"{m}: {v:.1f}" for m, v in zip(metricas, med_vis)]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=norm_man, theta=metricas, fill='toself', name='Casa (Mandante)',
                line_color=COR_MAN, fillcolor=f"rgba({int(COR_MAN[1:3],16)}, {int(COR_MAN[3:5],16)}, {int(COR_MAN[5:7],16)}, 0.4)",
                hoverinfo="text", text=hover_man
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=norm_vis, theta=metricas, fill='toself', name='Fora (Visitante)',
                line_color=COR_VIS, fillcolor=f"rgba({int(COR_VIS[1:3],16)}, {int(COR_VIS[3:5],16)}, {int(COR_VIS[5:7],16)}, 0.4)",
                hoverinfo="text", text=hover_vis
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, showticklabels=False, range=[0, 1], gridcolor='#30363d'), 
                    bgcolor='#0d1117'
                ),
                template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                margin=dict(l=45, r=45, t=30, b=30), showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with colB:
            st.markdown("#### ⚽ Evolução de Gols por Temporada")
            gols_ano = df_subset.groupby("Temporada")[["Gols_Mandante", "Gols_Visitante", "Gols_Total"]].mean().reset_index()
            fig_gols = go.Figure()
            fig_gols.add_trace(go.Scatter(x=gols_ano["Temporada"], y=gols_ano["Gols_Total"], mode='lines+markers', name='Total', line=dict(color=COR_POS, width=3)))
            fig_gols.add_trace(go.Scatter(x=gols_ano["Temporada"], y=gols_ano["Gols_Mandante"], mode='lines+markers', name='Mandante', line=dict(color=COR_MAN, width=2)))
            fig_gols.add_trace(go.Scatter(x=gols_ano["Temporada"], y=gols_ano["Gols_Visitante"], mode='lines+markers', name='Visitante', line=dict(color=COR_VIS, width=2)))
            fig_gols.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified")
            st.plotly_chart(fig_gols, use_container_width=True)

        colC, colD = st.columns(2)
        with colC:
            st.markdown("#### 📊 Distribuição de Resultados")
            res_counts = df_subset["Resultado"].value_counts().reset_index()
            res_counts.columns = ['Resultado', 'Contagem']
            color_map = {'Mandante': COR_MAN, 'Empate': COR_EMP, 'Visitante': COR_VIS}
            fig_res = px.pie(res_counts, names='Resultado', values='Contagem', hole=0.6, color='Resultado', color_discrete_map=color_map)
            fig_res.update_traces(textposition='outside', textinfo='percent+label', marker=dict(colors=[color_map[x] for x in res_counts['Resultado']]))
            fig_res.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
            st.plotly_chart(fig_res, use_container_width=True)
            
        with colD:
            st.markdown("#### 🌤️ Impacto Climático Histórico")
            clima = df_subset.groupby("Temporada")[["Temperatura_C", "Umidade_Relativa_%"]].mean().reset_index()
            fig_clima = go.Figure()
            hex_to_rgb = lambda hex_val: f"{int(hex_val[1:3], 16)}, {int(hex_val[3:5], 16)}, {int(hex_val[5:7], 16)}"
            fig_clima.add_trace(go.Scatter(x=clima["Temporada"], y=clima["Temperatura_C"], mode='lines', name='Temp (°C)', stackgroup='one', fillcolor=f'rgba({hex_to_rgb(COR_VIS)}, 0.3)', line=dict(color=COR_VIS)))
            fig_clima.add_trace(go.Scatter(x=clima["Temporada"], y=clima["Umidade_Relativa_%"], mode='lines', name='Umid (%)', stackgroup='two', fillcolor=f'rgba({hex_to_rgb(COR_MAN)}, 0.3)', line=dict(color=COR_MAN)))
            fig_clima.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified")
            st.plotly_chart(fig_clima, use_container_width=True)

# ---------------------------------------------------------------------
# FUNÇÃO AUXILIAR PARA TRADUZIR AS FEATURES
# ---------------------------------------------------------------------
def traduzir_feature(f):
    if "Diff" in f:
        j = f.split("_")[0].replace("Diff", "")
        tipo = f.split("_")[1]
        if tipo == "Ataque": return f"Saldo de Gols Pró (Casa - Fora) | {j} Jogos"
        if tipo == "Defesa": return f"Saldo de Defesa Sofrida (Casa - Fora) | {j} Jogos"
        if tipo == "Volume": return f"Saldo de Volume de Finalizações (Casa - Fora) | {j} Jogos"
    
    partes = f.split("_")
    time = "Casa" if partes[0] == "Mandante" else "Fora"
    j = partes[1].replace("Media", "")
    
    if "Gols_Feitos" in f: return f"Média Gols Feitos ({time}) | {j} Jogos"
    if "Gols_Sofridos" in f: return f"Média Gols Sofridos ({time}) | {j} Jogos"
    if "Fin_Feitas" in f: return f"Média Finalizações Feitas ({time}) | {j} Jogos"
    if "Fin_Sofridas" in f: return f"Média Finalizações Sofridas ({time}) | {j} Jogos"
    
    return f

# ---------------------------------------------------------------------
# FUNÇÃO PARA RENDERIZAR TAB DO MODELO DE ML
# ---------------------------------------------------------------------
def renderizar_tab_modelo(liga_nome):
    with st.spinner("Carregando o modelo e executando Backtest Walk-Forward (Isso pode levar alguns segundos)..."):
        nome_mod, limiar, df_ap, df_fut, params, pipeline_melhor, colunas_treino = treinar_e_avaliar_modelo(df_geral, liga_nome)
        
    lucro = df_ap['Resultado_Aposta'].sum()
    roi = (lucro / len(df_ap) * 100) if len(df_ap) > 0 else 0
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🧠 Modelo Base Selecionado", nome_mod, help="Modelo carregado do arquivo pkl.")
    c2.metric("🎯 Limiar EV Ótimo", f"{limiar*100:.1f}%", help="Ponto de corte de Valor Esperado fixo.")
    c3.metric("📈 Entradas (Teste Cego WF)", len(df_ap), help="Quantidade de jogos com valor usando Walk-Forward.")
    c4.metric("💰 Retorno s/ Investimento", f"{roi:.2f}%", f"{lucro:.2f}u", help="Lucratividade percentual.")

    colA, colB = st.columns([2, 3])
    with colA:
        st.markdown("<br>#### Desempenho no Backtest (Walk-Forward)", unsafe_allow_html=True)
        if len(df_ap) > 0:
            fig, ax = plt.subplots(figsize=(8, 5.5))
            cor_linha = COR_POS if lucro > 0 else COR_NEG
            ax.plot(range(len(df_ap)), df_ap['Lucro_Acumulado'], color=cor_linha, lw=2)
            ax.axhline(0, color='#30363d', ls='--')
            ax.fill_between(range(len(df_ap)), df_ap['Lucro_Acumulado'], 0, where=(df_ap['Lucro_Acumulado'] >= 0), color=COR_POS, alpha=0.15)
            ax.fill_between(range(len(df_ap)), df_ap['Lucro_Acumulado'], 0, where=(df_ap['Lucro_Acumulado'] < 0), color=COR_NEG, alpha=0.15)
            ax.set_ylabel("Unidades Financeiras", fontsize=9)
            ax.set_xlabel("Número de Apostas", fontsize=9)
            ax.grid(True, linestyle=':', alpha=0.3)
            st.pyplot(fig)
        else:
            st.warning("O modelo não encontrou apostas no teste.")

    with colB:
        st.markdown("<br>#### Previsões Futuras (Próximos Jogos)", unsafe_allow_html=True)
        if not df_fut.empty:
            df_fut['Decisão'] = np.where(df_fut['EV (%)'] > (limiar*100), TXT_ENTRAR, TXT_NAO_ENTRAR)
            cols_show = ['Data_Hora', 'Mandante', 'Visitante', 'Odd_Mandante', 'Prob_Modelo_Num', 'EV (%)', 'Decisão']
            display_fut = df_fut[cols_show].copy()
            
            st.dataframe(
                display_fut,
                column_config={
                    "Data_Hora": "Data do Jogo",
                    "Mandante": "Casa",
                    "Visitante": "Fora",
                    "Odd_Mandante": st.column_config.NumberColumn("Odd Casa", format="%.2f"),
                    "Prob_Modelo_Num": st.column_config.ProgressColumn(
                        "Probabilidade IA",
                        help="Chance calculada pelo modelo para vitória do mandante.",
                        format="%.2f",
                        min_value=0.0,
                        max_value=1.0,
                    ),
                    "EV (%)": st.column_config.NumberColumn("Valor Esperado", format="%.2f%%"),
                    "Decisão": st.column_config.TextColumn("Recomendação")
                },
                hide_index=True,
                height=350,
                use_container_width=True
            )
        else:
            st.info("Nenhum jogo sem resultado final foi encontrado na base de dados para gerar previsões futuras.")

    # =====================================================================
    # SEÇÃO: FEATURE IMPORTANCE / INTERPRETABILIDADE
    # =====================================================================
    st.markdown("---")
    st.markdown("### 🔍 Interpretabilidade do Modelo (Top 10 Variáveis)")
    st.markdown("Entenda quais variáveis estatísticas e táticas tiveram o maior peso na decisão da Inteligência Artificial.")
    
    try:
        modelo_interno = pipeline_melhor.named_steps['model']
        has_importance = hasattr(modelo_interno, 'feature_importances_')
        has_coef = hasattr(modelo_interno, 'coef_')
        
        if has_importance or has_coef:
            importancias = modelo_interno.feature_importances_ if has_importance else np.abs(modelo_interno.coef_[0])
            tipo_calculo = "Redução de Impureza nas Árvores (Gini/Log Loss)" if has_importance else "Valor Absoluto do Coeficiente Logístico"
            
            df_imp = pd.DataFrame({
                'Variável Original': colunas_treino,
                'Importância / Peso': importancias
            })
            
            df_imp['Descrição Mapeada'] = df_imp['Variável Original'].apply(traduzir_feature)
            df_imp = df_imp.sort_values(by='Importância / Peso', ascending=False).head(10)
            
            col_grafico, col_texto = st.columns([2, 1])
            
            with col_grafico:
                # Ordenar inverso para o gráfico horizontal mostrar o maior em cima
                df_imp_grafico = df_imp.sort_values(by='Importância / Peso', ascending=True)
                fig_imp = px.bar(
                    df_imp_grafico, 
                    x='Importância / Peso', 
                    y='Descrição Mapeada', 
                    orientation='h',
                    color='Importância / Peso',
                    color_continuous_scale='viridis'
                )
                fig_imp.update_layout(
                    template="plotly_dark",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    showlegend=False,
                    margin=dict(l=0, r=0, t=30, b=0),
                    yaxis_title="",
                    xaxis_title="Peso da Variável"
                )
                st.plotly_chart(fig_imp, use_container_width=True)
                
            with col_texto:
                st.info(f"**Como o cálculo é feito?**\n\nEste modelo utiliza o método de **{tipo_calculo}** para classificar as variáveis.")
                if has_coef:
                    st.write("Como a base do modelo é uma **Regressão Logística**, os dados passam por uma padronização pesada (RobustScaler). Após isso, o modelo atribui um coeficiente matemático para cada variável na equação de probabilidade. O gráfico ao lado mostra o valor absoluto desse coeficiente, ou seja, as variáveis com barras maiores são as que **puxam a probabilidade mais fortemente** para uma vitória do mandante ou contra ele.")
                else:
                    st.write("Como a base do modelo é um Ensemble de Árvores de Decisão (como Random Forest ou Boostings), a importância de uma variável mede o quanto ela ajudou a reduzir o erro (impureza) ao longo de todas as divisões criadas pelas árvores. Quanto maior a barra, **mais vezes a IA dependeu dessa estatística** para separar os jogos entre 'Vitória da Casa' e 'Tropeço'.")
                
                with st.expander("Ver Tabela Bruta (Top 10)"):
                    st.dataframe(df_imp[['Descrição Mapeada', 'Importância / Peso']].reset_index(drop=True), use_container_width=True)
        else:
            st.info("O modelo selecionado não possui um método nativo suportado para extração direta de importância de variáveis.")
            
    except Exception as e:
        st.warning(f"Não foi possível processar a importância das variáveis para este modelo. Erro: {e}")
        
    with st.expander("🛠️ Ver Hiperparâmetros Vencedores"):
        st.json(params)

with tab2:
    renderizar_tab_modelo("Brasileirão Série A")

with tab3:
    renderizar_tab_modelo("Premier League")

# ---------------------------------------------------------------------
# TAB 4: SIMULADOR DE APOSTAS
# ---------------------------------------------------------------------
with tab4:
    st.header("🎲 Simulador de Apostas")
    st.markdown("Avalie a variância no curto prazo. O sistema cria N usuários apostando na vitória do Mandante, com as regras ativas na barra lateral.")
    
    df_sim_pool = df_geral[
        (df_geral["Liga"].isin(ligas_sel)) & 
        (df_geral["Temporada"].isin(anos_sel)) &
        (df_geral["Mes"].isin(meses_sel))
    ].copy()
    
    if filtro_mandante_ativo:
        df_sim_pool = df_sim_pool[(df_sim_pool["Odd_Mandante"] >= odd_m_min) & (df_sim_pool["Odd_Mandante"] <= odd_m_max)]
    if filtro_empate_ativo:
        df_sim_pool = df_sim_pool[(df_sim_pool["Odd_Empate"] >= odd_e_min) & (df_sim_pool["Odd_Empate"] <= odd_e_max)]
    if filtro_visitante_ativo:
        df_sim_pool = df_sim_pool[(df_sim_pool["Odd_Visitante"] >= odd_v_min) & (df_sim_pool["Odd_Visitante"] <= odd_v_max)]
        
    df_sim_pool = df_sim_pool.dropna(subset=["Odd_Mandante", "Resultado"])

    st.info(f"O seu cruzamento de filtros tem **{len(df_sim_pool)}** partidas para o simulador.")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        num_apostas = st.number_input("Número de Apostas por Usuário:", min_value=5, max_value=1000, value=20, step=5, help="Quantos jogos o sistema vai sortear.")
    with col_s2:
        num_usuarios = st.number_input("Quantidade de Usuários Simulados:", min_value=1, max_value=100, value=6, step=1, help="Quantas linhas no gráfico.")
        
    if st.button("🚀 Iniciar Simulação", use_container_width=True):
        if len(df_sim_pool) < 1:
            st.error("Não há jogos suficientes no filtro para a simulação.")
        else:
            with st.spinner(f"Simulando {num_usuarios} usuários fazendo {num_apostas} apostas cada..."):
                trajetorias = {}
                rois_finais = []
                
                for i in range(1, num_usuarios + 1):
                    amostra = df_sim_pool.sample(n=num_apostas, replace=True).reset_index(drop=True)
                    lucro_array = np.where(amostra["Resultado"] == "Mandante", amostra["Odd_Mandante"] - 1, -1)
                    lucro_acumulado = np.cumsum(lucro_array)
                    
                    trajetorias[f"Usuário {i}"] = lucro_acumulado
                    rois_finais.append((lucro_acumulado[-1] / num_apostas) * 100)
                
                roi_medio = np.mean(rois_finais)
                prob_lucro = (np.array(rois_finais) > 0).mean() * 100
                
                st.markdown("### 📊 Resultados Globais da Simulação")
                c_res1, c_res2, c_res3, c_res4 = st.columns(4)
                c_res1.metric("Média do ROI Final", f"{roi_medio:.2f}%", help="O ROI médio do filtro.")
                c_res2.metric("Usuários no Lucro", f"{prob_lucro:.1f}%", help="Porcentagem de apostadores que terminaram acima de zero.")
                c_res3.metric("Pior ROI", f"{np.min(rois_finais):.2f}%", help="O pior ROI do grupo.")
                c_res4.metric("Melhor ROI", f"{np.max(rois_finais):.2f}%", help="O melhor ROI do grupo.")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                fig_sim = go.Figure()
                eixo_x = np.arange(1, num_apostas + 1)
                
                for usuario, traj in trajetorias.items():
                    fig_sim.add_trace(go.Scatter(
                        x=eixo_x, 
                        y=traj, 
                        mode='lines', 
                        name=usuario,
                        opacity=0.8,
                        line=dict(width=2)
                    ))
                
                fig_sim.add_hline(y=0, line_dash="solid", line_color="#8b949e", line_width=2, annotation_text=" Zero", annotation_position="top left")
                
                fig_sim.update_layout(
                    title={
                        'text': "Evolução do Saldo por Usuário",
                        'font': {'size': 20, 'color': '#ffffff'}
                    },
                    xaxis_title="Número da Aposta",
                    yaxis_title="Lucro Acumulado (Unidades)",
                    template="plotly_dark", 
                    plot_bgcolor='rgba(0,0,0,0)', 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    margin=dict(l=0, r=0, t=50, b=0),
                    hovermode="x unified",
                    legend_title_text='Simulações',
                    legend=dict(
                        font=dict(size=14, color="#ffffff"),
                        title_font=dict(size=16, color="#ffffff")
                    ),
                    xaxis=dict(
                        title_font=dict(size=16, color="#ffffff"),
                        tickfont=dict(size=14, color="#e0e0e0")
                    ),
                    yaxis=dict(
                        title_font=dict(size=16, color="#ffffff"),
                        tickfont=dict(size=14, color="#e0e0e0")
                    )
                )
                st.plotly_chart(fig_sim, use_container_width=True)
                
                st.markdown("### 📋 Exemplo: Apostas do Último Usuário Sorteado")
                st.markdown("Veja os jogos do último usuário do gráfico e o impacto no caixa.")
                
                amostra_exemplo = amostra[["Data_Hora", "Mandante", "Visitante", "Odd_Mandante", "Resultado"]].copy()
                amostra_exemplo["Lucro_Jogo"] = lucro_array
                amostra_exemplo["Lucro_Acumulado"] = lucro_acumulado
                amostra_exemplo.columns = ["Data do Jogo", "Casa", "Fora", "Odd Casa", "Vencedor Real", "P/L da Aposta", "Caixa Acumulado"]
                
                st.dataframe(
                    amostra_exemplo,
                    column_config={
                        "Odd Casa": st.column_config.NumberColumn(format="%.2f"),
                        "P/L da Aposta": st.column_config.NumberColumn(format="%+.2f u"),
                        "Caixa Acumulado": st.column_config.NumberColumn(format="%+.2f u")
                    },
                    hide_index=True,
                    use_container_width=True
                )

# ---------------------------------------------------------------------
# TAB 5: EFICIÊNCIA DAS ODDS
# ---------------------------------------------------------------------
with tab5:
    st.header("⚖️ Eficiência das Odds")
    st.markdown("Descubra se as casas de apostas precificam as ligas de forma correta. A eficiência é medida comparando a Probabilidade Implícita na Odd contra a Probabilidade Real da sua seleção.")
    
    st.info("💡 **Dica de Leitura:** Se o valor da coluna **Eficiência** for positivo, o mercado subestimou o evento e houve **Valor** naqueles filtros. Se for negativo, as odds estavam a favor da casa.")

    col_liga1, col_liga2 = st.columns(2)
    ligas_disp = list(df_geral["Liga"].unique())
    
    with col_liga1:
        liga_escolhida_1 = st.selectbox("Selecione a Liga A para comparar:", options=ligas_disp, index=0)
    with col_liga2:
        liga_escolhida_2 = st.selectbox("Selecione a Liga B para comparar:", options=ligas_disp, index=1 if len(ligas_disp) > 1 else 0)

    def render_eficiencia_linha(liga_nome, coluna_render):
        df_efi = df_geral[
            (df_geral["Liga"] == liga_nome) & 
            (df_geral["Temporada"].isin(anos_sel)) &
            (df_geral["Mes"].isin(meses_sel))
        ].copy()
        
        if filtro_mandante_ativo:
            df_efi = df_efi[(df_efi["Odd_Mandante"] >= odd_m_min) & (df_efi["Odd_Mandante"] <= odd_m_max)]
        if filtro_empate_ativo:
            df_efi = df_efi[(df_efi["Odd_Empate"] >= odd_e_min) & (df_efi["Odd_Empate"] <= odd_e_max)]
        if filtro_visitante_ativo:
            df_efi = df_efi[(df_efi["Odd_Visitante"] >= odd_v_min) & (df_efi["Odd_Visitante"] <= odd_v_max)]
            
        df_efi = df_efi.dropna(subset=["Odd_Mandante", "Odd_Empate", "Odd_Visitante", "Resultado"])

        with coluna_render:
            st.markdown(f"### 🏆 {liga_nome}")
            
            if len(df_efi) == 0:
                st.warning(f"Nenhum jogo da liga {liga_nome} encontrado.")
                return
                
            st.markdown(f"**Tamanho da Amostra:** {len(df_efi)} partidas.")

            odd_m_avg = df_efi["Odd_Mandante"].mean()
            odd_e_avg = df_efi["Odd_Empate"].mean()
            odd_v_avg = df_efi["Odd_Visitante"].mean()

            imp_m = (1 / odd_m_avg) * 100 if odd_m_avg > 0 else 0
            imp_e = (1 / odd_e_avg) * 100 if odd_e_avg > 0 else 0
            imp_v = (1 / odd_v_avg) * 100 if odd_v_avg > 0 else 0

            real_m = (df_efi["Resultado"] == "Mandante").mean() * 100
            real_e = (df_efi["Resultado"] == "Empate").mean() * 100
            real_v = (df_efi["Resultado"] == "Visitante").mean() * 100

            df_display = pd.DataFrame({
                "Mercado": ["Vitória Mandante", "Empate", "Vitória Visitante"],
                "Odd Média": [odd_m_avg, odd_e_avg, odd_v_avg],
                "Probabilidade Implícita": [imp_m, imp_e, imp_v],
                "Probabilidade Real": [real_m, real_e, real_v],
                "Eficiência": [real_m - imp_m, real_e - imp_e, real_v - imp_v]
            })

            st.dataframe(
                df_display.style.format({
                    "Odd Média": "{:.2f}",
                    "Probabilidade Implícita": "{:.1f}%",
                    "Probabilidade Real": "{:.1f}%",
                    "Eficiência": "{:+.1f}%"
                }).map(
                    lambda x: f"color: {COR_POS}; font-weight: bold;" if x > 0 else f"color: {COR_NEG}; font-weight: bold;", 
                    subset=["Eficiência"]
                ),
                hide_index=True,
                use_container_width=True
            )

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Probabilidade da Odd', 
                x=df_display['Mercado'], 
                y=df_display['Probabilidade Implícita'],
                marker_color='#8b949e',
                text=df_display['Probabilidade Implícita'].apply(lambda x: f"{x:.1f}%"),
                textposition='auto'
            ))
            fig.add_trace(go.Bar(
                name='Ocorrência Real', 
                x=df_display['Mercado'], 
                y=df_display['Probabilidade Real'],
                marker_color=COR_MAN,
                text=df_display['Probabilidade Real'].apply(lambda x: f"{x:.1f}%"),
                textposition='auto'
            ))

            fig.update_layout(
                barmode='group',
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
                yaxis=dict(title="Probabilidade (%)")
            )
            st.plotly_chart(fig, use_container_width=True)

    render_eficiencia_linha(liga_escolhida_1, col_liga1)
    render_eficiencia_linha(liga_escolhida_2, col_liga2)