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
from scipy.stats import uniform, randint
import warnings

warnings.filterwarnings('ignore')

# Configuração da Página
st.set_page_config(page_title="AI Sports Analytics", layout="wide", page_icon="📈")

# =====================================================================
# CSS CUSTOMIZADO (FULL DARK THEME PREMIUM)
# =====================================================================
st.markdown("""
<style>
    /* Fundo geral super escuro e cor de texto padrão */
    .stApp, .main {
        background-color: #0d1117 !important;
        color: #ffffff !important;
    }
    
    /* BARRA LATERAL (SIDEBAR) */
    section[data-testid="stSidebar"] {
        background-color: #0d1117 !important;
        border-right: 1px solid #30363d !important;
    }
    section[data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }

    /* MULTISELECTS, MENUS SUSPENSOS E CAIXAS DE BUSCA */
    div[data-baseweb="select"] > div, 
    div[data-baseweb="base-input"] > input {
        background-color: #161b22 !important;
        color: #ffffff !important;
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
    }
    span[data-baseweb="tag"] {
        background-color: #3498db !important;
        color: #ffffff !important;
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

    /* CARTÕES DE KPI (METRICS) FECHADOS */
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

    /* ABAS SUPERIORES */
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

# Força o Matplotlib a usar o estilo escuro transparente
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

# Header da Aplicação
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

    # Todas as colunas numéricas (incluindo as odds e defesas)
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
    
    # Variáveis Totais para KPIs
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
        df['Data_Hora_DT'] = pd.to_datetime(df['Data_Hora'], dayfirst=True, errors='coerce')
        df = df.sort_values(by=['Liga', 'Data_Hora_DT']).reset_index(drop=True)
        df['Mes'] = df['Data_Hora_DT'].dt.month.fillna(0).astype(int)

    return df

@st.cache_resource(show_spinner=False)
def treinar_e_avaliar_modelo(df_base, liga_nome):
    df_liga = df_base[df_base['Liga'] == liga_nome].copy()
    
    janelas = [3, 5]
    df_features = df_liga.copy()

    df_m = df_liga[['Data_Hora_DT', 'Mandante', 'Gols_Mandante', 'Gols_Visitante', 'Finalizacoes_Mandante', 'Finalizacoes_Visitante']].copy()
    df_m.columns = ['Data_Hora_DT', 'Equipe', 'Gols_Feitos', 'Gols_Sofridos', 'Fin_Feitas', 'Fin_Sofridas']
    df_v = df_liga[['Data_Hora_DT', 'Visitante', 'Gols_Visitante', 'Gols_Mandante', 'Finalizacoes_Visitante', 'Finalizacoes_Mandante']].copy()
    df_v.columns = ['Data_Hora_DT', 'Equipe', 'Gols_Feitos', 'Gols_Sofridos', 'Fin_Feitas', 'Fin_Sofridas']

    df_times = pd.concat([df_m, df_v]).sort_values(by=['Equipe', 'Data_Hora_DT']).reset_index(drop=True)

    cols_calc = ['Gols_Feitos', 'Gols_Sofridos', 'Fin_Feitas', 'Fin_Sofridas']
    for j in janelas:
        for col in cols_calc:
            df_times[f'Media{j}_{col}'] = df_times.groupby('Equipe')[col].transform(lambda x: x.shift(1).rolling(window=j, min_periods=1).mean())

    df_times_stats = df_times[['Data_Hora_DT', 'Equipe'] + [c for c in df_times.columns if 'Media' in c]].drop_duplicates(subset=['Data_Hora_DT', 'Equipe'])

    df_features = df_features.merge(df_times_stats, left_on=['Data_Hora_DT', 'Mandante'], right_on=['Data_Hora_DT', 'Equipe'], how='left').drop(columns=['Equipe'])
    df_features.rename(columns={c: f'Mandante_{c}' for c in df_times_stats.columns if 'Media' in c}, inplace=True)
    
    df_features = df_features.merge(df_times_stats, left_on=['Data_Hora_DT', 'Visitante'], right_on=['Data_Hora_DT', 'Equipe'], how='left').drop(columns=['Equipe'])
    df_features.rename(columns={c: f'Visitante_{c}' for c in df_times_stats.columns if 'Media' in c}, inplace=True)

    for j in janelas:
        df_features[f'Diff{j}_Ataque'] = df_features[f'Mandante_Media{j}_Gols_Feitos'] - df_features[f'Visitante_Media{j}_Gols_Feitos']
        df_features[f'Diff{j}_Defesa'] = df_features[f'Mandante_Media{j}_Gols_Sofridos'] - df_features[f'Visitante_Media{j}_Gols_Sofridos']
        df_features[f'Diff{j}_Volume'] = df_features[f'Mandante_Media{j}_Fin_Feitas'] - df_features[f'Visitante_Media{j}_Fin_Feitas']

    df_features['Alvo_Vitoria'] = (df_features['Gols_Mandante'] > df_features['Gols_Visitante']).astype(int)
    colunas_treino = [c for c in df_features.columns if ('Media' in c or 'Diff' in c)]
    cols_essenciais = colunas_treino + ['Odd_Mandante']

    df_futuro = df_features[df_features['Temporada'] == '2026'].copy()
    df_hist = df_features[df_features['Temporada'] != '2026'].dropna(subset=cols_essenciais).copy()
    df_hist = df_hist[df_hist['Odd_Mandante'] > 1.01].sort_values(by='Data_Hora_DT').reset_index(drop=True)

    n = len(df_hist)
    idx_treino = int(n * 0.60)
    idx_val = int(n * 0.80)

    X, y = df_hist[colunas_treino], df_hist['Alvo_Vitoria']
    X_treino, y_treino = X.iloc[:idx_treino], y.iloc[:idx_treino]
    X_val, y_val = X.iloc[idx_treino:idx_val], y.iloc[idx_treino:idx_val]
    X_teste, y_teste = X.iloc[idx_val:], y.iloc[idx_val:]

    tscv = TimeSeriesSplit(n_splits=3)
    
    modelos_params = [
        ('Random Forest', Pipeline([('scaler', RobustScaler()), ('model', RandomForestClassifier(random_state=42, n_jobs=-1))]),
         {'model__n_estimators': randint(100, 500), 'model__max_depth': randint(3, 10)}),
        ('XGBoost', Pipeline([('scaler', RobustScaler()), ('model', xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', n_jobs=-1, random_state=42))]),
         {'model__n_estimators': randint(100, 500), 'model__learning_rate': uniform(0.01, 0.1), 'model__max_depth': randint(3, 8)}),
        ('Regressão Logística', Pipeline([('scaler', RobustScaler()), ('model', LogisticRegression(random_state=42, max_iter=1000))]),
         {'model__C': uniform(0.1, 10)})
    ]

    resultados_modelos = {}
    for nome, pipe, param_dist in modelos_params:
        rs = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=5, scoring='neg_log_loss', cv=tscv, random_state=42, n_jobs=-1)
        rs.fit(X_treino, y_treino)
        melhor_pipe = rs.best_estimator_
        probs_val = melhor_pipe.predict_proba(X_val)[:, 1]
        resultados_modelos[nome] = {
            'pipeline': melhor_pipe,
            'probs_val': probs_val,
            'auc_val': roc_auc_score(y_val, probs_val),
            'best_score': rs.best_score_,
            'best_params': rs.best_params_
        }

    nome_melhor = max(resultados_modelos, key=lambda k: resultados_modelos[k]['auc_val'])
    info_melhor = resultados_modelos[nome_melhor]
    pipeline_melhor = info_melhor['pipeline']

    df_val_sim = df_hist.iloc[idx_treino:idx_val].copy()
    df_val_sim['Prob_Modelo'] = info_melhor['probs_val']
    df_val_sim['Odd_Justa_Modelo'] = 1 / df_val_sim['Prob_Modelo']
    df_val_sim['EV_Porcentagem'] = (df_val_sim['Odd_Mandante'] / df_val_sim['Odd_Justa_Modelo']) - 1

    melhor_limiar, melhor_lucro_loop = 0.0, -999999
    for val_limiar in np.arange(0.0, 0.31, 0.02):
        mask = df_val_sim['EV_Porcentagem'] > val_limiar
        sub_df = df_val_sim[mask]
        if len(sub_df) >= 5:
            lucro_temp = np.where(sub_df['Alvo_Vitoria'] == 1, sub_df['Odd_Mandante'] - 1, -1).sum()
            if lucro_temp > melhor_lucro_loop:
                melhor_lucro_loop = lucro_temp
                melhor_limiar = val_limiar

    probs_teste = pipeline_melhor.predict_proba(X_teste)[:, 1]
    df_teste = df_hist.iloc[idx_val:].copy()
    df_teste['Prob_Modelo'] = probs_teste
    df_teste['EV_Porcentagem'] = (df_teste['Odd_Mandante'] * df_teste['Prob_Modelo']) - 1
    
    df_apostas = df_teste[df_teste['EV_Porcentagem'] > melhor_limiar].copy()
    df_apostas['Resultado_Aposta'] = np.where(df_apostas['Alvo_Vitoria'] == 1, df_apostas['Odd_Mandante'] - 1, -1)
    df_apostas['Lucro_Acumulado'] = df_apostas['Resultado_Aposta'].cumsum()

    df_futuro_final = pd.DataFrame()
    if not df_futuro.empty:
        df_futuro = df_futuro.dropna(subset=colunas_treino)
        if not df_futuro.empty:
            probs_fut = pipeline_melhor.predict_proba(df_futuro[colunas_treino])[:, 1]
            df_futuro['Prob_Modelo_Num'] = probs_fut
            df_futuro['Odd_Justa'] = 1 / df_futuro['Prob_Modelo_Num']
            df_futuro['EV (%)'] = ((df_futuro['Odd_Mandante'] / df_futuro['Odd_Justa']) - 1) * 100
            df_futuro_final = df_futuro

    return nome_melhor, melhor_limiar, df_apostas, df_futuro_final, info_melhor['best_params']


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

    st.markdown("---")
    st.markdown("### 📅 Filtros do Painel")
    
    anos_disponiveis = sorted(df_geral["Temporada"].dropna().unique(), reverse=True)
    anos_sel = st.multiselect("Anos:", anos_disponiveis, default=anos_disponiveis)
    
    meses_disponiveis = sorted([m for m in df_geral["Mes"].unique() if m != 0])
    nomes_meses = {1:"Jan", 2:"Fev", 3:"Mar", 4:"Abr", 5:"Mai", 6:"Jun", 7:"Jul", 8:"Ago", 9:"Set", 10:"Out", 11:"Nov", 12:"Dez"}
    meses_formatados = [f"{m} - {nomes_meses[m]}" for m in meses_disponiveis]
    meses_sel_str = st.multiselect("Meses:", meses_formatados, default=meses_formatados)
    meses_sel = [int(m.split(" - ")[0]) for m in meses_sel_str]

    st.markdown("---")
    st.markdown("### 📊 Filtro de Odds (Casa)")
    odd_min, odd_max = st.slider(
        "Selecione o range de Odd Mandante:",
        min_value=1.0, max_value=15.0, value=(1.0, 15.0), step=0.1
    )

# ---------------------------------------------------------------------
# ABAS PRINCIPAIS
# ---------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Estatísticas e Tática", "🤖 Inteligência (Brasileirão)", "🤖 Inteligência (Premier League)"])

# ---------------------------------------------------------------------
# TAB 1: ESTATÍSTICAS E GRÁFICOS (PLOTLY)
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
            (df_geral["Mes"].isin(meses_sel)) &
            (df_geral["Odd_Mandante"] >= odd_min) &
            (df_geral["Odd_Mandante"] <= odd_max)
        ].copy()
        
        times_disponiveis = sorted(list(set(df_liga_filter["Mandante"].unique()) | set(df_liga_filter["Visitante"].unique())))
        
        with col_f2:
            selecionar_todos = st.toggle("Selecionar todas as equipes do filtro", value=True)
            if selecionar_todos:
                times_sel = times_disponiveis
            else:
                times_sel = st.multiselect("Equipes:", times_disponiveis, default=[times_disponiveis[0]] if times_disponiveis else [])
                
    df_subset = df_liga_filter[(df_liga_filter["Mandante"].isin(times_sel)) | (df_liga_filter["Visitante"].isin(times_sel))].copy()

    if df_subset.empty:
        st.warning("Nenhum dado encontrado com a combinação de filtros atuais.")
    else:
        # =========================================================
        # BLOCO DE KPIs (3 LINHAS)
        # =========================================================
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

        # --- LINHA 1 ---
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(label="🏟️ Partidas Filtradas", value=f"{total_jogos}")
        k2.metric(label="⚽ Gols por Jogo (Média)", value=f"{media_gols:.2f}")
        pub_val = f"{media_publico:,.0f}".replace(',', '.') if pd.notna(media_publico) else "N/A"
        k3.metric(label="👥 Público Médio", value=pub_val)
        k4.metric(label="🏠 Win Rate Mandante", value=f"{vit_mandante_pct:.1f}%")
        
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        
        # --- LINHA 2 ---
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
        
        # --- LINHA 3 ---
        k9, k10, k11, k12 = st.columns(4)
        val_om = f"{odd_m_media:.2f}" if pd.notna(odd_m_media) else "N/A"
        val_oe = f"{odd_e_media:.2f}" if pd.notna(odd_e_media) else "N/A"
        val_ov = f"{odd_v_media:.2f}" if pd.notna(odd_v_media) else "N/A"
        
        k9.metric(label="📊 Odd Mandante (Média)", value=val_om)
        k10.metric(label="📊 Odd Empate (Média)", value=val_oe)
        k11.metric(label="📊 Odd Visitante (Média)", value=val_ov)
        k12.metric(label="🔥 Ambas Marcam (BTTS)", value=f"{btts_pct:.1f}%")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # =========================================================
        # GRÁFICOS (PLOTLY AVANÇADO)
        # =========================================================
        colA, colB = st.columns([2, 2])
        
        with colA:
            st.markdown("#### ⚖️ Comparativo de Força (Casa vs Fora)")
            radar_stats = {
                'Métricas': ['Posse de Bola (%)', 'Finalizações', 'Finalizações no Alvo', 'Escanteios', 'Faltas'],
                'Mandante': [
                    df_subset['Posse_Mandante'].mean() if 'Posse_Mandante' in df_subset else 0, 
                    df_subset['Finalizacoes_Mandante'].mean(), 
                    df_subset['No_Alvo_Mandante'].mean() if 'No_Alvo_Mandante' in df_subset else 0, 
                    df_subset['Escanteios_Mandante'].mean() if 'Escanteios_Mandante' in df_subset else 0, 
                    df_subset['Faltas_Mandante'].mean() if 'Faltas_Mandante' in df_subset else 0
                ],
                'Visitante': [
                    df_subset['Posse_Visitante'].mean() if 'Posse_Visitante' in df_subset else 0, 
                    df_subset['Finalizacoes_Visitante'].mean(), 
                    df_subset['No_Alvo_Visitante'].mean() if 'No_Alvo_Visitante' in df_subset else 0, 
                    df_subset['Escanteios_Visitante'].mean() if 'Escanteios_Visitante' in df_subset else 0, 
                    df_subset['Faltas_Visitante'].mean() if 'Faltas_Visitante' in df_subset else 0
                ]
            }
            radar_stats['Mandante'] = [x if pd.notna(x) else 0 for x in radar_stats['Mandante']]
            radar_stats['Visitante'] = [x if pd.notna(x) else 0 for x in radar_stats['Visitante']]
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=radar_stats['Mandante'], theta=radar_stats['Métricas'], fill='toself', name='Casa (Mandante)',
                line_color=COR_MAN, fillcolor=f"rgba({int(COR_MAN[1:3],16)}, {int(COR_MAN[3:5],16)}, {int(COR_MAN[5:7],16)}, 0.4)"
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=radar_stats['Visitante'], theta=radar_stats['Métricas'], fill='toself', name='Fora (Visitante)',
                line_color=COR_VIS, fillcolor=f"rgba({int(COR_VIS[1:3],16)}, {int(COR_VIS[3:5],16)}, {int(COR_VIS[5:7],16)}, 0.4)"
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, showticklabels=False, gridcolor='#30363d'), bgcolor='#0d1117'),
                template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=40, r=40, t=30, b=30), showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
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
# FUNÇÃO PARA RENDERIZAR TAB DO MODELO DE ML
# ---------------------------------------------------------------------
def renderizar_tab_modelo(liga_nome):
    with st.spinner("Construindo Pipeline Preditivo e Rolagem de Médias. Aguarde..."):
        nome_mod, limiar, df_ap, df_fut, params = treinar_e_avaliar_modelo(df_geral, liga_nome)
        
    lucro = df_ap['Resultado_Aposta'].sum()
    roi = (lucro / len(df_ap) * 100) if len(df_ap) > 0 else 0
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🧠 Modelo Base Selecionado", nome_mod, help="Modelo escolhido através do RandomizedSearchCV no conjunto de treino por menor erro.")
    c2.metric("🎯 Limiar EV Ótimo", f"{limiar*100:.1f}%", help="Ponto de corte ótimo de Valor Esperado descoberto no conjunto de Validação.")
    c3.metric("📈 Entradas (Teste Cego)", len(df_ap), help="Quantidade de jogos em que o modelo identificou valor real.")
    c4.metric("💰 Retorno s/ Investimento", f"{roi:.2f}%", f"{lucro:.2f}u", help="Lucratividade percentual se houvesse apostado 1 unidade padrão.")

    colA, colB = st.columns([2, 3])
    with colA:
        st.markdown("<br>#### Desempenho no Backtest", unsafe_allow_html=True)
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
            st.warning("O modelo não encontrou apostas seguras no teste.")

    with colB:
        st.markdown("<br>#### Previsões Futuras (Calendário 2026)", unsafe_allow_html=True)
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
            st.info("Nenhum jogo futuro pendente encontrado com as métricas históricas preenchidas.")
            
    with st.expander("🛠️ Ver Hiperparâmetros Vencedores do Pipeline"):
        st.json(params)

with tab2:
    renderizar_tab_modelo("Brasileirão Série A")

with tab3:
    renderizar_tab_modelo("Premier League")