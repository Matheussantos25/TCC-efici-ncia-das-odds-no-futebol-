# ⚽ Evolução da Eficiência das Odds no Mercado de Apostas de Futebol

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://oddseficientes.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn%20%7C%20XGBoost%20%7C%20LightGBM-orange)]()
[![Web Scraping](https://img.shields.io/badge/Web%20Scraping-Selenium%20%7C%20BeautifulSoup-green)]()

Um ecossistema completo de Inteligência Artificial e Analytics aplicado ao futebol (Brasileirão Série A e Premier League). Este projeto analisa a eficiência das casas de apostas, identificando oportunidades de +EV (Valor Esperado Positivo) através de algoritmos de Machine Learning, estatísticas avançadas e fatores climáticos.

---

## 🎯 Objetivo do Projeto
Analisar como as odds das casas de apostas se comportam ao longo do tempo e se elas refletem com precisão as probabilidades reais dos eventos esportivos. A aplicação utiliza dados históricos, momento global das equipes e variáveis externas (como o clima) para precificar jogos futuros e testar estratégias de apostas em um ambiente controlado.

---

## 🏗️ Arquitetura e Pipeline de Dados

O projeto é dividido em quatro pilares principais:

### 1. Coleta de Dados (Web Scraping)
* **Extração Automatizada:** Utilização de `Selenium` e `BeautifulSoup` para raspar dados históricos e calendários futuros do Flashscore.
* **Métricas Coletadas:** Placar, odds (Casa, Empate, Fora), xG (Gols Esperados), posse de bola, finalizações (totais e no alvo), escanteios, cartões, e muito mais.
* **Checkpointing:** Sistema inteligente que identifica partidas já cadastradas para evitar requisições redundantes.

### 2. Contexto Climático (API Integration)
* **Open-Meteo API:** Mapeamento dinâmico das cidades/estádios para buscar o clima exato (Temperatura, Umidade e Velocidade do Vento) na hora exata em que a bola rolou.
* **Resiliência:** Tratamento de erros, pausas automáticas (rate limiting) e conversão de coordenadas geográficas.

### 3. Machine Learning & Backtesting
* **Engenharia de Features:** Criação de diferenciais táticos (Saldo de Ataque, Defesa e Volume) e médias móveis (janelas de 3 e 5 jogos) para capturar o "momento" das equipes.
* **Modelos Testados:** Logistic Regression, Random Forest, XGBoost, LightGBM e Gradient Boosting.
* **Walk-Forward Validation:** Simulação rigorosa e realista que treina o modelo de forma iterativa ao longo do tempo (evitando *data leakage*).
* **Otimização de EV:** Busca automática pelo melhor limiar de Valor Esperado na base de validação antes de testar na base de teste.

### 4. Aplicação Web (Dashboard)
* **Streamlit:** Interface de usuário interativa e performática.
* **Recursos Principais:**
  * Painel de estatísticas e radares táticos comparativos (Casa vs Fora).
  * Previsões de jogos futuros baseadas no modelo salvo.
  * Simulador de Apostas (Estresse de Variância) utilizando métodos de Monte Carlo.
  * Análise de Eficiência de Odds comparando Probabilidade Implícita vs Probabilidade Real.
  * *Acessibilidade:* Modo daltônico integrado para gráficos e alertas.

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python
* **Web Scraping:** `Selenium`, `BeautifulSoup`, `webdriver-manager`
* **Processamento de Dados:** `Pandas`, `NumPy`, `SciPy`
* **Machine Learning:** `Scikit-Learn`, `XGBoost`, `LightGBM`
* **Visualização:** `Plotly`, `Matplotlib`, `Seaborn`
* **Deploy/Frontend:** `Streamlit`
* **Persistência:** `joblib` (Pipelines ML), Arquivos CSV/Excel

---

## 📂 Estrutura do Repositório

* `Api clima.ipynb`: Script de automação para extração de dados das partidas.
* `Exportando Cidades.ipynb`: Integração com a API de clima e cruzamento com a base de jogos.
* `Modelos Br e PL.ipnyb`: Pipeline de treino, validação e teste do modelo da Série A e da liga inglesa
* `app.py`: Código-fonte da aplicação interativa do Streamlit.
* `*.pkl`: Modelos treinados salvos com seus respectivos hiperparâmetros campeões.

---

