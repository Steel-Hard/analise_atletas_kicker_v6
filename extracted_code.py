# ─────────────────────────────────────────────────────────
# IMPORTAÇÃO DAS BIBLIOTECAS
# ─────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from scipy import stats
from google.colab import files

# Configurações de visualização
plt.rcParams['figure.figsize'] = (14, 5)
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_theme(style='whitegrid', palette='muted')

print("✅ Todas as bibliotecas importadas com sucesso!")
print(f"   pandas  {pd.__version__} | numpy {np.__version__}")


# ─────────────────────────────────────────────────────────
# IMPORTAÇÃO DOS DADOS
# ─────────────────────────────────────────────────────────

uploaded = files.upload()  # Abre o seletor de arquivo
ARQUIVO = list(uploaded.keys())[0]  # Pega o nome do arquivo enviado

df_raw = pd.read_excel(ARQUIVO)

print(f"📊 Dimensões brutas: {df_raw.shape[0]:,} linhas × {df_raw.shape[1]} colunas")
print(f"📅 Período dos dados: {df_raw['Start Date'].min().date()} → {df_raw['Start Date'].max().date()}")
print(f"👤 Atletas únicos: {df_raw['Athlete ID'].nunique()}")
print(f"📂 Segmentos disponíveis: {list(df_raw['Segment Name'].unique())}")


from google.colab import drive
drive.mount('/content/drive')

# ─────────────────────────────────────────────────────────
# VISUALIZAÇÃO INICIAL
# ─────────────────────────────────────────────────────────
print("\n🔍 Primeiras 3 linhas do dataset bruto:")
df_raw.head(3)


# ─────────────────────────────────────────────────────────
# FILTRO: APENAS "Whole Session"
# ─────────────────────────────────────────────────────────
# Cada evento (treino/partida) gera múltiplas linhas (First Half, Second Half, etc.)
# Para evitar dupla contagem e garantir comparabilidade, usamos apenas a sessão completa.

df_ws = df_raw[df_raw['Segment Name'] == 'Whole Session'].copy()
df_ws = df_ws.reset_index(drop=True)

print(f"✅ Após filtro 'Whole Session': {df_ws.shape[0]:,} registros de {df_ws['Athlete ID'].nunique()} atletas")

# Distribuição de sessões por atleta
dist = df_ws.groupby('Athlete ID').size().rename('Sessões')
print("\n📋 Sessões por atleta (Top 10):")
print(dist.sort_values(ascending=False).head(10).to_string())


# ─────────────────────────────────────────────────────────
# PASSO 3.1 — INSPEÇÃO DE AUSENTES
# ─────────────────────────────────────────────────────────

ausentes = df_ws.isnull().sum()
pct_ausentes = (ausentes / len(df_ws) * 100).round(2)
resumo_ausentes = pd.DataFrame({
    'Ausentes': ausentes,
    '% Ausentes': pct_ausentes
}).query('Ausentes > 0').sort_values('% Ausentes', ascending=False)

print("🔎 Colunas com valores ausentes:")
print(resumo_ausentes.to_string())

# Visualização
fig, ax = plt.subplots(figsize=(14, 4))
resumo_ausentes['% Ausentes'].plot(kind='bar', ax=ax, color='salmon', edgecolor='black')
ax.set_title('Percentual de Valores Ausentes por Coluna', fontsize=13, fontweight='bold')
ax.set_ylabel('% Ausentes')
ax.set_xlabel('')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────
# PASSO 3.2 — VERIFICAÇÃO DE DUPLICATAS
# ─────────────────────────────────────────────────────────
# Identificador único de uma sessão = Atleta + Data + Segmento

chave_dedup = ['Athlete ID', 'Start Date', 'Segment Name']
n_duplicatas = df_ws.duplicated(subset=chave_dedup).sum()
print(f"♻️  Duplicatas encontradas: {n_duplicatas}")

if n_duplicatas > 0:
    df_ws = df_ws.drop_duplicates(subset=chave_dedup, keep='first')
    print(f"   → {n_duplicatas} registros duplicados removidos.")
else:
    print("   → Nenhuma duplicata encontrada. ✅")


# ─────────────────────────────────────────────────────────
# PASSO 3.3 — VERIFICAÇÃO DE ZEROS FISICAMENTE IMPLAUSÍVEIS
# ─────────────────────────────────────────────────────────
# Algumas colunas não devem ser zero em uma sessão completa registrada.
# Ex.: distância = 0 em uma sessão de 90min é suspeito (sensor falhou?).

cols_nao_zero = ['Distance (m)', 'Duration (mins)']
for col in cols_nao_zero:
    n_zeros = (df_ws[col] == 0).sum()
    if n_zeros > 0:
        print(f"⚠️  '{col}': {n_zeros} registros com valor zero → serão convertidos para NaN")
        df_ws[col] = df_ws[col].replace(0, np.nan)
    else:
        print(f"✅ '{col}': sem zeros problemáticos")


# ─────────────────────────────────────────────────────────
# PASSO 3.4 — VERIFICAÇÃO DE INCONSISTÊNCIAS LÓGICAS
# ─────────────────────────────────────────────────────────
# Regra de negócio: Sprint Distance ≤ Distance (m)
# Um atleta não pode percorrer mais em sprint do que a distância total.

inconsist = df_ws[df_ws['Sprint Distance (m)'] > df_ws['Distance (m)']].shape[0]
print(f"🔎 Inconsistências (Sprint > Distância Total): {inconsist} registros")

# Regra: Top Speed ≥ Avg Speed
inconsist2 = df_ws[df_ws['Top Speed (kph)'] < df_ws['Avg Speed (kph)']].shape[0]
print(f"🔎 Inconsistências (Top Speed < Avg Speed): {inconsist2} registros")


# ─────────────────────────────────────────────────────────
# PASSO 3.5 — IMPUTAÇÃO POR MEDIANA DO ATLETA
# ─────────────────────────────────────────────────────────
# Para variáveis numéricas com ausentes, imputamos pela mediana histórica
# de cada atleta individualmente. Isso preserva o perfil individual
# e evita que dados de outros atletas contaminem a série do jogador.

COLUNAS_FISICAS = [
    'Duration (mins)', 'Workload', 'Distance (m)',
    'Metres per Minute (m)', 'High Intensity Running (m)',
    'Sprint Distance (m)', 'Raw Top Speed (kph)', 'No. of Sprints',
    'Top Speed (kph)', 'Avg Speed (kph)',
    'Accelerations', 'Decelerations',
    'No. of High Intensity Events', 'Session Load'
]

# Colunas efetivamente disponíveis
COLUNAS_FISICAS = [c for c in COLUNAS_FISICAS if c in df_ws.columns]

antes = df_ws[COLUNAS_FISICAS].isnull().sum().sum()

df_ws[COLUNAS_FISICAS] = df_ws.groupby('Athlete ID')[COLUNAS_FISICAS].transform(
    lambda x: x.fillna(x.median())
)

# Se ainda houver NaN (atleta com 100% ausente na coluna), usa mediana global
df_ws[COLUNAS_FISICAS] = df_ws[COLUNAS_FISICAS].fillna(df_ws[COLUNAS_FISICAS].median())

depois = df_ws[COLUNAS_FISICAS].isnull().sum().sum()
print(f"🔧 Valores ausentes nas colunas físicas: {antes} → {depois}")
print(f"✅ Dataset limpo: {df_ws.shape[0]} registros prontos para análise")


# ─────────────────────────────────────────────────────────
# SELEÇÃO FINAL DAS VARIÁVEIS DO MODELO
# ─────────────────────────────────────────────────────────

FEATURES = [
    'Workload',
    'Distance (m)',
    'Metres per Minute (m)',
    'High Intensity Running (m)',
    'Sprint Distance (m)',
    'Top Speed (kph)',
    'Avg Speed (kph)',
    'Accelerations',
    'Decelerations',
    'No. of Sprints',
    'Duration (mins)'
]

# Verificar disponibilidade
FEATURES = [f for f in FEATURES if f in df_ws.columns]
print(f"✅ Variáveis selecionadas ({len(FEATURES)}):")
for i, f in enumerate(FEATURES, 1):
    print(f"   {i:2d}. {f}")


# ─────────────────────────────────────────────────────────
# ANÁLISE DE CORRELAÇÃO ENTRE AS VARIÁVEIS
# ─────────────────────────────────────────────────────────
# O heatmap de correlação nos ajuda a entender redundâncias.
# Variáveis com correlação > 0.95 poderiam ser removidas para evitar
# multicolinearidade, mas no Isolation Forest isso é menos crítico
# pois o algoritmo não assume distribuição específica nem calcula coeficientes.

corr_matrix = df_ws[FEATURES].corr()

fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt='.2f',
    cmap='RdYlGn', center=0, vmin=-1, vmax=1,
    linewidths=0.5, ax=ax, cbar_kws={'label': 'Correlação de Pearson'}
)
ax.set_title('Matriz de Correlação — Variáveis de Desempenho', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n💡 Interpretação: Valores próximos de 1.0 indicam alta correlação positiva (as variáveis")
print("   crescem juntas). Ex.: Distance e Duration correlacionadas sugere que sessões mais longas")
print("   naturalmente acumulam mais distância — faz sentido esportivamente.")


# ─────────────────────────────────────────────────────────
# PADRONIZAÇÃO GLOBAL (para comparação entre atletas)
# ─────────────────────────────────────────────────────────
# Fitamos o scaler em TODOS os dados (todos os atletas juntos).
# Isso cria uma referência de escala global, útil para comparação entre jogadores.

scaler_global = StandardScaler()
X_global = df_ws[FEATURES].values
X_scaled_global = scaler_global.fit_transform(X_global)

df_scaled = df_ws[['Athlete ID', 'Start Date', 'Segment Name']].copy()
for i, feat in enumerate(FEATURES):
    df_scaled[feat + '_z'] = X_scaled_global[:, i]

print("📐 Estatísticas PÓS-padronização (devem ser média≈0, std≈1):")
stats_pos = pd.DataFrame(X_scaled_global, columns=FEATURES).describe().loc[['mean','std']]
print(stats_pos.round(4).to_string())


# ─────────────────────────────────────────────────────────
# VISUALIZAÇÃO: ANTES E DEPOIS DA PADRONIZAÇÃO
# ─────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, len(FEATURES), figsize=(20, 6))

for i, feat in enumerate(FEATURES):
    # Antes
    axes[0, i].hist(df_ws[feat].dropna(), bins=30, color='steelblue', alpha=0.7, edgecolor='none')
    axes[0, i].set_title(feat.split('(')[0].strip()[:18], fontsize=7, fontweight='bold')
    axes[0, i].tick_params(labelsize=6)

    # Depois
    axes[1, i].hist(X_scaled_global[:, i], bins=30, color='darkorange', alpha=0.7, edgecolor='none')
    axes[1, i].tick_params(labelsize=6)

axes[0, 0].set_ylabel('Antes (escala original)', fontsize=9, color='steelblue')
axes[1, 0].set_ylabel('Depois (padronizado)', fontsize=9, color='darkorange')

fig.suptitle('Distribuição das Variáveis — Antes e Depois do StandardScaler', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n💡 Observe que o formato das distribuições é idêntico antes e depois.")
print("   O StandardScaler apenas reposiciona a média em 0 e escala o desvio para 1.")


# ─────────────────────────────────────────────────────────
# ESTATÍSTICAS DESCRITIVAS POR ATLETA
# ─────────────────────────────────────────────────────────

def estatisticas_atleta(df, features):
    """Calcula estatísticas descritivas para cada atleta."""
    resultados = []
    for athlete_id, grupo in df.groupby('Athlete ID'):
        for feat in features:
            serie = grupo[feat].dropna()
            if len(serie) < 2:
                continue
            resultados.append({
                'Athlete ID': athlete_id,
                'Variável': feat,
                'N Sessões': len(serie),
                'Média': serie.mean(),
                'Mediana': serie.median(),
                'Desvio Padrão': serie.std(ddof=1),   # ddof=1 = desvio amostral
                'Variância': serie.var(ddof=1),
                'Mínimo': serie.min(),
                'Máximo': serie.max(),
                'CV (%)': (serie.std(ddof=1) / serie.mean() * 100) if serie.mean() != 0 else np.nan
            })
    return pd.DataFrame(resultados)

df_stats = estatisticas_atleta(df_ws, FEATURES)

print(f"✅ Estatísticas calculadas: {df_stats.shape[0]} combinações atleta × variável")
print("\n📋 Exemplo — Athlete ID:", df_ws['Athlete ID'].unique()[0])
exemplo = df_stats[df_stats['Athlete ID'] == df_ws['Athlete ID'].unique()[0]]
print(exemplo[['Variável','N Sessões','Média','Desvio Padrão','CV (%)']].round(2).to_string(index=False))


# ─────────────────────────────────────────────────────────
# VISUALIZAÇÃO: BOXPLOT DE DISTÂNCIA POR ATLETA
# ─────────────────────────────────────────────────────────
# O boxplot mostra a distribuição completa (mediana, IQR, outliers)
# de uma variável para cada atleta — excelente para comparação visual.

fig, ax = plt.subplots(figsize=(16, 6))
order = df_ws.groupby('Athlete ID')['Distance (m)'].median().sort_values(ascending=False).index
sns.boxplot(
    data=df_ws, x='Athlete ID', y='Distance (m)',
    order=order, ax=ax, palette='Set2',
    flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.5}
)
ax.set_title('Distribuição de Distância Percorrida por Atleta (Whole Session)', fontsize=13, fontweight='bold')
ax.set_xlabel('Athlete ID')
ax.set_ylabel('Distância (m)')
plt.xticks(rotation=45, ha='right', fontsize=7)
plt.tight_layout()
plt.show()

print("💡 Cada caixa representa o intervalo interquartil (IQR = Q3 - Q1) do atleta.")
print("   A linha horizontal dentro da caixa é a mediana.")
print("   Os pontos fora dos 'whiskers' são outliers potenciais — candidatos a anomalias.")


# ─────────────────────────────────────────────────────────
# ISOLATION FOREST — APLICAÇÃO INDIVIDUAL POR ATLETA
# ─────────────────────────────────────────────────────────

def aplicar_isolation_forest(df, features, n_estimators=200, contamination=0.10, random_state=42):
    """
    Aplica Isolation Forest individualmente para cada atleta.

    Retorna DataFrame com colunas adicionais:
    - 'IF_Score': score de anomalia (-1 a 0, quanto mais negativo mais anômalo)
    - 'IF_Label': 1 = normal, -1 = anomalia (saída raw do sklearn)
    - 'IF_Anomalia': True se a sessão foi classificada como anomalia
    """
    resultados = []

    for athlete_id, grupo in df.groupby('Athlete ID'):
        grupo_c = grupo.copy()
        X = grupo_c[features].values
        n = len(X)

        if n < 5:
            # Poucos dados: não é possível treinar modelo confiável
            grupo_c['IF_Score'] = np.nan
            grupo_c['IF_Label'] = np.nan
            grupo_c['IF_Anomalia'] = False
            resultados.append(grupo_c)
            continue

        # Padronização LOCAL (por atleta) para o Isolation Forest
        # Importante: o scaler é fitado apenas nos dados do atleta,
        # garantindo que a detecção de anomalia seja relativa ao histórico dele.
        scaler_local = StandardScaler()
        X_scaled = scaler_local.fit_transform(X)

        # Treino e predição do modelo
        model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples='auto',
            random_state=random_state
        )
        model.fit(X_scaled)

        # score_samples retorna -score_anomalia (quanto mais negativo = mais anômalo)
        grupo_c['IF_Score'] = model.score_samples(X_scaled)
        grupo_c['IF_Label'] = model.predict(X_scaled)  # 1 = normal, -1 = anomalia
        grupo_c['IF_Anomalia'] = grupo_c['IF_Label'] == -1

        resultados.append(grupo_c)

    return pd.concat(resultados, ignore_index=True)

# Aplicar o modelo
df_if = aplicar_isolation_forest(df_ws, FEATURES)

# Sumário dos resultados
n_anomalias = df_if['IF_Anomalia'].sum()
pct_anomalias = n_anomalias / len(df_if) * 100
print(f"🔴 Sessões classificadas como anomalia: {n_anomalias} de {len(df_if)} ({pct_anomalias:.1f}%)")
print(f"🟢 Sessões normais: {(~df_if['IF_Anomalia']).sum()} ({100 - pct_anomalias:.1f}%)")

# Por atleta
por_atleta = df_if.groupby('Athlete ID')['IF_Anomalia'].agg(['sum','count'])
por_atleta.columns = ['Anomalias', 'Total']
por_atleta['% Anomalias'] = (por_atleta['Anomalias'] / por_atleta['Total'] * 100).round(1)
print("\n📊 Anomalias detectadas por atleta:")
print(por_atleta.sort_values('% Anomalias', ascending=False).head(10).to_string())


# ─────────────────────────────────────────────────────────
# VISUALIZAÇÃO: IF Score ao Longo do Tempo (Atleta Exemplo)
# ─────────────────────────────────────────────────────────

# Pegar o atleta com mais sessões para exemplo rico
atleta_exemplo = df_if.groupby('Athlete ID').size().idxmax()
df_atleta = df_if[df_if['Athlete ID'] == atleta_exemplo].sort_values('Start Date')

fig, ax = plt.subplots(figsize=(16, 5))

# Plot do score ao longo do tempo
ax.plot(df_atleta['Start Date'], df_atleta['IF_Score'],
        color='steelblue', linewidth=1.5, alpha=0.8, label='IF Score')
ax.scatter(df_atleta['Start Date'], df_atleta['IF_Score'],
           c=df_atleta['IF_Anomalia'].map({True: 'red', False: 'steelblue'}),
           s=60, zorder=5)

# Linha de threshold
threshold = df_atleta[df_atleta['IF_Label'] == -1]['IF_Score'].max() if df_atleta['IF_Anomalia'].any() else None
if threshold:
    ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.6, label=f'Limiar (≈{threshold:.3f})')

ax.set_title(f'Isolation Forest Score ao Longo do Tempo — Atleta {atleta_exemplo}',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Data')
ax.set_ylabel('IF Score (mais negativo = mais anômalo)')

# Legenda manual
normal_patch = mpatches.Patch(color='steelblue', label='Sessão Normal')
anom_patch = mpatches.Patch(color='red', label='Sessão Anômala')
ax.legend(handles=[normal_patch, anom_patch])

plt.tight_layout()
plt.show()

print(f"\n📋 Detalhes do atleta {atleta_exemplo}:")
print(f"   Total de sessões: {len(df_atleta)}")
print(f"   Anomalias detectadas: {df_atleta['IF_Anomalia'].sum()}")
print(f"   Período analisado: {df_atleta['Start Date'].min().date()} → {df_atleta['Start Date'].max().date()}")


# ─────────────────────────────────────────────────────────
# PASSO 8.1 — DEMONSTRAÇÃO MANUAL DO Z-SCORE (1 atleta, 1 variável)
# ─────────────────────────────────────────────────────────

# Selecionamos o atleta exemplo e a variável Distance (m)
serie = df_if[df_if['Athlete ID'] == atleta_exemplo]['Distance (m)'].dropna()

mu = serie.mean()
sigma = serie.std(ddof=1)  # ddof=1 = desvio amostral (não populacional)

print(f"📐 DEMONSTRAÇÃO MATEMÁTICA — Atleta {atleta_exemplo} | Variável: Distance (m)")
print(f"{'='*60}")
print(f"   N sessões históricasː {len(serie)}")
print(f"   Média (μ)           : {mu:.2f} m")
print(f"   Desvio Padrão (σ)   : {sigma:.2f} m")
print(f"{'='*60}")
print(f"\n   Fórmula: Z = (x - μ) / σ  =  (x - {mu:.1f}) / {sigma:.1f}")
print(f"\n   Exemplos de cálculo passo a passo:")
print(f"   {'Sessão':>8} | {'x (m)':>10} | {'(x - μ)':>10} | {'/ σ':>10} | {'Z-Score':>10} | Classificação")
print(f"   {'-'*70}")

# Mostrar 5 exemplos representativos
indices = serie.index[:5]
for idx in indices:
    x = serie[idx]
    diff = x - mu
    z = diff / sigma
    classe = "Alta ⬆️" if z > 1.5 else ("Queda ⬇️" if z < -1.5 else "Médio ➡️")
    print(f"   {str(idx):>8} | {x:>10.1f} | {diff:>+10.1f} | {diff/sigma:>+10.4f} | {z:>+10.4f} | {classe}")


# ─────────────────────────────────────────────────────────
# PASSO 8.2 — CÁLCULO DE Z-SCORE PARA TODOS OS ATLETAS E VARIÁVEIS
# ─────────────────────────────────────────────────────────

def calcular_zscore_por_atleta(df, features):
    """
    Calcula Z-score de cada observação em relação à média e desvio padrão
    históricos do PRÓPRIO atleta (não da população global).

    Isso é fundamental: estamos perguntando 'essa sessão é anormal para ESTE atleta?',
    não 'essa sessão é anormal para um atleta genérico?'
    """
    resultados = []

    for athlete_id, grupo in df.groupby('Athlete ID'):
        grupo_c = grupo.copy()
        for feat in features:
            serie = grupo_c[feat]
            mu_atleta = serie.mean()
            sigma_atleta = serie.std(ddof=1)

            if sigma_atleta == 0 or np.isnan(sigma_atleta):
                # Sem variação: Z-score indefinido, atribuímos 0
                grupo_c[f'Z_{feat}'] = 0.0
            else:
                grupo_c[f'Z_{feat}'] = (serie - mu_atleta) / sigma_atleta

        resultados.append(grupo_c)

    return pd.concat(resultados, ignore_index=True)

df_zscores = calcular_zscore_por_atleta(df_if, FEATURES)

# Verificação: Z-scores devem ter média ≈ 0 e std ≈ 1 por atleta
z_cols = [f'Z_{f}' for f in FEATURES if f'Z_{f}' in df_zscores.columns]
print(f"✅ Z-Scores calculados para {len(z_cols)} variáveis")
print(f"\n📊 Estatísticas globais dos Z-Scores (devem ser ≈ 0 e ≈ 1):")
print(df_zscores[z_cols].describe().loc[['mean','std']].round(4).to_string())


# ─────────────────────────────────────────────────────────
# PASSO 8.3 — Z-SCORE COMPOSTO (média dos Z-scores das variáveis-chave)
# ─────────────────────────────────────────────────────────
# Um Z-score composto resume o desempenho geral de uma sessão num único número.
# Usamos as variáveis de volume e intensidade mais representativas.

FEATURES_COMPOSTAS = [f for f in [
    'Distance (m)', 'Workload', 'High Intensity Running (m)',
    'Sprint Distance (m)', 'Accelerations', 'Decelerations'
] if f in FEATURES]

z_compostas = [f'Z_{f}' for f in FEATURES_COMPOSTAS]
df_zscores['Z_Composto'] = df_zscores[z_compostas].mean(axis=1)

print(f"📐 Z-Score Composto calculado como média de:")
for f in FEATURES_COMPOSTAS:
    print(f"   • Z_{f}")

print(f"\n📊 Distribuição do Z-Score Composto:")
print(df_zscores['Z_Composto'].describe().round(4).to_string())

# Histograma
fig, ax = plt.subplots(figsize=(12, 4))
df_zscores['Z_Composto'].hist(bins=50, ax=ax, color='steelblue', edgecolor='none', alpha=0.8)
ax.axvline(x=1.5, color='green', linestyle='--', linewidth=2, label='Z = +1.5 (Alta)')
ax.axvline(x=-1.5, color='red', linestyle='--', linewidth=2, label='Z = -1.5 (Queda)')
ax.axvline(x=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
ax.set_title('Distribuição do Z-Score Composto (todos os atletas)', fontsize=12, fontweight='bold')
ax.set_xlabel('Z-Score Composto')
ax.set_ylabel('Frequência')
ax.legend()
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────
# CLASSIFICAÇÃO COMBINADA
# ─────────────────────────────────────────────────────────

def classificar_sessao(row, z_threshold=1.5):
    """
    Aplica as regras de negócio combinando Isolation Forest e Z-Score Composto.

    Args:
        row: linha do DataFrame com IF_Anomalia e Z_Composto
        z_threshold: limiar de Z para separar alta/queda de desempenho médio

    Returns:
        str: 'Alta de Desempenho', 'Queda de Desempenho' ou 'Desempenho Médio'
    """
    if pd.isna(row['IF_Anomalia']) or pd.isna(row['Z_Composto']):
        return 'Desempenho Médio'

    if row['IF_Anomalia']:
        if row['Z_Composto'] > z_threshold:
            return 'Alta de Desempenho'
        elif row['Z_Composto'] < -z_threshold:
            return 'Queda de Desempenho'
        else:
            return 'Desempenho Médio'
    else:
        return 'Desempenho Médio'

df_zscores['Classificacao'] = df_zscores.apply(classificar_sessao, axis=1)

# Sumário
dist_class = df_zscores['Classificacao'].value_counts()
print("📊 Distribuição das classificações:")
for classe, n in dist_class.items():
    emoji = '⬆️' if 'Alta' in classe else ('⬇️' if 'Queda' in classe else '➡️')
    print(f"   {emoji} {classe}: {n} sessões ({n/len(df_zscores)*100:.1f}%)")


# ─────────────────────────────────────────────────────────
# VISUALIZAÇÃO: MAPA DE CLASSIFICAÇÕES POR ATLETA E TEMPO
# ─────────────────────────────────────────────────────────

# Agregar por atleta: contar cada categoria
cores_map = {
    'Alta de Desempenho': '#2ECC71',
    'Desempenho Médio': '#3498DB',
    'Queda de Desempenho': '#E74C3C'
}

fig, ax = plt.subplots(figsize=(16, 6))
for classe, cor in cores_map.items():
    subset = df_zscores[df_zscores['Classificacao'] == classe]
    ax.scatter(subset['Start Date'], subset['Athlete ID'].astype(str),
               c=cor, alpha=0.6, s=40, label=classe, zorder=3)

ax.set_title('Mapa de Classificações de Desempenho por Atleta ao Longo do Tempo',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Data')
ax.set_ylabel('Athlete ID')
plt.yticks(fontsize=7)
ax.legend(loc='upper left', framealpha=0.9)
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────
# RELATÓRIO DETALHADO — SESSÕES ANÔMALAS
# ─────────────────────────────────────────────────────────

print("🔍 SESSÕES CLASSIFICADAS COMO 'QUEDA DE DESEMPENHO':")
print("="*80)

quedas = df_zscores[df_zscores['Classificacao'] == 'Queda de Desempenho'].sort_values('Z_Composto')
colunas_rel = ['Athlete ID', 'Start Date', 'Z_Composto', 'IF_Score'] + z_compostas[:4]
colunas_rel = [c for c in colunas_rel if c in quedas.columns]

if len(quedas) > 0:
    print(quedas[colunas_rel].head(10).round(3).to_string(index=False))
else:
    print("Nenhuma queda de desempenho detectada com os thresholds atuais.")

print()
print("🔍 SESSÕES CLASSIFICADAS COMO 'ALTA DE DESEMPENHO':")
print("="*80)

altas = df_zscores[df_zscores['Classificacao'] == 'Alta de Desempenho'].sort_values('Z_Composto', ascending=False)
if len(altas) > 0:
    print(altas[colunas_rel].head(10).round(3).to_string(index=False))
else:
    print("Nenhuma alta de desempenho detectada com os thresholds atuais.")


# ─────────────────────────────────────────────────────────
# PASSO 10.1 — CONSTRUÇÃO DOS VETORES DE PERFIL
# ─────────────────────────────────────────────────────────
# Cada atleta é representado pela média de seus Z-Scores históricos.
# Isso captura o padrão relativo de desempenho de cada jogador.

perfis = df_zscores.groupby('Athlete ID')[z_cols].mean()

print(f"✅ Perfis construídos para {len(perfis)} atletas")
print(f"   Dimensão do vetor de perfil: {len(z_cols)} variáveis")
print()
print("📊 Primeiros 3 atletas — vetores de perfil (Z-Score médio por variável):")
print(perfis.head(3).round(3).to_string())


# ─────────────────────────────────────────────────────────
# PASSO 10.2 — MATRIZES DE SIMILARIDADE
# ─────────────────────────────────────────────────────────

X_perfis = perfis.values
ids_atletas = perfis.index.tolist()

# Distância Euclidiana
dist_eucl = euclidean_distances(X_perfis)
df_eucl = pd.DataFrame(dist_eucl, index=ids_atletas, columns=ids_atletas)

# Distância de Cosseno
dist_cos = cosine_distances(X_perfis)
df_cos = pd.DataFrame(dist_cos, index=ids_atletas, columns=ids_atletas)

print("📐 Matrizes de distância calculadas:")
print(f"   Euclidiana: {df_eucl.shape} — valores em [{df_eucl.values[df_eucl.values > 0].min():.2f}, {df_eucl.values.max():.2f}]")
print(f"   Cosseno:    {df_cos.shape} — valores em [0, {df_cos.values.max():.4f}]")

# Heatmap de similaridade (cosseno)
sim_cos = 1 - dist_cos  # Converter distância → similaridade
df_sim = pd.DataFrame(sim_cos, index=ids_atletas, columns=ids_atletas)

fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(df_sim, annot=False, cmap='YlOrRd', ax=ax,
            xticklabels=[str(a)[-6:] for a in ids_atletas],
            yticklabels=[str(a)[-6:] for a in ids_atletas])
ax.set_title('Mapa de Similaridade entre Atletas (Cosseno) — 1 = Perfis Idênticos',
             fontsize=12, fontweight='bold')
ax.tick_params(axis='x', rotation=45, labelsize=7)
ax.tick_params(axis='y', rotation=0, labelsize=7)
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────
# PASSO 10.3 — FUNÇÃO DE SUGESTÃO DE SUBSTITUTOS
# ─────────────────────────────────────────────────────────

def sugerir_substitutos(atleta_referencia, perfis, dist_eucl_df, dist_cos_df, top_n=3):
    """
    Dado um atleta de referência, retorna os N atletas mais similares.

    Args:
        atleta_referencia: ID do atleta que precisa de substituto
        perfis: DataFrame com vetores de perfil
        dist_eucl_df: DataFrame com distâncias euclidianas
        dist_cos_df: DataFrame com distâncias de cosseno
        top_n: número de substitutos a sugerir

    Returns:
        DataFrame com os substitutos e suas pontuações de similaridade
    """
    if atleta_referencia not in perfis.index:
        print(f"❌ Atleta {atleta_referencia} não encontrado nos perfis.")
        return None

    outros = [a for a in perfis.index if a != atleta_referencia]

    resultados = []
    for outro in outros:
        d_eucl = dist_eucl_df.loc[atleta_referencia, outro]
        d_cos = dist_cos_df.loc[atleta_referencia, outro]

        # Score combinado: penaliza tanto distância euclidiana quanto angular
        sim_eucl = 1 / (1 + d_eucl)   # 0 a 1, maior = mais similar
        sim_cos = max(0, 1 - d_cos)    # 0 a 1, maior = mais similar
        score = (sim_eucl * 0.4 + sim_cos * 0.6)  # Ponderado: cosseno tem mais peso

        resultados.append({
            'Atleta Candidato': outro,
            'Distância Euclidiana': d_eucl,
            'Distância Cosseno': d_cos,
            'Similaridade (0-1)': score
        })

    df_res = pd.DataFrame(resultados).sort_values('Similaridade (0-1)', ascending=False)
    return df_res.head(top_n)

# Demonstração: atleta com mais sessões como referência
atleta_ref = df_zscores.groupby('Athlete ID').size().idxmax()

print(f"🔍 Sugestão de substitutos para o Atleta {atleta_ref}")
print("="*70)
substitutos = sugerir_substitutos(atleta_ref, perfis, df_eucl, df_cos, top_n=5)
if substitutos is not None:
    print(substitutos.round(4).to_string(index=False))
print()
print("💡 O candidato com maior 'Similaridade (0-1)' é o melhor substituto")
print("   funcional — aquele cujo padrão de desempenho mais se aproxima do")
print("   atleta de referência em termos de volume, intensidade e padrão físico.")


# ─────────────────────────────────────────────────────────
# PASSO 10.4 — VISUALIZAÇÃO RADAR: COMPARAÇÃO DE PERFIS
# ─────────────────────────────────────────────────────────

import matplotlib.patches as mpatches

def radar_chart(atletas, perfis_df, features_radar, titulo='Comparação de Perfis'):
    """Gera gráfico radar comparando perfis de múltiplos atletas."""

    n = len(features_radar)
    angulos = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angulos += angulos[:1]  # Fechar o polígono

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    cores = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12']

    for i, atleta in enumerate(atletas):
        if atleta not in perfis_df.index:
            continue
        valores = perfis_df.loc[atleta, [f'Z_{f}' for f in features_radar if f'Z_{f}' in perfis_df.columns]].values.tolist()
        valores += valores[:1]

        ax.plot(angulos, valores, 'o-', linewidth=2, color=cores[i % len(cores)], label=f'Atleta {str(atleta)[-6:]}')
        ax.fill(angulos, valores, alpha=0.15, color=cores[i % len(cores)])

    labels_radar = [f.replace(' (m)', '').replace(' (kph)', '').replace('No. of ', '') for f in features_radar]
    ax.set_xticks(angulos[:-1])
    ax.set_xticklabels(labels_radar, size=9)
    ax.set_title(titulo, size=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    plt.tight_layout()
    plt.show()

# Features para o radar
features_radar = [f for f in ['Distance (m)', 'Sprint Distance (m)', 'Top Speed (kph)',
                               'Accelerations', 'Decelerations', 'High Intensity Running (m)'] if f in FEATURES]

# Comparar atleta referência com seu melhor substituto
if substitutos is not None:
    atleta_sub = substitutos.iloc[0]['Atleta Candidato']
    radar_chart(
        [atleta_ref, atleta_sub],
        perfis, features_radar,
        titulo=f'Comparação de Perfis: Referência vs. Melhor Substituto'
    )


# ─────────────────────────────────────────────────────────
# EXPORTAÇÃO DO DATASET FINAL COM TODAS AS ANÁLISES
# ─────────────────────────────────────────────────────────

colunas_export = (
    ['Athlete ID', 'Athlete Position', 'Start Date', 'Duration (mins)']
    + FEATURES
    + ['IF_Score', 'IF_Anomalia', 'Z_Composto', 'Classificacao']
    + z_cols[:6]  # Primeiros 6 Z-Scores individuais
)
colunas_export = [c for c in colunas_export if c in df_zscores.columns]

df_export = df_zscores[colunas_export].copy()
df_export.to_excel('resultados_pipeline.xlsx', index=False)

print("💾 Dataset com resultados exportado: 'resultados_pipeline.xlsx'")
print(f"   Linhas: {len(df_export):,} | Colunas: {len(df_export.columns)}")
print()

# Resumo final
print("="*60)
print("   RESUMO FINAL DO PIPELINE")
print("="*60)
print(f"   Atletas analisados       : {df_zscores['Athlete ID'].nunique()}")
print(f"   Sessões processadas      : {len(df_zscores):,}")
print(f"   Período                  : {df_zscores['Start Date'].min().date()} → {df_zscores['Start Date'].max().date()}")
print(f"   Variáveis no modelo      : {len(FEATURES)}")
print()
for classe in ['Alta de Desempenho', 'Desempenho Médio', 'Queda de Desempenho']:
    n = (df_zscores['Classificacao'] == classe).sum()
    emoji = '⬆️' if 'Alta' in classe else ('⬇️' if 'Queda' in classe else '➡️')
    print(f"   {emoji} {classe:25s}: {n:4d} sessões ({n/len(df_zscores)*100:.1f}%)")
print("="*60)
print("\n✅ Pipeline completo executado com sucesso!")
