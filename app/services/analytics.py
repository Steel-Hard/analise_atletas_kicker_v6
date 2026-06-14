import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

FEATURES_COMPOSTAS = [
    'Distance (m)', 'Workload', 'High Intensity Running (m)',
    'Sprint Distance (m)', 'Accelerations', 'Decelerations'
]

def calcular_zscore_por_atleta(df: pd.DataFrame, features: list) -> pd.DataFrame:
    resultados = []
    for athlete_id, grupo in df.groupby('Athlete ID'):
        grupo_c = grupo.copy()
        for feat in features:
            serie = grupo_c[feat]
            mu_atleta = serie.mean()
            sigma_atleta = serie.std(ddof=1)
            
            if sigma_atleta == 0 or pd.isna(sigma_atleta):
                grupo_c[f'Z_{feat}'] = 0.0
            else:
                grupo_c[f'Z_{feat}'] = (serie - mu_atleta) / sigma_atleta
                
        resultados.append(grupo_c)
        
    if not resultados:
        return df.copy()
        
    return pd.concat(resultados, ignore_index=True)

def calcular_z_composto(df: pd.DataFrame, features: list) -> pd.DataFrame:
    fc = [f for f in FEATURES_COMPOSTAS if f in features]
    z_compostas = [f'Z_{f}' for f in fc]
    df['Z_Composto'] = df[z_compostas].mean(axis=1)
    return df

def classificar_sessao(row, z_threshold=1.5):
    """
    Combina Isolation Forest + Z-Score Composto para classificar a sessão.

    O Isolation Forest sozinho só responde "esta sessão é estatisticamente
    incomum para este atleta?" (if_anomalia=True/False) — ele NÃO sabe se o
    desvio é positivo (ex: pico de desempenho) ou negativo (ex: queda de
    rendimento/fadiga). Quem dá a DIREÇÃO do desvio é o Z-Score Composto:
    positivo = acima da média do próprio atleta, negativo = abaixo.

    Por isso a regra de negócio é:
      1) Se o IF não marcou a sessão como anômala -> 'Desempenho Médio'
         (mesmo que o Z-score esteja um pouco deslocado, não é incomum
         o suficiente para virar alerta).
      2) Se o IF marcou como anômala, o Z_Composto decide a direção:
         - Z_Composto > +z_threshold -> 'Alta de Desempenho'
         - Z_Composto < -z_threshold -> 'Queda de Desempenho'
         - caso contrário (anomalia "fraca"/indecisa) -> 'Desempenho Médio'
    """
    z_composto = row.get('Z_Composto')
    if_anomalia = row.get('if_anomalia')

    # Sem dados suficientes para decidir
    if z_composto is None or pd.isna(z_composto) or if_anomalia is None or pd.isna(if_anomalia):
        return 'Desempenho Médio'

    # O Isolation Forest é o "gatilho": só falamos de Alta/Queda se a sessão
    # já foi sinalizada como estatisticamente anômala para o atleta.
    if not if_anomalia:
        return 'Desempenho Médio'

    # A partir daqui sabemos que é uma anomalia; o Z-Score dá a direção.
    if z_composto > z_threshold:
        return 'Alta de Desempenho'
    elif z_composto < -z_threshold:
        return 'Queda de Desempenho'
    else:
        return 'Desempenho Médio'

def classificar_df(df: pd.DataFrame) -> pd.DataFrame:
    df['Classificacao'] = df.apply(classificar_sessao, axis=1)
    return df

# Mapeamento entre os campos persistidos em SessaoFisica (snake_case) e os
# nomes de FEATURES usados na planilha/análise original.
CAMPO_PARA_FEATURE = {
    'workload': 'Workload',
    'distance_m': 'Distance (m)',
    'metres_per_minute': 'Metres per Minute (m)',
    'high_intensity_running_m': 'High Intensity Running (m)',
    'sprint_distance_m': 'Sprint Distance (m)',
    'top_speed_kph': 'Top Speed (kph)',
    'avg_speed_kph': 'Avg Speed (kph)',
    'accelerations': 'Accelerations',
    'decelerations': 'Decelerations',
    'no_of_sprints': 'No. of Sprints',
    'duration_mins': 'Duration (mins)',
}

def montar_df_features_brutas(sessoes: list) -> pd.DataFrame:
    """
    Recebe uma lista de dicts (ex: [s.model_dump() for s in sessoes]) e
    retorna um DataFrame com a coluna 'Athlete ID' + as colunas de FEATURES
    (valores BRUTOS, na nomenclatura original da planilha), prontos para
    alimentar `construir_perfis`.
    """
    registros = []
    for s in sessoes:
        rec = {'Athlete ID': s.get('athlete_id')}
        for campo, feature in CAMPO_PARA_FEATURE.items():
            rec[feature] = s.get(campo)
        registros.append(rec)

    if not registros:
        return pd.DataFrame(columns=['Athlete ID'] + list(CAMPO_PARA_FEATURE.values()))

    return pd.DataFrame(registros)

def construir_perfis(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Constrói o vetor de "perfil de desempenho" de cada atleta para fins de
    COMPARAÇÃO ENTRE ATLETAS (radar, sugestão de substitutos).

    IMPORTANTE — por que não usar a média dos Z-scores por atleta:
    O Z-score de `calcular_zscore_por_atleta` é calculado em relação à
    PRÓPRIA média/desvio do atleta. Por construção matemática,
    mean((x - mu_atleta) / sigma_atleta) == 0 para QUALQUER atleta.
    Ou seja, a média dos Z-scores por atleta é sempre (aprox.) o vetor
    zero — todos os perfis ficam idênticos, e qualquer cálculo de
    distância/similaridade entre atletas perde o sentido (essa era a
    causa da "comparação" sempre dar a mesma similaridade para todos os
    candidatos).

    A forma correta de comparar atletas entre si é:
      1) calcular a média dos valores BRUTOS de cada atleta (seu volume e
         intensidade "típicos"); e
      2) padronizar essas médias GLOBALMENTE (entre atletas), e não por
         atleta.

    O resultado é um vetor que mostra, em desvios-padrão, como cada atleta
    se compara à média do elenco em cada métrica — e que de fato varia de
    atleta para atleta.
    """
    features_disponiveis = [f for f in features if f in df.columns]
    medias_atleta = df.groupby('Athlete ID')[features_disponiveis].mean()

    # Padronização GLOBAL (entre atletas), não por atleta
    mu_global = medias_atleta.mean()
    sigma_global = medias_atleta.std(ddof=1)
    sigma_global = sigma_global.replace(0, 1.0).fillna(1.0)

    perfis = (medias_atleta - mu_global) / sigma_global
    perfis.columns = [f'Z_{c}' for c in perfis.columns]
    return perfis

def sugerir_substitutos(atleta_referencia: str, perfis: pd.DataFrame, top_n: int = 3):
    if atleta_referencia not in perfis.index:
        return []
        
    X_perfis = perfis.values
    ids_atletas = perfis.index.tolist()
    
    dist_eucl = euclidean_distances(X_perfis)
    df_eucl = pd.DataFrame(dist_eucl, index=ids_atletas, columns=ids_atletas)
    
    dist_cos = cosine_distances(X_perfis)
    df_cos = pd.DataFrame(dist_cos, index=ids_atletas, columns=ids_atletas)
    
    outros = [a for a in ids_atletas if a != atleta_referencia]
    resultados = []
    
    for outro in outros:
        d_eucl = df_eucl.loc[atleta_referencia, outro]
        d_cos = df_cos.loc[atleta_referencia, outro]
        
        sim_eucl = 1 / (1 + d_eucl)
        sim_cos = max(0, 1 - d_cos)
        score = (sim_eucl * 0.4 + sim_cos * 0.6)
        
        resultados.append({
            "atleta_candidato": str(outro),
            "distancia_euclidiana": float(d_eucl),
            "distancia_cosseno": float(d_cos),
            "similaridade": float(score)
        })
        
    resultados = sorted(resultados, key=lambda x: x["similaridade"], reverse=True)
    return resultados[:top_n]
