import pandas as pd
from app.services.analytics import (
    classificar_sessao, construir_perfis, sugerir_substitutos,
    montar_df_features_brutas, CAMPO_PARA_FEATURE
)

def test_classificar_sessao():
    # Anomalia e Z-score alto
    row1 = {"if_anomalia": True, "Z_Composto": 1.6}
    assert classificar_sessao(row1) == "Alta de Desempenho"
    
    # Anomalia e Z-score baixo
    row2 = {"if_anomalia": True, "Z_Composto": -1.6}
    assert classificar_sessao(row2) == "Queda de Desempenho"
    
    # Anomalia mas Z-score dentro do limiar
    row3 = {"if_anomalia": True, "Z_Composto": 1.0}
    assert classificar_sessao(row3) == "Desempenho Médio"
    
    # Não anomalia
    row4 = {"if_anomalia": False, "Z_Composto": 2.0}
    assert classificar_sessao(row4) == "Desempenho Médio"
    
    # Faltando dados
    row5 = {"if_anomalia": None, "Z_Composto": None}
    assert classificar_sessao(row5) == "Desempenho Médio"


def test_classificar_sessao_if_eh_o_gatilho():
    """
    O Isolation Forest decide SE a sessão é anômala; o Z-Score decide a
    DIREÇÃO. Sem if_anomalia=True, mesmo um Z-score extremo não deve gerar
    Alta/Queda de Desempenho.
    """
    # Z extremamente negativo, mas IF não considerou anômalo
    assert classificar_sessao({"if_anomalia": False, "Z_Composto": -5.0}) == "Desempenho Médio"
    # Z extremamente positivo, mas IF não considerou anômalo
    assert classificar_sessao({"if_anomalia": False, "Z_Composto": 5.0}) == "Desempenho Médio"
    # IF considerou anômalo e Z confirma a direção negativa
    assert classificar_sessao({"if_anomalia": True, "Z_Composto": -2.0}) == "Queda de Desempenho"


def _df_features_exemplo():
    """
    DataFrame sintético com 3 atletas e perfis claramente distintos:
    - A1: alto volume/intensidade em tudo
    - A2: baixo volume/intensidade em tudo
    - A3: perfil intermediário (próximo da média do grupo)
    Cada atleta tem 2 sessões (valores levemente diferentes entre si),
    simulando dados reais reconstruídos via montar_df_features_brutas.
    """
    registros = [
        {"Athlete ID": "A1", "Distance (m)": 9000, "Sprint Distance (m)": 400, "Top Speed (kph)": 32},
        {"Athlete ID": "A1", "Distance (m)": 9200, "Sprint Distance (m)": 420, "Top Speed (kph)": 33},
        {"Athlete ID": "A2", "Distance (m)": 4000, "Sprint Distance (m)": 100, "Top Speed (kph)": 24},
        {"Athlete ID": "A2", "Distance (m)": 4100, "Sprint Distance (m)": 110, "Top Speed (kph)": 24.5},
        {"Athlete ID": "A3", "Distance (m)": 6500, "Sprint Distance (m)": 250, "Top Speed (kph)": 28},
        {"Athlete ID": "A3", "Distance (m)": 6600, "Sprint Distance (m)": 260, "Top Speed (kph)": 28.5},
    ]
    return pd.DataFrame(registros)


def test_construir_perfis_gera_vetores_diferentes_por_atleta():
    """
    O perfil de cada atleta deve ser obtido pela padronização GLOBAL
    (entre atletas) da média de seus valores brutos — e não pela média
    dos Z-scores individuais (que sempre resulta em ~0 para todo atleta).
    """
    df = _df_features_exemplo()
    features = ["Distance (m)", "Sprint Distance (m)", "Top Speed (kph)"]
    perfis = construir_perfis(df, features)

    assert set(perfis.index) == {"A1", "A2", "A3"}
    assert list(perfis.columns) == [f"Z_{f}" for f in features]

    # Os perfis NÃO podem ser todos (quase) zero / idênticos
    assert not (perfis.abs() < 1e-6).all(axis=None)
    assert not perfis.loc["A1"].equals(perfis.loc["A2"])

    # A1 (alto volume) deve ter Z positivo; A2 (baixo volume) Z negativo
    assert perfis.loc["A1", "Z_Distance (m)"] > 0
    assert perfis.loc["A2", "Z_Distance (m)"] < 0
    # A3 fica entre os dois extremos
    assert perfis.loc["A2", "Z_Distance (m)"] < perfis.loc["A3", "Z_Distance (m)"] < perfis.loc["A1", "Z_Distance (m)"]


def test_sugerir_substitutos_diferencia_similaridade():
    """
    Atletas com perfis mais próximos do atleta de referência devem ter
    similaridade maior do que atletas com perfis muito distintos.
    """
    df = _df_features_exemplo()
    features = ["Distance (m)", "Sprint Distance (m)", "Top Speed (kph)"]
    perfis = construir_perfis(df, features)

    resultados = sugerir_substitutos("A3", perfis, top_n=2)
    assert len(resultados) == 2

    por_atleta = {r["atleta_candidato"]: r for r in resultados}
    # A3 é intermediário: deve ser mais parecido com A1 ou A2 do que estes
    # entre si seriam (mas o teste central é que as similaridades difiram)
    assert por_atleta["A1"]["similaridade"] != por_atleta["A2"]["similaridade"]


def test_montar_df_features_brutas_mapeia_campos_snake_case():
    sessoes = [
        {"athlete_id": "A1", "distance_m": 1000.0, "workload": 50.0, "top_speed_kph": 30.0},
    ]
    df = montar_df_features_brutas(sessoes)
    assert df.loc[0, "Athlete ID"] == "A1"
    assert df.loc[0, "Distance (m)"] == 1000.0
    assert df.loc[0, "Workload"] == 50.0
    assert df.loc[0, "Top Speed (kph)"] == 30.0
    # Todas as features mapeadas devem estar presentes como colunas
    for feature in CAMPO_PARA_FEATURE.values():
        assert feature in df.columns
