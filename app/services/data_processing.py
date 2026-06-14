import pandas as pd
import numpy as np
from io import BytesIO

COLUNAS_FISICAS = [
    'Duration (mins)', 'Workload', 'Distance (m)',
    'Metres per Minute (m)', 'High Intensity Running (m)',
    'Sprint Distance (m)', 'Raw Top Speed (kph)', 'No. of Sprints',
    'Top Speed (kph)', 'Avg Speed (kph)',
    'Accelerations', 'Decelerations',
    'No. of High Intensity Events', 'Session Load'
]

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

def process_upload_excel(file_content: bytes) -> pd.DataFrame:
    # Ler Excel
    df_raw = pd.read_excel(BytesIO(file_content))
    
    # 1. Filtro Whole Session
    df_ws = df_raw[df_raw['Segment Name'] == 'Whole Session'].copy()
    df_ws = df_ws.reset_index(drop=True)
    
    # 2. Remover duplicatas
    chave_dedup = ['Athlete ID', 'Start Date', 'Segment Name']
    df_ws = df_ws.drop_duplicates(subset=chave_dedup, keep='first')
    
    # 3. Zeros em distâncias/duração para NaN
    cols_nao_zero = ['Distance (m)', 'Duration (mins)']
    for col in cols_nao_zero:
        if col in df_ws.columns:
            df_ws[col] = df_ws[col].replace(0, np.nan)
            
    # 4. Imputação por mediana do atleta
    cols_disponiveis = [c for c in COLUNAS_FISICAS if c in df_ws.columns]
    df_ws[cols_disponiveis] = df_ws.groupby('Athlete ID')[cols_disponiveis].transform(
        lambda x: x.fillna(x.median())
    )
    # Mediana global fallback
    df_ws[cols_disponiveis] = df_ws[cols_disponiveis].fillna(df_ws[cols_disponiveis].median())
    
    return df_ws
