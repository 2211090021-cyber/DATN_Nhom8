import pandas as pd
import json

try:
    df_raw = pd.read_excel('docs/YCDL_Raw.xlsx')
    raw_cols = df_raw.columns.tolist()
except Exception as e:
    raw_cols = [str(e)]

try:
    df_mapped = pd.read_excel('data/preprocess/YCDL_Features_Mapped.xlsx')
    mapped_cols = df_mapped.columns.tolist()
except Exception as e:
    mapped_cols = [str(e)]

with open('scratch/cols.json', 'w', encoding='utf-8') as f:
    json.dump({'raw': raw_cols, 'mapped': mapped_cols}, f, ensure_ascii=False, indent=2)
