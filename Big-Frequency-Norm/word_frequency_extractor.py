import json
import pandas as pd
from tqdm import tqdm

extracted_data = []
data = pd.read_excel("word_frequency_list_60000_English.xlsx", sheet_name=0)
for _, row in tqdm(data.iterrows()):
    extracted_data.append({row['word']:row["TOTAL"]})
with open("word_frequency.json", "w") as f:
    json.dump(extracted_data, f, ensure_ascii=False)