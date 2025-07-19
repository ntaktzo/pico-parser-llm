import pdfplumber
import pandas as pd
import json

PDF_PATH = r"C:\Users\takat\pico-parser-llm\data\PDF\nsclc_kras_g12c\HTA submissions\DE\HTA_submission_sotosorib_DE.pdf"
all_tables = []

with pdfplumber.open(PDF_PATH) as pdf:
    for i, page in enumerate(pdf.pages):
        tables = page.extract_tables()
        for t in tables:
            if len(t) > 1:
                df = pd.DataFrame(t[1:], columns=t[0])
                all_tables.append({
                    "page": i + 1,
                    "table": df.to_dict(orient='records')
                })

# SAVE UNDER "new_tables" KEY
output = {"new_tables": all_tables}

with open(r"C:\Users\takat\pico-parser-llm\results\HTA_submission_sotosorib_DE_tables.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("DONE! Saved to results folder.")
