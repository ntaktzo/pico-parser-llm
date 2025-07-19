import pdfplumber
import pandas as pd
import json

def clean_tables(all_tables):
    cleaned = []
    for entry in all_tables:
        table = entry["table"]
        # Remove keys/cols that are all None or empty
        filtered = [
            {k: v for k, v in row.items() if k and (v or v == 0)}
            for row in table
        ]
        if any(filtered):
            cleaned.append({"page": entry["page"], "table": filtered})
    return cleaned

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

cleaned_tables = clean_tables(all_tables)
output = {"new_tables": cleaned_tables}

with open(r"C:\Users\takat\pico-parser-llm\results\HTA_submission_sotosorib_DE_tables_postProcessess.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("DONE! Saved to results folder.")
