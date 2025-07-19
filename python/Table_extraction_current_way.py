from process import PDFProcessor

pdf_path = r"C:\Users\takat\pico-parser-llm\data\PDF\nsclc_kras_g12c\HTA submissions\DE\HTA_submission_sotosorib_DE.pdf"
processor = PDFProcessor(pdf_path)
result = processor.extract_preliminary_chunks()

import json
with open(r"C:\Users\takat\pico-parser-llm\results\HTA_submission_sotosorib_DE_cleaned.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print("DONE! Saved to results folder.")
