import os
import json
import re

# ── CONFIG: your PICO categories and keywords ─────────────────────────────

CATEGORIES = {
    "population": {
        "extract_kw": ["Population", "Patients", "Adults", "Participants", "Subjects"],
        "score_kw":    ["population", "patients", "adults", "participants", "subjects"],
        "score_phrases":[
            "population with", "patients with", "adults with",
            "participants with", "subjects with", "diagnosed with"
        ]
    },
    "intervention": {
        "extract_kw": ["Intervention", "Treatment", "Drug", "Therapy", "Dose"],
        "score_kw":    ["intervention", "treatment", "drug", "therapy", "dose"],
        "score_phrases":["compared to", "versus", "vs", "in combination with"]
    },
    "comparator": {
        "extract_kw": ["Comparator", "Control", "Placebo", "Standard care", "Usual care"],
        "score_kw":    ["comparator", "control", "placebo", "standard care", "usual care"],
        "score_phrases":[]
    },
    "outcome": {
        "extract_kw": ["Outcome", "Result", "Effect", "Efficacy", "Safety"],
        "score_kw":    ["outcome", "result", "efficacy", "response", "survival"],
        "score_phrases":["hazard ratio", "median", "odds ratio", "risk reduction"]
    }
}

WINDOW = 300   # chars around each hit
TOP_N  = 10    # how many to keep per category

# ── UTILS ────────────────────────────────────────────────────────────────

def extract_context(text, keywords):
    hits = []
    for kw in keywords:
        for m in re.finditer(r'\b' + re.escape(kw) + r'\b', text, re.IGNORECASE):
            s,e = m.start(), m.end()
            snippet = text[max(0,s-WINDOW):min(len(text), e+WINDOW)].strip()
            hits.append({"keyword": kw, "context": snippet})
    return hits

def score_extract(txt, score_kw, score_phrases):
    s = 0
    low = txt.lower()
    for kw in score_kw:
        s += low.count(kw)
    if any(ph in low for ph in score_phrases):
        s += 5
    if len(txt) < 50:
        s -= 1
    if txt.count(".") > 1:
        s += 1
    return s

# parse headers in cleaned_<cat>.txt
PAT = re.compile(
    r"=== .*? Extract (\d+) \((.*?)\) ===\s*\n"  
    r"(.*?)(?=(?:=== .*? Extract \d+|\Z))",
    re.DOTALL
)

# ── MAIN ─────────────────────────────────────────────────────────────────

script_dir  = os.path.dirname(os.path.abspath(__file__))
project     = os.path.dirname(script_dir)
hta_base    = os.path.join(project, "data", "text_cleaned", "nsclc_kras_g12c", "clinical guidelines")
results_dir = os.path.join(project, "results")
os.makedirs(results_dir, exist_ok=True)

for lang in os.listdir(hta_base):
    lang_dir = os.path.join(hta_base, lang)
    if not os.path.isdir(lang_dir): continue

    for fname in os.listdir(lang_dir):
        if not fname.endswith("_cleaned.json"): continue

        full_path = os.path.join(lang_dir, fname)
        with open(full_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        chunks = doc.get("chunks", doc)
        full_text = " ".join(ch.get("text","") for ch in chunks)

        # collect per-category top extracts
        top_by_cat = {}

        for cat, cfg in CATEGORIES.items():
            # 1) extract raw contexts
            ext = extract_context(full_text, cfg["extract_kw"])
            # write raw
            raw_name = f"{fname}.cleaned_{cat}.txt"
            with open(os.path.join(results_dir, raw_name), "w", encoding="utf-8") as o:
                if not ext:
                    o.write("No matches.\n")
                else:
                    for i,e in enumerate(ext,1):
                        o.write(f"=== {cat.capitalize()} Extract {i} ({e['keyword']}) ===\n{e['context']}\n\n")

            # 2) score + rank
            scored = [(score_extract(e["context"], cfg["score_kw"], cfg["score_phrases"]), i, e)
                      for i,e in enumerate(ext,1)]
            scored.sort(key=lambda x:(-x[0], x[1]))
            top_by_cat[cat] = scored[:TOP_N]

            # 3) write per-category ranked
            rank_name = f"ranked_{cat}_{fname}"
            with open(os.path.join(results_dir, rank_name), "w", encoding="utf-8") as o:
                o.write(f"# Top {TOP_N} {cat} extracts from {fname}\n\n")
                for rank,(sc,i,e) in enumerate(top_by_cat[cat],1):
                    o.write(f"=== Ranked {cat.capitalize()} {rank} (Extract #{i}, {e['keyword']}) — Score: {sc} ===\n")
                    o.write(e["context"] + "\n\n")

            print(f"✅ {cat} raw→{raw_name}, ranked→{rank_name}")

        # 4) write combined PICO file
        pico_name = f"ranked_PICO_{fname}"
        with open(os.path.join(results_dir, pico_name), "w", encoding="utf-8") as o:
            o.write(f"# Combined top {TOP_N} from each PICO category for {fname}\n\n")
            for cat in ["population","intervention","comparator","outcome"]:
                o.write(f"## {cat.capitalize()}\n\n")
                for rank,(sc,i,e) in enumerate(top_by_cat[cat],1):
                    o.write(f"{rank}. ({e['keyword']}, score {sc}) {e['context']}\n\n")

        print(f"🔗 Combined → {pico_name}")

print("\n🎉 All done!")
