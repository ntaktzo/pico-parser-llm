import os
import json
import re
import hashlib
import textwrap
from difflib import SequenceMatcher

# ── USER-PROVIDED KEYWORD PLACEHOLDERS ──

# These lists will be populated with user-provided keywords for each PICO category.
# You can fill these lists with your own keywords as needed.
Population_keywords_user   = []
Intervention_keywords_user = []
Comparator_keywords_user   = []
Outcome_keywords_user      = ["non-small cell lung cancer", "NSCLC", "KRAS G12C", "KRAS mutation", "lung cancer", "advanced NSCLC"]

# ──  PICO categories and keywords ─────────────────────────────
CATEGORIES = {
    "population": {
        "extract_kw": ["Population", "Patients", "Adults", "Participants", "Subjects"] + Population_keywords_user,
        "score_kw":    ["population", "patients", "adults", "participants", "subjects"] + [kw.lower() for kw in Population_keywords_user],
        "score_phrases":[
            "population with", "patients with", "adults with", "adults whose", 
            "participants with", "subjects with", "diagnosed with"
        ] + Population_keywords_user
    },
    "intervention": {
        "extract_kw": ["Intervention", "Treatment", "Drug", "Therapy", "Dose"] + Intervention_keywords_user,
        "score_kw":    ["intervention", "treatment", "drug", "therapy", "dose"] + [kw.lower() for kw in Intervention_keywords_user],
        "score_phrases":["compared to", "versus", "vs", "in combination with"] + Intervention_keywords_user
    },
    "comparator": {
        "extract_kw": ["Comparator", "Control", "Placebo", "Standard care", "Usual care"] + Comparator_keywords_user,
        "score_kw":    ["comparator", "control", "placebo", "standard care", "usual care"] + [kw.lower() for kw in Comparator_keywords_user],
        "score_phrases": Comparator_keywords_user
    },
    "outcome": {
        "extract_kw": ["Outcome", "Result", "Effect", "Efficacy", "Safety"] + Outcome_keywords_user,
        "score_kw":    ["outcome", "result", "efficacy", "response", "survival"] + [kw.lower() for kw in Outcome_keywords_user],
        "score_phrases":["hazard ratio", "median", "odds ratio", "risk reduction"] + Outcome_keywords_user
    }
}

WINDOW = 300   #chars around 
FINAL_WINDOW = 3000  # chars around each hit in final PICO extracts
TOP_N  = 100    # how many to keep per category

# ── UTILS ────────────────────────────────────────────────────────────────

def extract_context(text, keywords):
    hits = []
    for kw in keywords:
        for m in re.finditer(r'\b' + re.escape(kw) + r'\b', text, re.IGNORECASE):
            s,e = m.start(), m.end()
            snippet = text[max(0,s-WINDOW):min(len(text), e+WINDOW)].strip()
            hits.append({"keyword": kw, "context": snippet})
    return hits

# helper: build one big regex per category once
CATEGORY_REGEXES = {
    cat: re.compile(r'\b(?:' + '|'.join(map(re.escape, cfg['score_kw'])) + r')\b', re.I)
    for cat, cfg in CATEGORIES.items()
}

STUDY_REGEX = re.compile(
    r'\b(randomi[sz]ed|phase\s*[i-v]+|confidence interval|hazard ratio|'
    r'placebo[- ]controlled|double[- ]blind|cohort study|trial)|'
    # NEW: journal citations "J Clin Oncol. 2021; 33(4):123‑9."
    r'[A-Z][a-z]+ [A-Z][a-z]+\. \d{4};\s*\d{1,4}\([\dA-Za-z]+\):\d{1,5}'
    # NEW: "N Engl J Med 2020; 383:1328"
    r'|[A-Z][a-z]+\sJ\sMed\.\s?\d{4};'
    # NEW: standalone citations like "2017;39(5):881- 4"
    r'|\b\d{4};\d{1,4}(\(\d+\))?:\d{1,5}(-\s?\d{1,4})?'
    # NEW: "published Online First" phrase
    r'|published Online First'
    # NEW: "et al." reference marker
    r'|et al\.'
    r'|PubMed\s\d{2}\.\d{2}\.\d{4}'
    ## ── ADD-ONS: broader trial & citation cues ──────────────
    
    # RCT shorthand or study-design keywords
    r'|RCT\b|controlled\s+trial|open[- ]label|case[- ]control|case[- ]series|'
    r'prospective\s+study|retrospective\s+study|multicentre|multi[- ]centre|'
    r'double[- ]masked|single[- ]masked|cross[- ]over|cross[- ]over\s+trial|'
    # Phase written with arabic numerals (Phase 2, Phase 3b, Phase 1/2)
    r'\b[Pp]hase\s*[1-4IVX]+[a-c]?(/[1-4IVX]+[a-c]?)?\b'
    # ClinicalTrials.gov or ISRCTN registry IDs
    r'|NCT\d{8}\b|ISRCTN\d{8}\b'
    # DOI, PMID, or PMCID tokens
    r'|doi:\s*10\.\d{4,9}/[-._;()/:A-Za-z0-9]+'
    r'|PMID:\s*\d{6,}'
    r'|PMCID:\s*PMC\d+'
    # Author-list citation lines (e.g., "Smith AB, Jones C. 2019;12:100-110")
    r'|(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*)+(?:et al\.,?\s*)?\d{4};\s*\d{1,4}'
    # Vancouver-style numeric in-text citations "[12]" or "[3, 5, 7]"
    r'|\[\d+(?:\s*,\s*\d+)*\]'
    # Author-year in-text citations "(Smith 2020)" or "(Doe et al., 1999)"
    r'|\([A-Z][A-Za-z\-]+[^()]*?(?:19|20)\d{2}[a-z]?\)'
    # Journal abbreviations with ≥2 capitalised tokens ("J Exp Med", "Acta Physiol.")
    r'|\b(?:[A-Z][A-Za-z]+\.){2,}\s?',
    re.I
)


STUDY_PENALTY = 50   #NUUUUUUUUUKE THEM 


def count_categories_present(text):
    """Return how many distinct PICO categories have ≥1 keyword in `text`."""
    return sum(bool(rx.search(text)) for rx in CATEGORY_REGEXES.values())

def categories_in(text: str) -> list[str]:
    """Return list like ['P','I','O'] for the snippet."""
    mapping = {'population':'P', 'intervention':'I',
               'comparator':'C', 'outcome':'O'}
    found = []
    for cat, rx in CATEGORY_REGEXES.items():
        if rx.search(text):
            found.append(mapping[cat])
    return found

def score_extract(txt, score_kw, score_phrases, bingo_bonus=7, bingo_min_cats=3):
    """
    txt             : snippet
    score_kw        : usual keyword list for *this* category
    score_phrases   : usual phrase bonuses for *this* category
    """
    s = 0
    low = txt.lower()

    # old keyword frequency score
    for kw in score_kw:
        s += low.count(kw)

    # phrase bonus
    if any(ph in low for ph in score_phrases):
        s += 5

    # --- NEW: penalise obvious trial-table text -----------
    if STUDY_REGEX.search(low):
        s -= STUDY_PENALTY

    # prefer longer sentences
    if txt.count(".") > 1:
        s += 1
    if len(txt) < 50:
        s -= 1

    # --- NEW: fusion bonus ------------------------------------
    cats_here = count_categories_present(low)
    if cats_here >= bingo_min_cats:
        s += bingo_bonus
        # you can also attach metadata later:
        # e['bingo'] = True
    # ----------------------------------------------------------

    return s

#deduplicate extracts by normalised text

SIM_THRESHOLD = (0.90 * WINDOW)/6.8  # 90 % similarity → treat as duplicate

def word_set(text):
    """Return a set of lowercased words from the text."""
    return set(re.findall(r'\w+', text.lower()))

def very_similar(a, b, threshold=SIM_THRESHOLD):
    """
    Return True if a and b share at least `threshold` words.
    `threshold` is auto-calculated from SIM_THRESHOLD.
    """
    wa = word_set(a)
    wb = word_set(b)
    return len(wa & wb) >= int(threshold)

def expand_snippet(context, full_text, expand_size):
        """Expand the original context by ±expand_size chars in full_text."""
        i = full_text.lower().find(context[:60].lower())  # anchor on snippet beginning
        if i == -1:
            return context  # fallback
        s = max(0, i - expand_size)
        e = min(len(full_text), i + len(context) + expand_size)
        return full_text[s:e].strip()

# ── MAIN ─────────────────────────────────────────────────────────────────

script_dir  = os.path.dirname(os.path.abspath(__file__))
project     = os.path.dirname(script_dir)
hta_base    = os.path.join(project, "data", "text_cleaned", "nsclc_kras_g12c", "clinical guidelines")
results_dir = os.path.join(project, "results")
os.makedirs(results_dir, exist_ok=True)

WRITE_DEBUG = False          # ← toggle to True when you need raw / per-cat files

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

            # --- DEDUPLICATE CONTEXTS (fuzzy) ---
            dedup_ext = []
            for e in ext:
                ctx = e["context"]
                if any(very_similar(ctx, kept["context"]) for kept in dedup_ext):
                    continue  # skip near‑duplicate
                dedup_ext.append(e)
            ext = dedup_ext          # replace original list
            # ------------------------------------

            # Only write raw and ranked files if WRITE_DEBUG is True
            if WRITE_DEBUG:
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

            if WRITE_DEBUG:
                rank_name = f"ranked_{cat}_{fname}"
                with open(os.path.join(results_dir, rank_name), "w", encoding="utf-8") as o:
                    o.write(f"# Top {TOP_N} {cat} extracts from {fname}\n\n")
                    for rank,(sc,i,e) in enumerate(top_by_cat[cat],1):
                        o.write(f"=== Ranked {cat.capitalize()} {rank} (Extract #{i}, {e['keyword']}) — Score: {sc} ===\n")
                        o.write(e["context"] + "\n\n")
                print(f"✅ {cat} raw→{raw_name}, ranked→{rank_name}")



    

        # 4) write combined PICO file
        # gather every scored snippet from all cats in ONE pile
        all_scored = []
        for cat in CATEGORIES:
            for sc, idx, e in top_by_cat[cat]:
                all_scored.append((sc, cat, e))

        all_scored.sort(key=lambda x: -x[0])
        all_scored = all_scored[:TOP_N]   # global TOP_N now!

        print(f"→ After scoring we kept {len(all_scored)} snippets for {fname}")

        pico_name = f"ranked_PICO_{fname}"
        with open(os.path.join(results_dir, pico_name), "w", encoding="utf-8") as o:
            o.write(f"# Top {TOP_N} PICO-like extracts for {fname}\n\n")
            for rank, (sc, cat, e) in enumerate(all_scored, 1):
                cats = categories_in(e['context'].lower())      # e.g. ['P','I','O']
                bingo = f" ✅ BINGO ({''.join(cats)})" if len(cats) >= 3 else ""
                o.write(f"{rank}. [{cat}] score {sc}{bingo}\n")
                expanded = expand_snippet(e["context"], full_text, (FINAL_WINDOW -WINDOW)  // 2)
                o.write(textwrap.fill(expanded, 120) + "\n\n")  
        print(f"🔗 Combined → {pico_name}")

print("\n🎉 All done!")



import sys
print("Exiting now...")
sys.exit(0)