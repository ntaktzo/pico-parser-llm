import os
import re
import json
import pdfplumber
from collections import Counter
import statistics

class PDFProcessor:
    def __init__(
        self,
        pdf_path,
        boilerplate_threshold=0.3,
        doc_id=None,
        language="unknown",
        region="unknown",
        title="Unknown Title",
        created_date="unknown_date",
        keywords=None
    ):
        self.pdf_path = pdf_path
        self.boilerplate_threshold = boilerplate_threshold
        self.doc_id = doc_id or os.path.splitext(os.path.basename(pdf_path))[0]
        self.language = language
        self.region = region
        self.title = title
        self.created_date = created_date
        self.keywords = keywords or []

    def print_boilerplate_lines(self, boilerplate_normed, normed_line_pages):
        print(f"Boilerplate lines removed from {self.pdf_path}:")
        for normed in boilerplate_normed:
            pages = sorted(normed_line_pages[normed])
            print(f"'{normed}' found on pages: {pages}")

    @staticmethod
    def remove_links(line: str) -> str:
        line_no_links = re.sub(r'\(?(?:https?://|www\.)\S+\)?', '', line)
        line_no_links = re.sub(r'\(\s*\)', '', line_no_links)
        return line_no_links.strip()

    @staticmethod
    def advanced_normalize(line: str) -> str:
        norm = line.lower()
        norm = re.sub(r'\d+', '', norm)          # remove digits
        norm = re.sub(r'[^a-z]+', '', norm)      # remove non-alpha
        return norm.strip()

    @staticmethod
    def contains_of_contents(line: str) -> bool:
        return bool(re.search(r'\bof\s+contents\b', line, re.IGNORECASE))

    @staticmethod
    def is_toc_dotline(line: str) -> bool:
        return bool(re.search(r'\.{5,}\s*\d+$', line))

    @staticmethod
    def is_footnote_source(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False

        # e.g. "3 " or "[2]" or "2."
        if not re.match(r'^(\[\d+\]|[\d\.]+)\b', stripped):
            return False

        reference_patterns = [
            r'et al\.',
            r'Disponible en ligne',
            r'consulté le',
            r'NEJM',
            r'PubMed',
            r'doi'
        ]
        combined = '(' + '|'.join(reference_patterns) + ')'
        if re.search(combined, stripped, flags=re.IGNORECASE):
            return True
        return False

    def detect_headings_by_font(self, page):
        """
        Detect headings based on font size, boldness, numbering, uppercase, and length.
        """
        words = page.extract_words(
            x_tolerance=2, 
            y_tolerance=2, 
            extra_attrs=["fontname", "size"]
        )
        if not words:
            return []

        # sort top->down, left->right
        words_sorted = sorted(words, key=lambda w: (round(w.get('top', 0)), w.get('x0', 0)))
        lines = []
        current_line = []
        line_tolerance = 5

        for w in words_sorted:
            # skip words missing crucial bounding info
            if w.get('size') is None or w.get('top') is None or w.get('bottom') is None:
                continue

            if not current_line:
                current_line = [w]
                continue

            last_word = current_line[-1]
            if abs(w['top'] - last_word['top']) <= line_tolerance:
                current_line.append(w)
            else:
                lines.append(current_line)
                current_line = [w]
        if current_line:
            lines.append(current_line)

        line_objects = []
        font_sizes = []

        for line_words in lines:
            text_parts = [lw["text"] for lw in line_words if lw.get("text")]
            if not text_parts:
                continue

            text = " ".join(text_parts).strip()

            # average font size
            sizes = [lw['size'] for lw in line_words if lw.get('size') is not None]
            avg_size = statistics.mean(sizes) if sizes else 10.0

            line_objects.append({
                "text": text,
                "avg_size": avg_size,
                "words": line_words
            })
            font_sizes.append(avg_size)

        if not line_objects:
            return []

        median_size = statistics.median(font_sizes)

        for obj in line_objects:
            text = obj["text"]
            avg_size = obj["avg_size"]
            lw = obj["words"]
            word_count = len(text.split())

            # Font size ratio scoring
            ratio_to_median = avg_size / median_size if median_size else 1.0
            font_score = 2 if ratio_to_median >= 1.3 else (1 if ratio_to_median >= 1.1 else 0)

            # Uppercase ratio scoring
            letters = [c for c in text if c.isalpha()]
            uppercase_ratio = sum(c.isupper() for c in letters) / len(letters) if letters else 0
            uppercase_score = 2 if uppercase_ratio > 0.8 else (1 if uppercase_ratio > 0.6 else 0)

            # Length scoring (shorter lines more likely headings)
            length_score = 2 if word_count < 6 else (1 if word_count < 12 else 0)

            # Boldface scoring
            bold_words = [
                w for w in lw if w.get('fontname') and 'bold' in w['fontname'].lower()
            ]
            bold_ratio = len(bold_words) / len(lw) if lw else 0
            bold_score = 2 if bold_ratio > 0.7 else (1 if bold_ratio > 0.3 else 0)

            # Numbering at start scoring (newly added)
            numbering_score = 0
            numbering_pattern = r'^(\d+[\.\,\)](\d+[\.\,\)])*\s+)'
            if re.match(numbering_pattern, text):
                numbering_score = 2  # numbering strongly indicates heading

            # Sum all scores (without colon check)
            total_score = font_score + uppercase_score + length_score + bold_score + numbering_score

            # Final threshold (>=3)
            obj["likely_heading"] = (total_score >= 3)

        return line_objects

    def extract_preliminary_chunks(self):
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                num_pages = len(pdf.pages)
                normed_line_pages = {}
                pages_with_lines = []

                # gather lines from each page
                for i, page in enumerate(pdf.pages, start=1):
                    line_objs = self.detect_headings_by_font(page)
                    pages_with_lines.append((i, line_objs))

                    # gather normalized lines for boilerplate detection
                    unique_normed = set()
                    for lo in line_objs:
                        norm = self.advanced_normalize(lo["text"])
                        if norm:
                            unique_normed.add(norm)
                    for n in unique_normed:
                        normed_line_pages.setdefault(n, set()).add(i)

                # identify boilerplate lines
                boilerplate_normed = set()
                for normed_line, pset in normed_line_pages.items():
                    if len(pset) / num_pages >= self.boilerplate_threshold:
                        boilerplate_normed.add(normed_line)

                self.print_boilerplate_lines(boilerplate_normed, normed_line_pages)

                # flatten & filter
                filtered_lines = []
                in_footnote = False

                for page_num, line_objs in pages_with_lines:
                    for lo in line_objs:
                        line_text = lo["text"]
                        if not line_text.strip():
                            continue

                        if in_footnote:
                            # stop skipping if blank or heading
                            if not line_text.strip() or lo["likely_heading"]:
                                in_footnote = False
                                continue
                            else:
                                continue

                        if self.is_footnote_source(line_text):
                            in_footnote = True
                            continue

                        line_text = self.remove_links(line_text)
                        if not line_text.strip():
                            continue

                        # skip boilerplate
                        norm = self.advanced_normalize(line_text)
                        if not norm or norm in boilerplate_normed:
                            continue

                        if self.contains_of_contents(line_text):
                            continue

                        if self.is_toc_dotline(line_text):
                            continue

                        heading_flag = lo["likely_heading"]
                        filtered_lines.append((page_num, line_text.strip(), heading_flag))

                # build chunks
                chunks = []
                buffer = []
                current_heading = ""

                def flush_buffer():
                    if buffer:
                        min_page = min(x[0] for x in buffer)
                        max_page = max(x[0] for x in buffer)
                        chunk_text = "\n".join(x[1] for x in buffer)
                        chunks.append({
                            "heading": current_heading,
                            "text": chunk_text,
                            "start_page": min_page,
                            "end_page": max_page
                        })
                        buffer.clear()

                for (pnum, text_line, is_heading) in filtered_lines:
                    if is_heading:
                        flush_buffer()
                        current_heading = text_line
                    else:
                        buffer.append((pnum, text_line))

                flush_buffer()
                return chunks

        except Exception as e:
            print(f"Error reading {self.pdf_path}: {e}")
            return []

    @staticmethod
    def flatten_table(table):
        flattened = []
        for row in table:
            cleaned_cells = [cell.strip() for cell in row if cell and cell.strip()]
            if cleaned_cells:
                row_text = " | ".join(cleaned_cells)
                flattened.append(row_text)
        return "\n".join(flattened)

    @staticmethod
    def process_pdfs(
        input_dir,
        output_dir,
        boilerplate_threshold=0.3,
        doc_id=None,
        language="unknown",
        region="unknown",
        title="Unknown Title",
        created_date="unknown_date",
        keywords=None
    ):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(".pdf"):
                    pdf_path = os.path.join(root, file)
                    processor = PDFProcessor(
                        pdf_path,
                        boilerplate_threshold=boilerplate_threshold,
                        doc_id=os.path.splitext(file)[0],
                        language=language,
                        region=region,
                        title=title,
                        created_date=created_date,
                        keywords=keywords or []
                    )

                    chunks = processor.extract_preliminary_chunks()
                    if chunks:
                        relative_path = os.path.relpath(root, input_dir)
                        out_dir = os.path.join(output_dir, relative_path)
                        os.makedirs(out_dir, exist_ok=True)

                        output_file = os.path.join(
                            out_dir,
                            f"{os.path.splitext(file)[0]}_cleaned.json"
                        )
                        with open(output_file, "w", encoding="utf-8") as f:
                            json.dump(chunks, f, indent=2, ensure_ascii=False)
                    else:
                        print(f"Error reading {pdf_path}")




import os
import shutil
import re
import torch
import gc
from langdetect import detect
from transformers import pipeline

class Translator:
    def __init__(self, input_dir, output_dir, max_chunk_length=300):  
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.max_chunk_length = max_chunk_length
        self.models = {
            'fr': 'Helsinki-NLP/opus-mt-fr-en',
            'de': 'Helsinki-NLP/opus-mt-de-en',
            'pl': 'Helsinki-NLP/opus-mt-pl-en',
            'es': 'Helsinki-NLP/opus-mt-es-en',
        }
        self.translators = {}

        # ✅ Detect GPU Support (Windows & Mac Compatible)
        if torch.cuda.is_available():  # ✅ NVIDIA CUDA (Windows/Linux)
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():  # ✅ MPS (Mac M1/M2)
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")  # ✅ Fallback to CPU
        
        print(f"Using device: {self.device}")

    def detect_language(self, text):
        """Detects the language of the input text."""
        try:
            lang = detect(text)
            print(f"Detected language: {lang}")
            return lang
        except Exception as e:
            print(f"Language detection failed: {e}")
            return None

    def get_translator(self, lang):
        """Returns a translation pipeline for the given language."""
        if lang in self.translators:
            return self.translators[lang]

        model_name = self.models.get(lang)
        if model_name:
            print(f"Loading translation model: {model_name} on {self.device}")
            translator = pipeline(
                "translation",
                model=model_name,
                torch_dtype=torch.float16 if self.device.type != "cpu" else torch.float32,  # ✅ Reduce memory usage on GPU
                device=self.device
            )
            self.translators[lang] = translator
            return translator
        else:
            print(f"No translation model available for language: {lang}")
            return None

    def chunk_text(self, text):
        """Splits long text into smaller chunks for translation."""
        sentences = re.split(r'(?<=\.) ', text)  # ✅ Better sentence splitting
        chunks, current_chunk = [], ''
        for sentence in sentences:
            if len(current_chunk.split()) + len(sentence.split()) <= self.max_chunk_length:
                current_chunk += ' ' + sentence
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def translate_documents(self):
        """Translates all text files in the input directory and saves output."""
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith(".txt"):
                    input_path = os.path.join(root, file)
                    with open(input_path, "r", encoding="utf-8") as f:
                        text = f.read()

                    lang = self.detect_language(text)
                    relative_path = os.path.relpath(root, self.input_dir)
                    output_subdir = os.path.join(self.output_dir, relative_path)
                    os.makedirs(output_subdir, exist_ok=True)
                    output_file = os.path.join(output_subdir, file)

                    if lang == 'en':
                        # ✅ Copy English files directly without translation
                        print(f"{file} is already in English. Copying without translation.")
                        shutil.copy(input_path, output_file)  # ✅ More efficient copying
                        continue

                    translator = self.get_translator(lang)
                    if not translator:
                        continue

                    print(f"Translating '{file}' from {lang} to English...")
                    chunks = self.chunk_text(text)
                    translated_chunks = []

                    for chunk in chunks:
                        translation = translator(chunk, truncation=True)[0]['translation_text']
                        translated_chunks.append(translation)

                    translated_text = "\n\n".join(translated_chunks)

                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(translated_text)

                    print(f"Translated text saved to {output_file}")

                    # ✅ Free GPU memory after each file to prevent crashes
                    gc.collect()
                    if self.device.type == "mps":
                        torch.mps.empty_cache()
                    elif self.device.type == "cuda":
                        torch.cuda.empty_cache()

