import os
import re
import json
import pdfplumber
from collections import Counter
import statistics

import os
import re
import json
import statistics
import pdfplumber

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
        norm = re.sub(r'\d+', '', norm)       # remove digits
        norm = re.sub(r'[^a-z]+', '', norm)     # remove all non-alpha characters
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
        Detect headings based on font size, boldness, numbering, uppercase, length,
        and vertical spacing (space above and below).
        """
        words = page.extract_words(
            x_tolerance=2, 
            y_tolerance=2, 
            extra_attrs=["fontname", "size"]
        )
        if not words:
            return []

        # Sort words top->down, left->right
        words_sorted = sorted(words, key=lambda w: (round(w.get('top', 0)), w.get('x0', 0)))

        # 1) Group words into raw lines
        raw_lines = []
        current_line = []
        line_tolerance = 5

        for w in words_sorted:
            if w.get('size') is None or w.get('top') is None:
                continue

            if not current_line:
                current_line = [w]
                continue

            last_word = current_line[-1]
            if abs(w['top'] - last_word['top']) <= line_tolerance:
                current_line.append(w)
            else:
                raw_lines.append(current_line)
                current_line = [w]
        if current_line:
            raw_lines.append(current_line)

        # 2) Merge adjacent lines if same font style, etc. (multi-line headings)
        merged_lines = []
        current_group = raw_lines[0]

        for next_line in raw_lines[1:]:
            current_avg_size = statistics.mean([w['size'] for w in current_group])
            next_avg_size = statistics.mean([w['size'] for w in next_line])
            current_bold = all('bold' in w.get('fontname', '').lower() for w in current_group)
            next_bold = all('bold' in w.get('fontname', '').lower() for w in next_line)
            vertical_gap = next_line[0]['top'] - current_group[-1]['bottom']

            if (
                abs(current_avg_size - next_avg_size) < 1.0 and
                current_bold == next_bold and
                vertical_gap <= 10
            ):
                current_group.extend(next_line)
            else:
                merged_lines.append(current_group)
                current_group = next_line
        if current_group:
            merged_lines.append(current_group)

        # 3) Build line_objects
        line_objects = []
        font_sizes = []

        for line_words in merged_lines:
            text_parts = [lw["text"] for lw in line_words if lw.get("text")]
            if not text_parts:
                continue

            text = " ".join(text_parts).strip()
            sizes = [lw['size'] for lw in line_words if lw.get('size') is not None]
            avg_size = statistics.mean(sizes) if sizes else 10.0
            top_pos = min(lw['top'] for lw in line_words)
            bottom_pos = max(lw['bottom'] for lw in line_words)

            line_objects.append({
                "text": text,
                "avg_size": avg_size,
                "top": top_pos,
                "bottom": bottom_pos,
                "words": line_words
            })
            font_sizes.append(avg_size)

        if not line_objects:
            return []

        median_size = statistics.median(font_sizes)

        # 4) Score each line
        for idx, obj in enumerate(line_objects):
            text = obj["text"]
            avg_size = obj["avg_size"]
            lw = obj["words"]
            word_count = len(text.split())

            # Font size ratio
            ratio_to_median = avg_size / median_size if median_size else 1.0
            font_score = 2 if ratio_to_median >= 1.3 else (1 if ratio_to_median >= 1.1 else 0)

            # Uppercase ratio
            letters = [c for c in text if c.isalpha()]
            uppercase_ratio = sum(c.isupper() for c in letters) / len(letters) if letters else 0
            uppercase_score = 2 if uppercase_ratio > 0.8 else (1 if uppercase_ratio > 0.6 else 0)

            # Length
            length_score = 2 if word_count < 6 else (1 if word_count < 12 else 0)

            # Boldface
            bold_words = [w for w in lw if w.get('fontname') and 'bold' in w['fontname'].lower()]
            bold_ratio = len(bold_words) / len(lw) if lw else 0
            bold_score = 2 if bold_ratio > 0.7 else (1 if bold_ratio > 0.3 else 0)

            # Numbering
            numbering_score = 0
            numbering_pattern = r'^(\d+(\.\d+)*[\.\,\)]?\s+)'
            if re.match(numbering_pattern, text):
                numbering_score = 2

            # Vertical spacing
            vertical_space_score = 0
            space_above = obj["top"] - line_objects[idx-1]["bottom"] if idx > 0 else 100
            space_below = line_objects[idx+1]["top"] - obj["bottom"] if idx < len(line_objects)-1 else 100

            if space_above > 15 and space_below > 10:
                vertical_space_score = 2
            elif space_above > 10 or space_below > 8:
                vertical_space_score = 1

            total_score = (
                font_score +
                uppercase_score +
                length_score +
                bold_score +
                numbering_score +
                vertical_space_score
            )
            obj["likely_heading"] = (total_score >= 5)

        return line_objects

    def extract_tables_from_pdf(self, pdf):
        """
        Extracts tables from the PDF and returns them as formatted strings that preserve their 2D structure.
        Returns a list of dictionaries, each containing:
          - page (int): The page number where the table was found.
          - table_index (int): The index of the table on that page.
          - text (str): A formatted string representation of the table.
        """
        tables_info = []
        for page_num, page in enumerate(pdf.pages, start=1):
            page_tables = page.extract_tables()
            if not page_tables:
                continue
            for i, table in enumerate(page_tables, start=1):
                if not table:
                    continue
                formatted_table = self.flatten_table(table)
                if formatted_table.strip():
                    tables_info.append({
                        "page": page_num,
                        "table_index": i,
                        "text": formatted_table
                    })
        return tables_info

    @staticmethod
    def flatten_table(table):
        """
        Converts a table (list of lists) into a readable, formatted text representation with aligned columns.
        This version cleans each cell by replacing newlines and multiple spaces with a single space.
        :param table: 2D list representing the table.
        :return: A formatted table as a single string.
        """
        if not table:
            return ""
        # Determine the maximum number of columns in any row
        ncols = max(len(row) for row in table)
        col_widths = [0] * ncols
        cleaned_table = []
        # Clean each cell in the table (remove newlines and extra spaces)
        for row in table:
            cleaned_row = []
            for cell in row:
                if cell is None:
                    cleaned_cell = ""
                else:
                    # Replace any sequence of whitespace (including newlines) with a single space
                    cleaned_cell = re.sub(r'\s+', ' ', cell).strip()
                cleaned_row.append(cleaned_cell)
            cleaned_table.append(cleaned_row)
        # Compute maximum width per column
        for row in cleaned_table:
            for i in range(len(row)):
                col_widths[i] = max(col_widths[i], len(row[i]))
        # Build each row with appropriate padding
        formatted_rows = []
        for row in cleaned_table:
            formatted_row = []
            for i in range(ncols):
                cell = row[i] if i < len(row) else ""
                formatted_row.append(cell.ljust(col_widths[i]))
            formatted_rows.append(" | ".join(formatted_row))
        return "\n".join(formatted_rows)

    def extract_preliminary_chunks(self):
        """
        Main function that:
          - Extracts text from each page, performs cleanup, and identifies chunks based on headings.
          - Extracts and formats tables from each page.
        Returns:
          Dictionary containing:
            - doc_id: Document identifier.
            - chunks: List of text chunks with headings.
            - tables: List of formatted tables with page numbers.
        """
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                # Extract tables
                tables_list = self.extract_tables_from_pdf(pdf)

                num_pages = len(pdf.pages)
                normed_line_pages = {}
                pages_with_lines = []

                # Extract text and headings from each page
                for i, page in enumerate(pdf.pages, start=1):
                    line_objs = self.detect_headings_by_font(page)
                    pages_with_lines.append((i, line_objs))

                    # Identify potential boilerplate lines
                    unique_normed = set()
                    for lo in line_objs:
                        norm = self.advanced_normalize(lo["text"])
                        if norm:
                            unique_normed.add(norm)
                    for n in unique_normed:
                        normed_line_pages.setdefault(n, set()).add(i)

                # Determine boilerplate lines
                boilerplate_normed = set()
                for normed_line, pset in normed_line_pages.items():
                    if len(pset) / num_pages >= self.boilerplate_threshold:
                        boilerplate_normed.add(normed_line)
                self.print_boilerplate_lines(boilerplate_normed, normed_line_pages)

                filtered_lines = []
                in_footnote = False

                for page_num, line_objs in pages_with_lines:
                    for lo in line_objs:
                        line_text = lo["text"]
                        if not line_text.strip():
                            continue

                        if in_footnote:
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

                        norm = self.advanced_normalize(line_text)
                        if not norm or norm in boilerplate_normed:
                            continue
                        if self.contains_of_contents(line_text):
                            continue
                        if self.is_toc_dotline(line_text):
                            continue

                        heading_flag = lo["likely_heading"]
                        filtered_lines.append((page_num, line_text.strip(), heading_flag))

                # Compile text chunks
                chunks = []
                buffer = []
                current_heading = ""

                def flush_buffer():
                    nonlocal current_heading
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

                return {
                    "doc_id": self.doc_id,
                    "chunks": chunks,
                    "tables": tables_list
                }

        except Exception as e:
            print(f"Error reading {self.pdf_path}: {e}")
            return {"doc_id": self.doc_id, "chunks": [], "tables": []}

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

                    result = processor.extract_preliminary_chunks()
                    if result and result["chunks"]:
                        relative_path = os.path.relpath(root, input_dir)
                        out_dir = os.path.join(output_dir, relative_path)
                        os.makedirs(out_dir, exist_ok=True)

                        output_file = os.path.join(
                            out_dir,
                            f"{os.path.splitext(file)[0]}_cleaned.json"
                        )
                        with open(output_file, "w", encoding="utf-8") as f:
                            json.dump(result, f, indent=2, ensure_ascii=False)
                    else:
                        print(f"Error reading {pdf_path}")



import os
import re
import json
import gc
import shutil
import torch
from transformers import pipeline
from langdetect import detect

class Translator:
    """
    Translator class that processes JSON files with the following structure:
    
      {
        "doc_id": "…",
        "chunks": [
          {
            "heading": "…",
            "text": "…",
            ...
          },
          …
        ],
        "tables": [
          {
            "page": …,
            "table_index": …,
            "text": "…"
          },
          …
        ]
      }
    
    The class translates the "doc_id", each chunk's "heading" and "text", and each table's "text"
    from the original language to English. If the language is already English (or undetected),
    the file is copied unmodified.
    
    Further improvements could include:
      - More robust sentence splitting (using libraries such as spaCy or nltk).
      - Handling documents that contain mixed languages by detecting and translating per-field.
      - Adding logging and more error handling for production use.
      - Implementing parallel processing for large batches.
    """
    
    def __init__(self, input_dir, output_dir, max_chunk_length=300):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.max_chunk_length = max_chunk_length

        # Mapping from source language to the corresponding Helsinki-NLP model.
        self.models = {
            'fr': 'Helsinki-NLP/opus-mt-fr-en',
            'de': 'Helsinki-NLP/opus-mt-de-en',
            'pl': 'Helsinki-NLP/opus-mt-pl-en',
            'es': 'Helsinki-NLP/opus-mt-es-en',
        }
        self.translators = {}

        # Detect available hardware (CUDA, MPS, or CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
        print(f"Using device: {self.device}")

    def detect_language(self, text):
        """Detects the language of the input text."""
        try:
            lang = detect(text)
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
                torch_dtype=torch.float16 if self.device.type != "cpu" else torch.float32,
                device=self.device
            )
            self.translators[lang] = translator
            return translator
        else:
            print(f"No translation model available for language: {lang}")
            return None

    def chunk_text(self, text):
        """
        Splits text into smaller chunks for translation.
        Uses a simple regex to split on sentences ending with a period.
        """
        sentences = re.split(r'(?<=\.)\s+', text.strip())
        chunks, current_chunk = [], ''
        for sentence in sentences:
            if not sentence.strip():
                continue
            if len(current_chunk.split()) + len(sentence.split()) <= self.max_chunk_length:
                current_chunk = f"{current_chunk} {sentence}".strip() if current_chunk else sentence
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def translate_text(self, text, translator):
        """Translates a string using the provided translator pipeline in chunks."""
        if not text.strip():
            return text
        chunks = self.chunk_text(text)
        translated_chunks = []
        for chunk in chunks:
            translation = translator(chunk, truncation=True)[0]['translation_text']
            translated_chunks.append(translation)
        return "\n\n".join(translated_chunks)

    def translate_json_file(self, input_path, output_path):
        """
        Reads a JSON file, detects its language from its combined text fields,
        translates the "doc_id", each chunk's "heading" and "text", and each table's "text",
        and writes the translated content to a new JSON file.
        """
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Concatenate text from doc_id, chunks, and tables for language detection.
        all_texts = []
        if 'doc_id' in data:
            all_texts.append(data['doc_id'])
        if 'chunks' in data:
            for ch in data['chunks']:
                all_texts.append(ch.get('heading', ''))
                all_texts.append(ch.get('text', ''))
        if 'tables' in data:
            for tb in data['tables']:
                all_texts.append(tb.get('text', ''))
        combined_text = "\n".join(all_texts).strip()
        lang = self.detect_language(combined_text)
        print(f"Detected language for '{input_path}': {lang}")

        if lang == 'en' or not lang:
            print(f"File '{input_path}' is in English or language is unknown. Copying file unmodified.")
            shutil.copy(input_path, output_path)
            return

        translator = self.get_translator(lang)
        if not translator:
            print(f"No translator available for {lang}. Copying file unmodified.")
            shutil.copy(input_path, output_path)
            return

        # Translate "doc_id"
        if 'doc_id' in data and data['doc_id'].strip():
            data['doc_id'] = self.translate_text(data['doc_id'], translator)

        # Translate chunks: headings and texts.
        if 'chunks' in data:
            for ch in data['chunks']:
                if 'heading' in ch and ch['heading'].strip():
                    ch['heading'] = self.translate_text(ch['heading'], translator)
                if 'text' in ch and ch['text'].strip():
                    ch['text'] = self.translate_text(ch['text'], translator)

        # Translate table text.
        if 'tables' in data:
            for tb in data['tables']:
                if 'text' in tb and tb['text'].strip():
                    tb['text'] = self.translate_text(tb['text'], translator)

        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump(data, out_f, indent=2, ensure_ascii=False)
        print(f"Translated JSON saved to '{output_path}'")

        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()

    def translate_documents(self):
        """
        Translates all JSON files in the input directory and saves the translated
        versions in the corresponding structure in the output directory.
        """
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if not file.endswith(".json"):
                    continue
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, self.input_dir)
                output_subdir = os.path.join(self.output_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)
                output_path = os.path.join(output_subdir, file)
                self.translate_json_file(input_path, output_path)

