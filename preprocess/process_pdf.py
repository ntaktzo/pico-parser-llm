import os
import re
import pdfplumber

from langdetect import detect
from transformers import pipeline
import torch
import gc

import pdfplumber
import re
import pandas as pd
import os

class PDFProcessor:
    def __init__(self, pdf_path):
        """
        Initializes the PDFProcessor class with the given PDF file path.
        """
        self.pdf_path = pdf_path

    def extract_text(self):
        """
        Extracts text from the PDF file.
        Single-column assumption.
        1) Single pass pdfplumber extract.
        2) Minimal disclaimers removal.
        3) Flatten tables with explicit newlines.
        4) Fix spacing and merge lines.
        5) Remove references.
        6) Chunk by headings + paragraphs.
        """
        try:
            all_pages = []

            with pdfplumber.open(self.pdf_path) as pdf:
                for page in pdf.pages:
                    raw_text = page.extract_text(x_tolerance=2, y_tolerance=2)
                    if raw_text is None:
                        raw_text = ""

                    cleaned_page = self.remove_boilerplate(raw_text)

                    page_tables = page.extract_tables()
                    tables_text = []
                    if page_tables:
                        for tbl in page_tables:
                            tables_text.append(self.flatten_table(tbl))

                    combined = cleaned_page.strip()
                    if tables_text:
                        combined += "\n\n" + "\n\n".join(tables_text)

                    if combined.strip():
                        all_pages.append(combined)

            text = "\n".join(all_pages)

            chunks = self.logical_chunking(text)

            return chunks
        except Exception as e:
            print(f"Error reading {self.pdf_path}: {e}")
            return None

    @staticmethod
    def remove_boilerplate(text):
        disclaimers = [
            "© NICE", "All rights reserved", "Subject to Notice", "Page ",
            "REGIONALA CANCERCENTRUM", "Richtlijnendatabase"
        ]
        lines = text.split("\n")
        kept = []
        for line in lines:
            if not any(d.lower() in line.lower() for d in disclaimers):
                kept.append(line)
        return "\n".join(kept)

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
    def is_likely_heading(prev_line, line, next_line):
        line_stripped = line.strip()

        # Strong numeric heading pattern (e.g., 3., 3.1, 07.1.1)
        strong_numbered_heading = bool(re.match(r'^(\d+(\.\d+)*\.?)\s+[A-Z]', line_stripped))

        # Short line ending with colon, likely heading
        short_colon_ending = len(line_stripped.split()) <= 8 and line_stripped.endswith(':')

        # Isolated line surrounded by blank lines and uppercase dominance
        prev_blank = not prev_line.strip()
        next_blank = not next_line.strip()
        letters = [ch for ch in line_stripped if ch.isalpha()]
        uppercase_ratio = sum(ch.isupper() for ch in letters) / len(letters) if letters else 0
        isolated_uppercase = uppercase_ratio > 0.75 and prev_blank and next_blank and len(line_stripped.split()) < 10

        return strong_numbered_heading or short_colon_ending or isolated_uppercase

    @staticmethod
    def logical_chunking(text):
        lines = text.split('\n')
        chunks = []
        buffer = []

        def flush_buffer():
            if buffer:
                chunk = ' '.join(buffer).strip()
                if chunk:
                    chunks.append(chunk)
                buffer.clear()

        padded_lines = [""] + lines + [""]  # padding to avoid index errors
        for i in range(1, len(padded_lines) - 1):
            line = padded_lines[i]
            prev_line = padded_lines[i - 1]
            next_line = padded_lines[i + 1]

            if not line.strip():
                flush_buffer()
                continue

            if PDFProcessor.is_likely_heading(prev_line, line, next_line):
                flush_buffer()
                chunks.append(line.strip())
            else:
                buffer.append(line.strip())

        flush_buffer()

        return [c for c in chunks if c.strip()]

    @staticmethod
    def process_pdfs(input_dir, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(".pdf"):
                    pdf_path = os.path.join(root, file)
                    processor = PDFProcessor(pdf_path)
                    chunks = processor.extract_text()
                    if chunks:
                        relative_path = os.path.relpath(root, input_dir)
                        out_dir = os.path.join(output_dir, relative_path)
                        os.makedirs(out_dir, exist_ok=True)

                        output_file = os.path.join(out_dir, f"{os.path.splitext(file)[0]}.txt")
                        with open(output_file, "w", encoding="utf-8") as f:
                            f.write("\n\n".join(chunks))
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

