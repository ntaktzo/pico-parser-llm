import os
import re
import pdfplumber

from langdetect import detect
from transformers import pipeline
import torch
import gc

class PDFProcessor:
    def __init__(self, pdf_path):
        """
        Initializes the PDFProcessor class with the given PDF file path.
        """
        self.pdf_path = pdf_path

    def extract_text(self):
        """
        Extracts text from the initialized PDF file using pdfplumber, 
        with improved handling for layout issues and formatting.
        """
        try:
            all_text = []
            with pdfplumber.open(self.pdf_path) as pdf:
                for page in pdf.pages:
                    raw_text = page.extract_text(x_tolerance=2, y_tolerance=3)
                    if raw_text:
                        cleaned_text = self.remove_header_footer(raw_text)
                        all_text.append(cleaned_text)

            text = "\n".join(all_text)
            text = self.insert_heading_breaks(text)
            text = self.fix_faulty_spacing(text)
            text = self.merge_broken_lines(text)
            text = self.remove_references(text)

            return text

        except Exception as e:
            print(f"Error reading {self.pdf_path}: {e}")
            return None

    @staticmethod
    def remove_references(text):
        """
        Removes references commonly found in academic and medical texts.
        """
        text = re.sub(r"\[\s*[\w\s,.-]+\s*\]", "", text)
        text = re.sub(r"\([A-Za-z,.\s]+\d{4}.*?\)", "", text)
        text = re.sub(r"(?:^|\n)(References|Bibliography|Literature|Bibliografie|Referenties)\s*\n.*",
                      "",
                      text,
                      flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"\s{2,}", " ", text)
        return text

    @staticmethod
    def remove_header_footer(page_text):
        """
        Cleans up boilerplate headers and footers found in PDFs.
        """
        disclaimer_phrases = [
            "© NICE", "All rights reserved", "Subject to Notice of rights",
            "conditions#notice-of-rights", "Contents Overview",
            "PDF aangemaakt op", "REGIONALA CANCERCENTRUM",
            "© Leitlinienprogramm Onkologie", "Langversion", "März 2024",
            "Richtlijnendatabase"
        ]

        lines = page_text.split("\n")
        cleaned_lines = [line for line in lines if not (
            re.match(r"^\s*\d+\s*$", line.strip()) or any(phrase in line for phrase in disclaimer_phrases)
        )]
        return "\n".join(cleaned_lines)

    @staticmethod
    def insert_heading_breaks(text):
        """
        Ensures headings are properly formatted with newlines.
        """
        text_with_markers = re.sub(
            r"(?:(?<=\n)|^)(\d+\.\s[A-Z]+.*|[A-Z\s]+:)",
            r"\n\1",
            text
        )

        def add_newline_before_uppercase(m):
            return "\n" + m.group(1)

        text_with_markers = re.sub(
            r"(?:(?<=\n)|^)([A-Z][A-Z\s]{2,})",
            add_newline_before_uppercase,
            text_with_markers
        )
        return text_with_markers

    @staticmethod
    def fix_faulty_spacing(text):
        """
        Fixes faulty spacing caused by line breaks and OCR issues.
        """
        text = re.sub(r"-\n", "", text)
        text = re.sub(r"-\s+", "", text)
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        text = re.sub(r"(\d)([A-Za-z])", r"\1 \2", text)
        text = re.sub(r"([A-Za-z])(\d)", r"\1 \2", text)
        text = re.sub(r"([a-z]{2,})([A-Z][a-z]+)", r"\1 \2", text)
        common_prefixes = r"\b(in|de|het|met|zonder|te|van|voor|op|aan|als|door|en|of|om|uit|over|bij)\b"
        text = re.sub(r"(\w)" + common_prefixes, r"\1 \2", text)
        return text

    @staticmethod
    def merge_broken_lines(text):
        """
        Merges unnecessarily broken lines while maintaining paragraph integrity.
        """
        lines = text.split("\n")
        merged_lines = []
        buffer_line = ""

        for line in lines:
            line = line.strip()
            if not line:
                if buffer_line:
                    merged_lines.append(buffer_line)
                    buffer_line = ""
                merged_lines.append("")
                continue

            if not buffer_line:
                buffer_line = line
            else:
                if re.search(r"[.!?;:]$", buffer_line):
                    merged_lines.append(buffer_line)
                    buffer_line = line
                else:
                    buffer_line += " " + line

        if buffer_line:
            merged_lines.append(buffer_line)

        return "\n".join(merged_lines)

    @staticmethod
    def process_pdfs(input_dir, output_dir):
        """
        Processes all PDFs in the input directory, extracts text, and saves them as text files.
        """
        print("Processing PDFs...")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith(".pdf"):
                    pdf_path = os.path.join(root, file)
                    processor = PDFProcessor(pdf_path)
                    text = processor.extract_text()
                    if text:
                        relative_path = os.path.relpath(root, input_dir)
                        output_subdir = os.path.join(output_dir, relative_path)
                        os.makedirs(output_subdir, exist_ok=True)
                        output_file = os.path.join(output_subdir, f"{os.path.splitext(file)[0]}.txt")
                        with open(output_file, "w", encoding="utf-8") as f:
                            f.write(text)
                        print(f"Extracted text saved to {output_file}")
                    else:
                        print(f"Failed to extract text from {pdf_path}")




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

