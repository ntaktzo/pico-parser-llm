import os
import re
import json
import statistics
import pdfplumber
from collections import defaultdict
import numpy as np  # Add this to the existing imports


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
        self.global_headings_list = []
        self.page_headings_map = defaultdict(list)

        # Immediately extract and store submission year upon initialization
        self.created_date = self.find_submission_year()

        # Extract country from folder structure
        self.country = self.extract_country_from_path()

        print("--------------------------------------------")
        print(f"Submission year for '{self.pdf_path}': {self.created_date}")
        print(f"Country for '{self.pdf_path}': {self.country}")

    def extract_country_from_path(self):
        """Extracts the country code from the parent directory name."""
        parent_dir = os.path.basename(os.path.dirname(self.pdf_path))
        
        # All EU country codes plus additional European countries and special codes
        country_codes = {
            # EU member states
            "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", 
            "DE", "EL", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", 
            "NL", "PL", "PO", "PT", "RO", "SK", "SI", "ES", "SE",
            
            # Non-EU European countries
            "CH", "NO", "IS", "UK", "GB", "UA", "RS", "ME", "MK", "AL", 
            "BA", "MD", "XK", "LI",
            
            # Special codes and regions
            "EU", "EN", "INT", # INT for international
            
            # Other codes found in your folder structure
            "AE", "TR"
        }
        
        return parent_dir if parent_dir in country_codes else "unknown"

    def find_submission_year(self):
        """
        Finds the submission/report year from the first page of the PDF.
        Stores the year in self.created_date, or 'unknown_year' if none found.
        """
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                if not pdf.pages:
                    return "unknown_year"
                first_page_text = pdf.pages[0].extract_text() or ""
        except Exception:
            return "unknown_year"

        year_pattern = r"\b(19|20)\d{2}\b"

        triggers = [
            "submission date", "submitted on", "report date", "date of submission", "date of issue",
            "soumission", "data di presentazione", "fecha de presentación",
            "datum der einreichung", "fecha de remisión", "submitted:", "issued on", "rapport",
            "published:", "published", "publication date", "date of publication"
        ]

        lines = first_page_text.splitlines()

        for line in lines:
            line_lower = line.lower()
            if any(trigger in line_lower for trigger in triggers):
                years_in_line = re.findall(year_pattern, line)
                if years_in_line:
                    return years_in_line[0]

        all_years = re.findall(r"\b(?:19|20)\d{2}\b", first_page_text)
        if all_years:
            return all_years[0]

        return "unknown_year"
    
    def print_boilerplate_lines(self, boilerplate_normed, normed_line_pages):
        """Logs boilerplate lines that were removed from the PDF."""
        print(f"Boilerplate lines removed from {self.pdf_path}:")
        for normed in boilerplate_normed:
            pages = sorted(normed_line_pages[normed])
            print(f"'{normed}' found on pages: {pages}")

    @staticmethod
    def remove_links(line: str) -> str:
        """Removes URLs and empty parentheses from a given text line."""
        line_no_links = re.sub(r'\(?(?:https?://|www\.)\S+\)?', '', line)
        line_no_links = re.sub(r'\(\s*\)', '', line_no_links)
        return line_no_links.strip()

    @staticmethod
    def advanced_normalize(line: str) -> str:
        """Normalizes text by removing digits and non-alpha characters."""
        norm = line.lower()
        norm = re.sub(r'\d+', '', norm)
        norm = re.sub(r'[^a-z]+', '', norm)
        return norm.strip()

    @staticmethod
    def contains_of_contents(line: str) -> bool:
        """Checks if the line contains 'of contents', indicating a table of contents entry."""
        return bool(re.search(r'\bof\s+contents\b', line, re.IGNORECASE))

    @staticmethod
    def is_toc_dotline(line: str) -> bool:
        """Checks if a line is a dotted table of contents entry."""
        return bool(re.search(r'\.{5,}\s*\d+$', line))

    @staticmethod
    def is_footnote_source(line: str) -> bool:
        """Determines if the line is a reference or footnote source."""
        stripped = line.strip()
        if not stripped:
            return False

        if not re.match(r'^(\[\d+\]|[\d\.]+)\b', stripped):
            return False

        reference_patterns = [r'et al\.', r'Disponible en ligne', r'consulté le', r'NEJM', r'PubMed', r'doi']
        combined = '(' + '|'.join(reference_patterns) + ')'
        return bool(re.search(combined, stripped, flags=re.IGNORECASE))

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
        current_group = raw_lines[0] if raw_lines else []

        for next_line in raw_lines[1:]:
            if not current_group:
                current_group = next_line
                continue
                
            current_avg_size = statistics.mean([w['size'] for w in current_group if 'size' in w])
            next_avg_size = statistics.mean([w['size'] for w in next_line if 'size' in w])
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
        Extract tables from PDF, handling both single and multi-column layouts.
        For each table, find an appropriate heading.
        """
        tables_info = []
        
        for page_num, page in enumerate(pdf.pages, start=1):
            # Attempt to detect columns
            num_columns, column_boundaries = self.detect_columns(page)
            
            # Try to extract tables from the whole page first (for tables spanning columns)
            page_tables = page.extract_tables()
            
            if page_tables:
                # Process whole-page tables (might span columns)
                for i, table_data in enumerate(page_tables, start=1):
                    flattened = self.flatten_table(table_data)
                    if not flattened.strip():
                        continue
                    
                    # Find a suitable heading
                    table_title = self.find_table_heading(page_num, i)
                    
                    tables_info.append({
                        "page": page_num,
                        "heading": table_title,
                        "text": flattened
                    })
            
            # For multi-column layouts, also try finding tables in each column
            if num_columns > 1:
                for i in range(num_columns):
                    left_bound = column_boundaries[i]
                    right_bound = column_boundaries[i+1]
                    
                    # Extract tables only within this column's boundaries
                    column_area = (left_bound, 0, right_bound, page.height)
                    column = page.crop(column_area)
                    column_tables = column.extract_tables()
                    
                    for j, table_data in enumerate(column_tables, start=1):
                        # Skip if already found in whole-page extraction
                        flattened = self.flatten_table(table_data)
                        if not flattened.strip():
                            continue
                            
                        # Check if this is a duplicate of a table we already found
                        if any(t.get("text", "") == flattened for t in tables_info if t["page"] == page_num):
                            continue
                        
                        # Find a suitable heading
                        table_title = self.find_table_heading(page_num, i*10+j, column_index=i)
                        
                        tables_info.append({
                            "page": page_num,
                            "heading": table_title,
                            "text": flattened
                        })
        
        return tables_info

    @staticmethod
    def flatten_table(table):
        """
        Converts a table (list of lists) into lines where each row is:
            Row X: cell1|cell2|cell3
        with no extra padding around the '|'.
        """
        if not table:
            return ""

        # Clean each cell in the table (remove newlines and extra spaces)
        cleaned_table = []
        for row in table:
            cleaned_row = []
            for cell in row:
                if cell is None:
                    cell = ""
                # Replace any sequence of whitespace (including newlines) with a single space
                cleaned_cell = re.sub(r'\s+', ' ', str(cell)).strip()
                cleaned_row.append(cleaned_cell)
            cleaned_table.append(cleaned_row)

        # Build lines with a Row index
        lines = []
        for i, row in enumerate(cleaned_table, start=1):
            row_str = "|".join(row)
            lines.append(f"Row {i}: {row_str}")

        return "\n".join(lines)

    def extract_preliminary_chunks(self):
        """
        Main function that:
          1. Extracts text and identifies headings from each page (skipping footnotes, boilerplate, etc.).
          2. Stores headings per page in self.page_headings_map.
          3. Extracts tables, tries to detect table titles, and inserts them as separate chunks.
          4. Returns a dictionary:
               {
                 "doc_id": self.doc_id,
                 "chunks": [ { "heading": ..., "text": ..., "start_page": ..., "end_page": ...}, ... ]
               }
        """
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                # We'll first gather all lines/heading info (without handling tables)
                num_pages = len(pdf.pages)
                normed_line_pages = {}
                pages_with_lines = []

                # Pass 1: gather potential headings and normal lines, handling columns
                for i, page in enumerate(pdf.pages, start=1):
                    # First detect headings using font information
                    line_objs = self.detect_headings_by_font(page)
                    
                    # Create a mapping of heading text to heading status
                    likely_headings = {}
                    for lo in line_objs:
                        likely_headings[lo["text"]] = lo.get("likely_heading", False)
                    
                    # Extract text column by column
                    column_texts = self.extract_text_by_columns(page)
                    
                    # Track all lines with their page numbers for boilerplate detection
                    all_lines = []
                    
                    # Process each column
                    for col_idx, column_text in enumerate(column_texts):
                        # Split column text into lines
                        col_lines = column_text.split('\n')
                        
                        # Process each line for headings and content
                        for line in col_lines:
                            if not line.strip():
                                continue
                                
                            # Create a simple line object
                            line_obj = {
                                "text": line.strip(),
                                "likely_heading": False
                            }
                            
                            # Check if this line matches any of our detected headings
                            for heading_text, is_heading in likely_headings.items():
                                if heading_text in line and is_heading:
                                    line_obj["likely_heading"] = True
                                    break
                            
                            all_lines.append(line_obj)
                        
                    pages_with_lines.append((i, all_lines))
                    
                    # Identify potential boilerplate lines
                    unique_normed = set()
                    for lo in all_lines:
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

                # Filter out footnotes and boilerplate
                for page_num, line_objs in pages_with_lines:
                    for lo in line_objs:
                        line_text = lo["text"]
                        if not line_text.strip():
                            continue

                        if in_footnote:
                            # If we hit a new heading or an empty line, end footnote consumption
                            if not line_text.strip() or lo["likely_heading"]:
                                in_footnote = False
                                continue
                            else:
                                continue

                        # If line looks like a footnote reference, skip
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

                # Build text chunks, track headings
                chunks = []
                buffer = []
                current_heading = ""

                def flush_buffer():
                    """Dump the buffered lines into a chunk if any."""
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
                        # Before we switch heading, flush existing buffer
                        flush_buffer()
                        current_heading = text_line
                        self.global_headings_list.append((pnum, text_line))
                        self.page_headings_map[pnum].append(text_line)
                    else:
                        buffer.append((pnum, text_line))

                flush_buffer()

                # Now extract tables with knowledge of headings
                tables_info = self.extract_tables_from_pdf(pdf)

                # Insert each table as its own chunk
                for tinfo in tables_info:
                    pg = tinfo["page"]
                    heading_for_table = tinfo["heading"]
                    table_text = tinfo["text"]
                    chunks.append({
                        "heading": heading_for_table,
                        "text": table_text,
                        "start_page": pg,
                        "end_page": pg
                    })

                # Return the final structure
                return {
                    "doc_id": self.doc_id,
                    "created_date": self.created_date,
                    "country:": self.country,
                    "chunks": chunks
                }

        except Exception as e:
            print(f"Error reading {self.pdf_path}: {e}")
            return {"doc_id": self.doc_id, "created_date": self.created_date, "chunks": []}

    def detect_columns(self, page):
        """
        Detect if a page has multiple columns by analyzing text positions.
        Returns the number of columns and their x-boundaries.
        """
        import numpy as np
        words = page.extract_words(x_tolerance=2, y_tolerance=2)
        if not words:
            return 1, []  # Default to single column if no words
            
        # Collect x-positions (horizontal position) of words
        x_positions = [word['x0'] for word in words]
        
        # Identify potential column gaps using histogram analysis
        hist, bin_edges = np.histogram(x_positions, bins=20)
        
        # Find significant gaps in word positions
        significant_gaps = []
        for i in range(len(hist)):
            if hist[i] < max(hist) * 0.1:  # Threshold for considering a gap
                left_edge = bin_edges[i]
                right_edge = bin_edges[i+1]
                middle = (left_edge + right_edge) / 2
                significant_gaps.append(middle)
        
        # Determine number of columns based on gaps
        if len(significant_gaps) == 0:
            return 1, []  # Single column
        elif len(significant_gaps) == 1:
            # Likely a two-column layout
            # Determine boundaries between columns
            column_boundaries = [0, significant_gaps[0], page.width]
            return 2, column_boundaries
        else:
            # Multi-column layout (more than two)
            # Sort gaps and create column boundaries
            significant_gaps.sort()
            column_boundaries = [0] + significant_gaps + [page.width]
            return len(column_boundaries) - 1, column_boundaries

    def extract_text_by_columns(self, page):
        """
        Extract text from a page considering column layout.
        Returns text organized by columns and preserving reading order.
        """
        num_columns, column_boundaries = self.detect_columns(page)
        
        if num_columns == 1:
            # For single column, just extract text normally
            return [page.extract_text()]
        
        # For multi-column layout, extract text for each column separately
        column_texts = []
        
        for i in range(num_columns):
            left_bound = column_boundaries[i]
            right_bound = column_boundaries[i+1]
            
            # Extract text only within this column's boundaries
            column_area = (left_bound, 0, right_bound, page.height)
            column_text = page.crop(column_area).extract_text()
            
            if column_text.strip():
                column_texts.append(column_text)
        
        return column_texts

    def find_table_heading(self, page_num, table_index, column_index=None):
        """
        Find an appropriate heading for a table based on nearby headings.
        Takes into account the table's column if provided.
        """
        # Look for headings on this page that contain the word 'table'
        table_headings = []
        for h in self.page_headings_map[page_num]:
            if re.search(r'table', h, re.IGNORECASE):
                table_headings.append(h)
        
        if table_headings:
            # Use the last table heading found
            return table_headings[-1]
        
        # If no specific table heading, use the last heading on this page
        if self.page_headings_map[page_num]:
            fallback_heading = self.page_headings_map[page_num][-1]
            if column_index is not None:
                return f"table under heading <{fallback_heading}> (column {column_index+1})"
            else:
                return f"table under heading <{fallback_heading}>"
        
        # If no heading on this page, look at previous pages
        for prev_page in range(page_num - 1, 0, -1):
            if self.page_headings_map[prev_page]:
                last_heading = self.page_headings_map[prev_page][-1]
                if column_index is not None:
                    return f"table under heading <{last_heading}> (from previous page, column {column_index+1})"
                else:
                    return f"table under heading <{last_heading}> (from previous page)"
        
        # No heading found
        if column_index is not None:
            return f"table under heading <none> (column {column_index+1})"
        else:
            return "table under heading <none>"

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

        processed_files = 0
        errors = 0
        
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(".pdf"):
                    pdf_path = os.path.join(root, file)
                    try:
                        # Create processor and extract content
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
                            # Get exact relative path from input directory
                            rel_path = os.path.relpath(root, input_dir)
                            
                            # Create identical folder structure in output directory
                            output_subdir = os.path.join(output_dir, rel_path)
                            os.makedirs(output_subdir, exist_ok=True)

                            # Output file path with same name as input (plus _cleaned)
                            output_file = os.path.join(
                                output_subdir,
                                f"{os.path.splitext(file)[0]}_cleaned.json"
                            )
                            
                            with open(output_file, "w", encoding="utf-8") as f:
                                json.dump(result, f, indent=2, ensure_ascii=False)
                                
                            processed_files += 1
                            print(f"Successfully processed: {pdf_path} -> {output_file}")
                        else:
                            errors += 1
                            print(f"Warning: No chunks extracted from {pdf_path}")
                    except Exception as e:
                        errors += 1
                        print(f"Error processing {pdf_path}: {e}")
        
        print(f"Processing complete. Successfully processed {processed_files} files with {errors} errors.")







import os
import re
import json
import shutil
import gc
import torch
from typing import Optional
from langdetect import detect
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

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
    
    The class translates each chunk's "heading" and "text", and each table's "text"
    from the original language to English. If the language is already English (or undetected),
    the file is copied unmodified.
    """
    
    def __init__(self, input_dir, output_dir, max_chunk_length=300):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.max_chunk_length = max_chunk_length

        # Mapping from source language to the corresponding Helsinki-NLP model.
        # Covering major European languages with direct models
        self.models = {
            'fr': 'Helsinki-NLP/opus-mt-fr-en',
            'de': 'Helsinki-NLP/opus-mt-de-en',
            'pl': 'Helsinki-NLP/opus-mt-pl-en',
            'es': 'Helsinki-NLP/opus-mt-es-en',
            'it': 'Helsinki-NLP/opus-mt-it-en',
            'nl': 'Helsinki-NLP/opus-mt-nl-en',
            'da': 'Helsinki-NLP/opus-mt-da-en',
            'fi': 'Helsinki-NLP/opus-mt-fi-en',
            'sv': 'Helsinki-NLP/opus-mt-sv-en',
            'cs': 'Helsinki-NLP/opus-mt-cs-en',
            'el': 'Helsinki-NLP/opus-mt-tc-big-el-en',
            'hu': 'Helsinki-NLP/opus-mt-hu-en',
            'bg': 'Helsinki-NLP/opus-mt-bg-en',
            'sk': 'Helsinki-NLP/opus-mt-sk-en',
            'et': 'Helsinki-NLP/opus-mt-et-en',
            'lv': 'Helsinki-NLP/opus-mt-lv-en',
            'lt': 'Helsinki-NLP/opus-mt-tc-big-lt-en',
            'mt': 'Helsinki-NLP/opus-mt-mt-en',
        }
        
        # Define language groups for the fallback Helsinki group models
        self.language_groups = {
            'facebook/nllb-200-distilled-600M': {
                'model_type': 'nllb',
                'langs': set([
                    'af', 'am', 'ar', 'as', 'az', 'be', 'bg', 'bn', 'br', 'bs', 'ca', 'cs', 'cy', 'da', 
                    'de', 'el', 'en', 'es', 'et', 'fa', 'ff', 'fi', 'fr', 'fy', 'ga', 'gd', 'gl', 'gu', 
                    'ha', 'he', 'hi', 'hr', 'hu', 'hy', 'id', 'ig', 'is', 'it', 'ja', 'jv', 'ka', 'kk', 
                    'km', 'kn', 'ko', 'ku', 'ky', 'lb', 'lg', 'ln', 'lo', 'lt', 'lv', 'mg', 'mk', 'ml', 
                    'mn', 'mr', 'ms', 'mt', 'my', 'ne', 'nl', 'no', 'ns', 'ny', 'om', 'or', 'pa', 'pl', 
                    'ps', 'pt', 'ro', 'ru', 'sd', 'si', 'sk', 'sl', 'so', 'sq', 'sr', 'ss', 'su', 'sv', 
                    'sw', 'ta', 'te', 'th', 'tl', 'tn', 'tr', 'uk', 'ur', 'uz', 'vi', 'wo', 'xh', 'yi', 
                    'yo', 'zh', 'zu'
                ])
            }
        }
        
        # Map ISO language codes to NLLB format
        self.nllb_lang_map = {
            'en': 'eng_Latn', 'fr': 'fra_Latn', 'de': 'deu_Latn', 'es': 'spa_Latn', 
            'it': 'ita_Latn', 'pt': 'por_Latn', 'nl': 'nld_Latn', 'pl': 'pol_Latn',
            'ru': 'rus_Cyrl', 'zh': 'zho_Hans', 'ja': 'jpn_Jpan', 'ar': 'ara_Arab',
            'hi': 'hin_Deva', 'bg': 'bul_Cyrl', 'cs': 'ces_Latn', 'da': 'dan_Latn',
            'fi': 'fin_Latn', 'el': 'ell_Grek', 'hu': 'hun_Latn', 'ro': 'ron_Latn',
            'sk': 'slk_Latn', 'sl': 'slv_Latn', 'sv': 'swe_Latn', 'uk': 'ukr_Cyrl',
            'hr': 'hrv_Latn', 'no': 'nno_Latn', 'et': 'est_Latn', 'lv': 'lav_Latn',
            'lt': 'lit_Latn', 'tr': 'tur_Latn', 'he': 'heb_Hebr', 'th': 'tha_Thai',
            'ko': 'kor_Hang', 'vi': 'vie_Latn', 'fa': 'fas_Arab', 'sr': 'srp_Cyrl'
        }
        
        self.translators = {}
        self.multilingual_models = {}

        # Detect available hardware (CUDA, MPS, or CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def detect_language(self, text: str) -> Optional[str]:
        """
        Detects the language of the input text.
        Returns a two-letter language code if successful, otherwise None.
        """
        try:
            lang = detect(text)
            return lang
        except Exception as e:
            print(f"Language detection failed: {e}")
            return None

    def get_translator(self, lang: str):
        """
        Returns a translation function for the given language code.
        Tries different model options in order of preference.
        """
        # Return cached translator if already loaded
        if lang in self.translators:
            return self.translators[lang]

        # 1. Try direct Helsinki-NLP model first (language-specific)
        if lang in self.models:
            model_name = self.models[lang]
            print(f"Using model: {model_name}")
            
            try:
                translator = pipeline(
                    "translation",
                    model=model_name,
                    torch_dtype=torch.float16 if self.device.type != "cpu" else torch.float32,
                    device=self.device
                )
                self.translators[lang] = translator
                return translator
            except Exception as e:
                print(f"Failed to load model {model_name}: {e}")
                # Continue to next option
        
        # 2. Try NLLB multilingual model as fallback
        nllb_model = 'facebook/nllb-200-distilled-600M'
        if lang in self.language_groups[nllb_model]['langs']:
            print(f"Using model: {nllb_model}")
            
            # Load or retrieve cached NLLB model
            if nllb_model not in self.multilingual_models:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(nllb_model)
                    model = AutoModelForSeq2SeqLM.from_pretrained(nllb_model).to(self.device)
                    self.multilingual_models[nllb_model] = (model, tokenizer)
                except Exception as e:
                    print(f"Failed to load multilingual model {nllb_model}: {e}")
                    return None
            else:
                model, tokenizer = self.multilingual_models[nllb_model]
            
            # Get NLLB-formatted language codes
            src_lang = self.nllb_lang_map.get(lang, f"{lang}_Latn")  # Fallback to Latin script
            tgt_lang = 'eng_Latn'  # Always translate to English
            
            # Create translator function
            def nllb_translate(text, **kwargs):
                # Set the source language
                inputs = tokenizer(text, return_tensors="pt").to(self.device)
                
                # Get the tokenizer's language ID for the target language
                with torch.no_grad():
                    translated_tokens = model.generate(
                        **inputs,
                        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
                        max_length=512
                    )
                
                translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                return [{'translation_text': translation}]
            
            self.translators[lang] = nllb_translate
            return nllb_translate

        # No suitable model found
        return None

    def chunk_text(self, text: str) -> list:
        """
        Splits text into smaller chunks for translation.
        Uses a simple regex to split on sentences ending with a period.
        """
        sentences = re.split(r'(?<=\.)\s+', text.strip())
        chunks, current_chunk = [], ''
        for sentence in sentences:
            if not sentence.strip():
                continue
            # If adding this sentence doesn't exceed max_chunk_length, add it to current chunk
            if len(current_chunk.split()) + len(sentence.split()) <= self.max_chunk_length:
                current_chunk = f"{current_chunk} {sentence}".strip() if current_chunk else sentence
            else:
                # Current chunk is full, start a new one
                chunks.append(current_chunk.strip())
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def translate_text(self, text: str, translator) -> str:
        """
        Translates a string using the provided translator function in chunks.
        Returns the translated text.
        """
        if not text.strip():
            return text
        chunks = self.chunk_text(text)
        translated_chunks = []
        for chunk in chunks:
            translation = translator(chunk, truncation=True)[0]['translation_text']
            translated_chunks.append(translation)
        return "\n\n".join(translated_chunks)

    def translate_json_file(self, input_path: str, output_path: str):
        """
        Reads a JSON file, detects its language from its combined text fields,
        translates each chunk's 'heading' and 'text', as well as each table's 'text',
        and writes the translated content to a new JSON file.
        
        If the document is already English or undetected, or if no translator is available
        for that language, the file is copied unmodified.
        """
        # Get document name and parent folder
        file_name = os.path.basename(input_path)
        parent_folder = os.path.basename(os.path.dirname(input_path))
        
        # Print document info
        print(f"Document: {file_name} (in folder: {parent_folder})")
        
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Concatenate text from doc_id, chunks, and tables for language detection.
        all_texts = []
        if 'doc_id' in data:
            all_texts.append(data['doc_id'])  # doc_id is used for detection only (not translated!)
        if 'chunks' in data:
            for ch in data['chunks']:
                all_texts.append(ch.get('heading', ''))
                all_texts.append(ch.get('text', ''))
        if 'tables' in data:
            for tb in data['tables']:
                all_texts.append(tb.get('text', ''))

        combined_text = "\n".join([t for t in all_texts if t]).strip()
        lang = self.detect_language(combined_text)
        print(f"Detected language: {lang}")

        # If the document is already English, or detection is uncertain, just copy it unmodified
        if lang == 'en' or not lang:
            print("No translation needed (English or undetected language)")
            shutil.copy(input_path, output_path)
            return

        translator = self.get_translator(lang)
        
        if not translator:
            print(f"No translator available for {lang}. Copying file unmodified.")
            shutil.copy(input_path, output_path)
            return

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

        # Save the translated JSON
        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump(data, out_f, indent=2, ensure_ascii=False)

        # Free up GPU memory if applicable
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