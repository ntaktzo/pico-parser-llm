import os
import re
import json
import statistics
import pdfplumber
from collections import defaultdict
import numpy as np  # Add this to the existing imports
from typing import Dict, Any, Optional, Union, List


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

        # Extract source type 
        self.source_type = self.extract_source_type_from_path()

        # Medical terms that should NOT be translated
        self.preserve_terms = {
            # Drug names
            'sorafenib', 'lenvatinib', 'sotorasib', 'atezolizumab', 
            'bevacizumab', 'tecentriq', 'nexavar', 'lenvima', 'lumykras',
            'imjudo', 'sandoz',
            
            # Medical abbreviations
            'HCC', 'NSCLC', 'KRAS', 'G12C', 'PFS', 'OS', 'ORR', 'DCR',
            'ECOG', 'BCLC', 'Child-Pugh', 'mRECIST', 'RECIST',
            
            # Clinical terms
            'hepatocellular carcinoma', 'non-small cell lung cancer',
            'progression-free survival', 'overall survival'
        }

        print("--------------------------------------------")
        print(f"Source type for '{self.pdf_path}': {self.source_type}")
        print(f"Submission year for '{self.pdf_path}': {self.created_date}")
        print(f"Country for '{self.pdf_path}': {self.country}")


    def preserve_medical_terms(self, text: str) -> tuple[str, dict]:
        """Replace medical terms with placeholders before translation."""
        preserved = {}
        modified_text = text
        
        for i, term in enumerate(self.preserve_terms):
            if term.lower() in text.lower():
                placeholder = f"__MEDICAL_TERM_{i}__"
                # Case-insensitive replacement
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                modified_text = pattern.sub(placeholder, modified_text)
                preserved[placeholder] = term
                
        return modified_text, preserved

    # Modify python/process.py - PDFProcessor class
    def extract_source_type_from_path(self):
        """Identifies whether this is an HTA submission or clinical guideline."""
        path_parts = self.pdf_path.lower().split(os.sep)
        
        if "hta submission" in self.pdf_path.lower() or "hta submissions" in self.pdf_path.lower():
            return "hta_submission"
        elif "clinical guideline" in self.pdf_path.lower() or "clinical guidelines" in self.pdf_path.lower():
            return "clinical_guideline"
        else:
            return "unknown"

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
        Enhanced with better error handling for problematic PDFs.
        """
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                # We'll first gather all lines/heading info (without handling tables)
                num_pages = len(pdf.pages)
                normed_line_pages = {}
                pages_with_lines = []
                problematic_pages = 0
                total_pages_processed = 0

                # Pass 1: gather potential headings and normal lines, handling columns
                for i, page in enumerate(pdf.pages, start=1):
                    try:
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
                        total_pages_processed += 1
                        
                        # Identify potential boilerplate lines
                        unique_normed = set()
                        for lo in all_lines:
                            norm = self.advanced_normalize(lo["text"])
                            if norm:
                                unique_normed.add(norm)
                        for n in unique_normed:
                            normed_line_pages.setdefault(n, set()).add(i)
                    
                    except ValueError as page_err:
                        # Check specifically for negative width/height error
                        if "negative width or height" in str(page_err):
                            print(f"Warning: Skipping page {i} due to layout issues")
                            problematic_pages += 1
                            continue
                        else:
                            raise page_err
                    except Exception as page_err:
                        print(f"Warning: Error processing page {i}: {page_err}")
                        problematic_pages += 1
                        continue

                # If too many pages were problematic, switch to fallback method
                if problematic_pages > 0 and (total_pages_processed == 0 or problematic_pages / num_pages > 0.2):
                    print(f"Too many problematic pages ({problematic_pages}/{num_pages}). Switching to fallback method.")
                    return self.extract_using_fallback()

                # If we couldn't extract any content, try the fallback method
                if not pages_with_lines:
                    print(f"No content extracted from {self.pdf_path}. Switching to fallback method.")
                    return self.extract_using_fallback()

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
                    "country": self.country,
                    "source_type": self.source_type,
                    "chunks": chunks
                }

        except Exception as e:
            print(f"Error reading {self.pdf_path}: {e}")
            # Try fallback method if primary extraction fails
            return self.extract_using_fallback()


    def extract_using_fallback(self):
        """
        Fallback method for problematic PDFs that can't be processed normally.
        Uses PyPDF2 for more reliable text extraction when pdfplumber fails.
        """
        try:
            import PyPDF2
            
            print(f"Using fallback extraction for {self.pdf_path}")
            
            # Create a basic document structure
            doc_structure = {
                "doc_id": self.doc_id,
                "created_date": self.created_date,
                "country": self.country,
                "source_type": self.source_type,
                "chunks": []
            }
            
            # Open with PyPDF2 as a more robust alternative
            with open(self.pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                print(f"PDF has {len(reader.pages)} pages")
                
                # Extract text page by page
                current_chunk_text = []
                current_heading = "Introduction"
                start_page = 1
                
                # Process each page
                for page_num, page in enumerate(reader.pages, 1):
                    try:
                        text = page.extract_text()
                        
                        # Skip empty pages
                        if not text or not text.strip():
                            continue
                        
                        # Check for potential section headings
                        lines = text.split('\n')
                        potential_heading = None
                        
                        for line in lines[:5]:  # Look at first few lines for potential headings
                            clean_line = line.strip()
                            if not clean_line:
                                continue
                                
                            # Simple heuristic for headings: short lines that are capitalized or numbered
                            if (len(clean_line.split()) <= 7 and 
                                (clean_line.isupper() or 
                                (any(char.isdigit() for char in clean_line[:3]) and 
                                not clean_line.lower().startswith("page")))):
                                potential_heading = clean_line
                                break
                        
                        # If we found a potential heading, start a new chunk
                        if potential_heading and len(current_chunk_text) > 0:
                            # Save the previous chunk
                            combined_text = "\n\n".join(current_chunk_text)
                            doc_structure["chunks"].append({
                                "heading": current_heading,
                                "text": combined_text,
                                "start_page": start_page,
                                "end_page": page_num - 1
                            })
                            
                            # Start a new chunk
                            current_chunk_text = [text]
                            current_heading = potential_heading
                            start_page = page_num
                        else:
                            # Continue with current chunk
                            current_chunk_text.append(text)
                        
                        # Create a new chunk every few pages if no natural breaks found
                        if page_num - start_page >= 4 and not potential_heading:
                            combined_text = "\n\n".join(current_chunk_text)
                            doc_structure["chunks"].append({
                                "heading": current_heading,
                                "text": combined_text,
                                "start_page": start_page,
                                "end_page": page_num
                            })
                            
                            # Reset for next chunk
                            current_chunk_text = []
                            current_heading = f"Section starting on page {page_num + 1}"
                            start_page = page_num + 1
                            
                    except Exception as page_error:
                        print(f"Warning: Error processing page {page_num} in fallback method: {page_error}")
                        continue
                
                # Add any remaining text
                if current_chunk_text:
                    combined_text = "\n\n".join(current_chunk_text)
                    doc_structure["chunks"].append({
                        "heading": current_heading,
                        "text": combined_text,
                        "start_page": start_page,
                        "end_page": len(reader.pages)
                    })
            
            print(f"Fallback extraction completed: created {len(doc_structure['chunks'])} chunks")
            return doc_structure
            
        except Exception as fallback_error:
            print(f"Fallback extraction also failed: {fallback_error}")
            # Return empty structure if all methods fail
            return {
                "doc_id": self.doc_id,
                "created_date": self.created_date,
                "country": self.country,
                "source_type": self.source_type,
                "chunks": []
            }

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
from typing import Optional, List, Dict, Any
from langdetect import detect
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

import os
import json
import shutil
import gc
import torch
from typing import Optional
from langdetect import detect
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

import os
import re
import json
import shutil
import gc
import torch
from typing import Optional, List, Dict, Any
from langdetect import detect
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

class Translator:
    """
    Enhanced Translator class with robust CUDA handling, medical term preservation,
    and specialized table handling for HTA documents.
    """

    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.english_chunks_preserved = 0
        self.chunks_translated = 0

        # Medical terms that should NOT be translated
        self.preserve_terms = {
            # Drug names
            'sorafenib', 'lenvatinib', 'sotorasib', 'atezolizumab', 
            'bevacizumab', 'tecentriq', 'nexavar', 'lenvima', 'lumykras',
            'imjudo', 'sandoz',
            
            # Medical abbreviations
            'HCC', 'NSCLC', 'KRAS', 'G12C', 'PFS', 'OS', 'ORR', 'DCR',
            'ECOG', 'BCLC', 'Child-Pugh', 'mRECIST', 'RECIST',
            
            # Clinical terms
            'hepatocellular carcinoma', 'non-small cell lung cancer',
            'progression-free survival', 'overall survival'
        }

        # Available Helsinki-NLP models (language-specific, better quality)
        self.helsinki_models = {
            'fr': 'Helsinki-NLP/opus-mt-fr-en',
            'de': 'Helsinki-NLP/opus-mt-de-en',
            'es': 'Helsinki-NLP/opus-mt-es-en',
            'it': 'Helsinki-NLP/opus-mt-it-en',
            'nl': 'Helsinki-NLP/opus-mt-nl-en',
            'pl': 'Helsinki-NLP/opus-mt-pl-en',
            'pt': 'Helsinki-NLP/opus-mt-tc-big-pt-en',
            'ru': 'Helsinki-NLP/opus-mt-ru-en',
            'da': 'Helsinki-NLP/opus-mt-da-en',
            'sv': 'Helsinki-NLP/opus-mt-sv-en',
            'no': 'Helsinki-NLP/opus-mt-no-en',
            'fi': 'Helsinki-NLP/opus-mt-fi-en',
            'cs': 'Helsinki-NLP/opus-mt-cs-en',
            'hu': 'Helsinki-NLP/opus-mt-hu-en',
            'bg': 'Helsinki-NLP/opus-mt-bg-en',
            'sk': 'Helsinki-NLP/opus-mt-sk-en',
            'sl': 'Helsinki-NLP/opus-mt-sl-en',
            'hr': 'Helsinki-NLP/opus-mt-hr-en',
            'et': 'Helsinki-NLP/opus-mt-et-en',
            'lv': 'Helsinki-NLP/opus-mt-lv-en',
            'lt': 'Helsinki-NLP/opus-mt-tc-big-lt-en',
            'el': 'Helsinki-NLP/opus-mt-tc-big-el-en',
            'ro': 'Helsinki-NLP/opus-mt-ro-en',
            'tr': 'Helsinki-NLP/opus-mt-tr-en',
        }

        # Language mapping for Facebook NLLB model
        self.nllb_lang_mapping = {
            'en': 'eng_Latn', 'fr': 'fra_Latn', 'de': 'deu_Latn', 'es': 'spa_Latn',
            'it': 'ita_Latn', 'pt': 'por_Latn', 'nl': 'nld_Latn', 'pl': 'pol_Latn',
            'ru': 'rus_Cyrl', 'zh': 'zho_Hans', 'ja': 'jpn_Jpan', 'ar': 'ara_Arab',
            'hi': 'hin_Deva', 'bg': 'bul_Cyrl', 'cs': 'ces_Latn', 'da': 'dan_Latn',
            'fi': 'fin_Latn', 'el': 'ell_Grek', 'hu': 'hun_Latn', 'ro': 'ron_Latn',
            'sk': 'slk_Latn', 'sl': 'slv_Latn', 'sv': 'swe_Latn', 'uk': 'ukr_Cyrl',
            'hr': 'hrv_Latn', 'no': 'nno_Latn', 'et': 'est_Latn', 'lv': 'lav_Latn',
            'lt': 'lit_Latn', 'tr': 'tur_Latn', 'he': 'heb_Hebr', 'th': 'tha_Thai',
            'ko': 'kor_Hang', 'vi': 'vie_Latn', 'fa': 'fas_Arab', 'sr': 'srp_Cyrl',
            'ca': 'cat_Latn', 'mt': 'mlt_Latn', 'cy': 'cym_Latn', 'is': 'isl_Latn',
        }

        # Translation artifact patterns for cleaning
        self.translation_artifact_patterns = [
            # Original patterns from the class
            r'(\d+\.\d+\.\d+\.)+\d+',  # Repeated decimal patterns
            r'([!?.,:;-])\1{3,}',      # Repeated punctuation
            r'(\w+\s+)\1{3,}',         # Repeated words
            
            # Enhanced patterns for medical documents
            r'((?:clinical trial|study|patient|treatment)\s+){3,}',
            r'(p[<=]\d+\.\d+\s*){3,}',  # Repeated p-values
            r'(CI:\s*\d+\.\d+-\d+\.\d+\s*){3,}',  # Repeated confidence intervals
            r'(\d+\s*mg(?:/m2)?\s+){3,}',  # Repeated dosage information
        ]

        # Medical terms that should be preserved even if they appear repetitive
        self.medical_exclusions = [
            # Valid medical repetitions
            r'dose-dose\s+(?:escalation|reduction)',
            r'first-line.*second-line.*third-line',
            r'pre-treatment.*post-treatment',
        ]

        # Robust CUDA setup for Google Colab
        self.use_cuda = False
        self.device = "cpu"

        if torch.cuda.is_available():
            try:
                # Clear any existing CUDA context first
                if torch.cuda.is_initialized():
                    torch.cuda.empty_cache()
                    gc.collect()

                # Test CUDA more conservatively
                test_tensor = torch.tensor([1.0], dtype=torch.float32)
                test_tensor = test_tensor.to('cuda:0')
                result = test_tensor + 1
                test_tensor = test_tensor.cpu()
                del test_tensor, result
                torch.cuda.empty_cache()

                self.use_cuda = True
                self.device = "cuda:0"
                print("✓ Using CUDA")
            except Exception as e:
                print(f"⚠️  CUDA initialization failed, using CPU: {str(e)[:100]}")
                # Force cleanup and use CPU
                try:
                    torch.cuda.empty_cache()
                except:
                    pass
                gc.collect()
                self.device = "cpu"
        else:
            print("⚠️  CUDA not available, using CPU")

        # Current loaded translator
        self.current_translator = None
        self.current_language = None

    def preserve_medical_terms(self, text: str) -> tuple[str, dict]:
        """Replace medical terms with placeholders before translation."""
        preserved = {}
        modified_text = text
        
        for i, term in enumerate(self.preserve_terms):
            if term.lower() in text.lower():
                placeholder = f"__MEDICAL_TERM_{i}__"
                # Case-insensitive replacement
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                modified_text = pattern.sub(placeholder, modified_text)
                preserved[placeholder] = term
                
        return modified_text, preserved

    def restore_medical_terms(self, text: str, preserved: dict) -> str:
        """Restore medical terms after translation."""
        for placeholder, term in preserved.items():
            text = text.replace(placeholder, term)
        return text

    def is_table_content(self, text: str) -> bool:
        """Detect if content is likely a table."""
        table_indicators = [
            r'Row \d+:',
            r'\|.*\|.*\|',  # Pipe-separated content
            r'^\s*\d+\.\d+\s+\d+\.\d+',  # Numerical data patterns
            text.count('|') > 5,
            text.count('Row') > 3
        ]
        return any(re.search(pattern, text) if isinstance(pattern, str) else pattern 
                   for pattern in table_indicators)

    def translate_table_content(self, text: str, translator) -> str:
        """Special handling for table content."""
        # Preserve medical terms first
        text_with_placeholders, preserved_terms = self.preserve_medical_terms(text)
        
        # Split by rows
        rows = re.split(r'(Row \d+:)', text_with_placeholders)
        translated_rows = []
        
        for i, row in enumerate(rows):
            if re.match(r'Row \d+:', row):
                # Keep row markers as-is
                translated_rows.append(row)
            elif row.strip():
                # For table cells, preserve structure
                cells = row.split('|')
                translated_cells = []
                
                for cell in cells:
                    if cell.strip() and not re.match(r'^\d+\.?\d*$', cell.strip()):
                        # Only translate non-numeric cells
                        translated_cell = self.translate_single_chunk(cell.strip(), translator)
                        translated_cells.append(translated_cell)
                    else:
                        translated_cells.append(cell)
                        
                translated_rows.append('|'.join(translated_cells))
        
        # Restore medical terms
        translated_text = ''.join(translated_rows)
        return self.restore_medical_terms(translated_text, preserved_terms)

    def clean_translation_artifacts(self, text: str) -> str:
        """Clean common translation artifacts while preserving medical terminology."""
        # First check if this matches any medical exclusions
        for exclusion_pattern in self.medical_exclusions:
            if re.search(exclusion_pattern, text, re.IGNORECASE):
                return text  # Don't clean if it's a valid medical pattern
        
        # Apply artifact cleaning patterns
        cleaned_text = text
        for pattern in self.translation_artifact_patterns:
            # Replace repetitive patterns with single occurrence
            cleaned_text = re.sub(pattern, r'\1', cleaned_text)
        
        return cleaned_text

    def detect_document_language(self, text: str) -> Optional[str]:
        """
        Detect the primary language of a document.
        Returns 2-letter language code or None if detection fails.
        """
        if not text or len(text.strip()) < 20:
            return None

        try:
            # Clean text for better detection
            clean_text = ' '.join(text.split()[:200])  # Use first 200 words
            detected_lang = detect(clean_text)
            print(f"    Detected language: {detected_lang}")
            return detected_lang
        except Exception:
            print("    Language detection failed")
            return None

    def is_english_chunk(self, text: str) -> bool:
        """
        Quick check if a chunk is in English.
        Uses simple heuristics for speed.
        """
        if not text or len(text.strip()) < 10:
            return True  # Treat short text as English (safer)

        text_lower = text.lower()

        # Common English function words
        english_words = [
            ' the ', ' and ', ' of ', ' to ', ' a ', ' in ', ' is ', ' it ', ' you ', ' that ',
            ' he ', ' was ', ' for ', ' on ', ' are ', ' as ', ' with ', ' his ', ' they ',
            ' i ', ' at ', ' be ', ' this ', ' have ', ' from ', ' or ', ' one ', ' had ',
            ' by ', ' word ', ' but ', ' not ', ' what ', ' all ', ' were ', ' we '
        ]

        # Count English words
        english_count = sum(1 for word in english_words if word in f' {text_lower} ')
        total_words = len(text.split())

        if total_words > 5:
            english_ratio = english_count / total_words
            return english_ratio > 0.1  # 10% threshold

        return False

    def load_translator_for_language(self, language: str):
        """
        Load the appropriate translator for the given language.
        Try Helsinki model first, fall back to Facebook NLLB.
        Robust CUDA handling with CPU fallback.
        """
        if self.current_language == language and self.current_translator:
            return self.current_translator

        # Clear previous translator
        self.clear_translator()

        print(f"    Loading translator for language: {language}")

        # Try Helsinki model first (higher quality)
        if language in self.helsinki_models:
            try:
                model_name = self.helsinki_models[language]
                print(f"    Trying Helsinki model: {model_name}")

                # Try with current device first
                try:
                    if self.device.startswith("cuda"):
                        # Force clean CUDA state for Colab
                        torch.cuda.empty_cache()
                        gc.collect()

                    translator = pipeline(
                        "translation",
                        model=model_name,
                        device=self.device,
                        torch_dtype=torch.float32,  # Use float32 for stability
                        trust_remote_code=False,
                    )

                    # Test the translator with a simple phrase
                    test_result = translator("Hello world", max_length=50)

                    self.current_translator = translator
                    self.current_language = language
                    print(f"    ✓ Helsinki model loaded successfully on {self.device}")
                    return translator

                except Exception as e:
                    if self.device.startswith("cuda"):
                        print(f"    ⚠️  CUDA failed, trying CPU: {str(e)[:50]}")
                        # Fallback to CPU
                        translator = pipeline(
                            "translation",
                            model=model_name,
                            device="cpu",
                            torch_dtype=torch.float32,
                            trust_remote_code=False,
                        )

                        # Test the translator
                        test_result = translator("Hello world", max_length=50)

                        self.current_translator = translator
                        self.current_language = language
                        self.device = "cpu"  # Switch to CPU for remaining files
                        print(f"    ✓ Helsinki model loaded successfully on CPU")
                        return translator
                    else:
                        raise e

            except Exception as e:
                print(f"    ✗ Helsinki model failed: {str(e)[:50]}")

        # Fall back to Facebook NLLB model
        if language in self.nllb_lang_mapping:
            try:
                print(f"    Trying Facebook NLLB model...")

                model_name = "facebook/nllb-200-distilled-600M"

                # Load with current device
                try:
                    if self.device.startswith("cuda"):
                        # Clean CUDA state before loading large model
                        torch.cuda.empty_cache()
                        gc.collect()

                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float32,  # Use float32 for stability
                        trust_remote_code=False,
                    )

                    if self.device.startswith("cuda"):
                        model = model.to(self.device)

                    # Create translation function with generation parameters
                    def nllb_translate(text, generation_params=None, **kwargs):
                        try:
                            src_lang = self.nllb_lang_mapping[language]
                            tgt_lang = 'eng_Latn'

                            # Tokenize
                            max_input_length = min(512, generation_params.get('max_length', 400) + 50) if generation_params else 256
                            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_input_length)
                            if self.device.startswith("cuda"):
                                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                            # Set up generation parameters
                            gen_kwargs = {
                                'forced_bos_token_id': tokenizer.convert_tokens_to_ids(tgt_lang),
                                'max_length': generation_params.get('max_length', 256) if generation_params else 256,
                                'num_beams': generation_params.get('num_beams', 2) if generation_params else 2,
                                'length_penalty': generation_params.get('length_penalty', 0.9) if generation_params else 0.9,
                                'do_sample': generation_params.get('do_sample', False) if generation_params else False,
                                'no_repeat_ngram_size': generation_params.get('no_repeat_ngram_size', 2) if generation_params else 2,
                                'repetition_penalty': generation_params.get('repetition_penalty', 1.1) if generation_params else 1.1,
                            }
                            
                            # Add early stopping if specified
                            if generation_params and generation_params.get('early_stopping'):
                                gen_kwargs['early_stopping'] = True

                            # Generate translation
                            with torch.no_grad():
                                translated_tokens = model.generate(**inputs, **gen_kwargs)

                            # Decode
                            translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                            return [{'translation_text': translation}]

                        except Exception as e:
                            print(f"      NLLB translation error: {str(e)[:50]}")
                            return [{'translation_text': text}]  # Return original on error

                    # Test the translator
                    test_result = nllb_translate("Hello world")

                    self.current_translator = nllb_translate
                    self.current_language = language
                    print(f"    ✓ Facebook NLLB model loaded successfully on {self.device}")
                    return nllb_translate

                except Exception as e:
                    if self.device.startswith("cuda"):
                        print(f"    ⚠️  CUDA failed, trying CPU: {str(e)[:50]}")
                        # Fallback to CPU
                        tokenizer = AutoTokenizer.from_pretrained(model_name)
                        model = AutoModelForSeq2SeqLM.from_pretrained(
                            model_name,
                            torch_dtype=torch.float32,
                            trust_remote_code=False,
                        )
                        # Keep model on CPU

                        def nllb_translate_cpu(text, generation_params=None, **kwargs):
                            try:
                                src_lang = self.nllb_lang_mapping[language]
                                tgt_lang = 'eng_Latn'

                                # Tokenize with dynamic max length
                                max_input_length = min(512, generation_params.get('max_length', 400) + 50) if generation_params else 256
                                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_input_length)

                                # Set up generation parameters
                                gen_kwargs = {
                                    'forced_bos_token_id': tokenizer.convert_tokens_to_ids(tgt_lang),
                                    'max_length': generation_params.get('max_length', 256) if generation_params else 256,
                                    'num_beams': generation_params.get('num_beams', 2) if generation_params else 2,
                                    'length_penalty': generation_params.get('length_penalty', 0.9) if generation_params else 0.9,
                                    'do_sample': generation_params.get('do_sample', False) if generation_params else False,
                                    'no_repeat_ngram_size': generation_params.get('no_repeat_ngram_size', 2) if generation_params else 2,
                                    'repetition_penalty': generation_params.get('repetition_penalty', 1.1) if generation_params else 1.1,
                                }
                                
                                # Add early stopping if specified
                                if generation_params and generation_params.get('early_stopping'):
                                    gen_kwargs['early_stopping'] = True

                                with torch.no_grad():
                                    translated_tokens = model.generate(**inputs, **gen_kwargs)

                                translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                                return [{'translation_text': translation}]

                            except Exception as e:
                                print(f"      NLLB CPU translation error: {str(e)[:50]}")
                                return [{'translation_text': text}]

                        # Test the translator
                        test_result = nllb_translate_cpu("Hello world")

                        self.current_translator = nllb_translate_cpu
                        self.current_language = language
                        self.device = "cpu"  # Switch to CPU for remaining files
                        print(f"    ✓ Facebook NLLB model loaded successfully on CPU")
                        return nllb_translate_cpu
                    else:
                        raise e

            except Exception as e:
                print(f"    ✗ Facebook NLLB model failed: {str(e)[:50]}")

        print(f"    ✗ No translator available for language: {language}")
        return None

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for sequential translation."""
        # Enhanced sentence splitting that handles medical abbreviations
        # Common medical abbreviations that shouldn't trigger sentence breaks
        medical_abbrevs = r'(?:Dr|Mr|Mrs|Ms|Prof|vs|etc|i\.e|e\.g|cf|approx|max|min|Fig|Tab|Ref|Vol|No|pg|pp|PFS|OS|ORR|DCR|CI|HR|OR|RR|AE|SAE|ECOG|BCLC|HCC|NSCLC|KRAS|G12C)'
        
        # Replace medical abbreviations temporarily
        protected_text = re.sub(f'({medical_abbrevs})\\.', r'\1__PERIOD__', text, flags=re.IGNORECASE)
        
        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+\s+', protected_text)
        
        # Restore periods in abbreviations
        sentences = [s.replace('__PERIOD__', '.') for s in sentences if s.strip()]
        
        return sentences

    def detect_repetition(self, text: str, max_repeat_ratio: float = 0.3) -> bool:
        """Detect if text has excessive repetition."""
        if not text or len(text) < 20:
            return False
        
        words = text.lower().split()
        if len(words) < 10:
            return False
        
        # Check for repeated n-grams
        for n in [2, 3, 4]:  # Check 2-grams, 3-grams, 4-grams
            ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
            if not ngrams:
                continue
                
            # Count occurrences
            ngram_counts = {}
            for ngram in ngrams:
                ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
            
            # Check if any n-gram appears too frequently
            max_count = max(ngram_counts.values())
            repeat_ratio = max_count / len(ngrams)
            
            if repeat_ratio > max_repeat_ratio:
                return True
        
        return False

    def get_generation_params(self, attempt: int = 1) -> dict:
        """Get generation parameters that become more conservative with retries."""
        base_params = {
            'max_length': 400 - (attempt * 50),  # Reduce max length on retries
            'truncation': True,
            'no_repeat_ngram_size': 2 + attempt,  # Increase anti-repetition
            'repetition_penalty': 1.1 + (attempt * 0.1),  # Increase penalty
            'do_sample': False,  # Use deterministic generation initially
        }
        
        # For retries, add more conservative parameters
        if attempt > 1:
            base_params.update({
                'num_beams': min(4, 2 + attempt),  # Use beam search for retries
                'length_penalty': 1.0,
                'early_stopping': True
            })
        
        return base_params

    def translate_with_retry(self, text: str, translator, max_attempts: int = 3) -> str:
        """Translate text with retry logic for handling repetition."""
        for attempt in range(1, max_attempts + 1):
            try:
                # Get generation parameters for this attempt
                gen_params = self.get_generation_params(attempt)
                
                # For Helsinki models (pipeline), filter supported parameters
                if hasattr(translator, 'model'):
                    # This is a HuggingFace pipeline
                    pipeline_params = {k: v for k, v in gen_params.items() 
                                     if k in ['max_length', 'truncation', 'no_repeat_ngram_size', 
                                             'repetition_penalty', 'do_sample', 'num_beams', 
                                             'length_penalty', 'early_stopping']}
                    result = translator(text, **pipeline_params)
                else:
                    # This is our custom NLLB function - modify to accept params
                    result = translator(text, generation_params=gen_params)
                
                translated_text = result[0]['translation_text']
                
                # Check for repetition
                if not self.detect_repetition(translated_text):
                    return translated_text
                else:
                    print(f"      Repetition detected in attempt {attempt}, retrying...")
                    if attempt == max_attempts:
                        print(f"      Max attempts reached, using best available translation")
                        return translated_text
                    
            except Exception as e:
                print(f"      Translation attempt {attempt} failed: {str(e)[:50]}")
                if attempt == max_attempts:
                    return text  # Return original if all attempts fail
        
        return text

    def translate_single_chunk(self, text: str, translator) -> str:
        """Translate a single chunk of text with improved segmentation and anti-repetition."""
        if not text.strip() or not translator:
            return text

        try:
            # Preserve medical terms
            protected_text, preserved_terms = self.preserve_medical_terms(text)
            
            # Instead of truncating, split into smaller segments
            words = protected_text.split()
            
            if len(words) <= 50:
                # Short text - translate directly
                translated_text = self.translate_with_retry(protected_text, translator)
            elif len(words) <= 150:
                # Medium text - try sentence-by-sentence translation
                sentences = self.split_into_sentences(protected_text)
                
                if len(sentences) > 1:
                    # Translate sentence by sentence
                    translated_sentences = []
                    for sentence in sentences:
                        if sentence.strip():
                            trans_sentence = self.translate_with_retry(sentence.strip(), translator)
                            translated_sentences.append(trans_sentence)
                    translated_text = ' '.join(translated_sentences)
                else:
                    # Single long sentence - translate with retry
                    translated_text = self.translate_with_retry(protected_text, translator)
            else:
                # Very long text - split into chunks and translate sequentially
                chunk_size = 100  # words per chunk
                translated_chunks = []
                
                for i in range(0, len(words), chunk_size):
                    chunk_words = words[i:i + chunk_size]
                    chunk_text = ' '.join(chunk_words)
                    
                    if chunk_text.strip():
                        trans_chunk = self.translate_with_retry(chunk_text, translator)
                        translated_chunks.append(trans_chunk)
                
                translated_text = ' '.join(translated_chunks)
            
            # Restore medical terms
            final_text = self.restore_medical_terms(translated_text, preserved_terms)
            
            # Clean translation artifacts
            cleaned_text = self.clean_translation_artifacts(final_text)
            
            return cleaned_text

        except Exception as e:
            print(f"      Translation error: {str(e)[:50]}")
            return text  # Return original on error

    def translate_text(self, text: str) -> str:
        """
        Main translation method that handles both regular and table content.
        """
        if not text.strip() or not self.current_translator:
            return text

        # Check if this is table content
        if self.is_table_content(text):
            return self.translate_table_content(text, self.current_translator)
        else:
            return self.translate_single_chunk(text, self.current_translator)

    def clear_translator(self):
        """Clear current translator and free memory."""
        self.current_translator = None
        self.current_language = None

        # Safe memory cleanup
        gc.collect()
        if self.use_cuda and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except:
                pass  # Ignore CUDA cleanup errors

    def process_json_file(self, input_path: str, output_path: str):
        """
        Process a single JSON file:
        1. Detect document language
        2. Load appropriate translator
        3. Process each chunk (check if English, translate if not)
        """
        file_name = os.path.basename(input_path)
        print(f"\n📄 Processing: {file_name}")

        # Load JSON
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ✗ Error loading JSON: {e}")
            return

        # Extract all text for document-level language detection
        all_text_parts = []
        if 'doc_id' in data:
            all_text_parts.append(str(data['doc_id']))

        if 'chunks' in data:
            for chunk in data['chunks']:
                if 'heading' in chunk and chunk['heading']:
                    all_text_parts.append(chunk['heading'])
                if 'text' in chunk and chunk['text']:
                    all_text_parts.append(chunk['text'])

        # Detect document language
        combined_text = ' '.join(all_text_parts)
        document_language = self.detect_document_language(combined_text)

        if not document_language or document_language == 'en':
            print(f"  📋 Document is English, copying without translation")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copy(input_path, output_path)
            return

        # Load translator for detected language
        translator = self.load_translator_for_language(document_language)
        if not translator:
            print(f"  📋 No translator available, copying original file")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copy(input_path, output_path)
            return

        # Process chunks
        if 'chunks' not in data:
            print(f"  📋 No chunks found, copying original file")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copy(input_path, output_path)
            return

        total_chunks = len(data['chunks'])
        translated_count = 0
        english_count = 0

        print(f"  📊 Processing {total_chunks} chunks...")

        for i, chunk in enumerate(data['chunks']):
            if i % 10 == 0 or i == total_chunks - 1:
                print(f"    Chunk {i+1}/{total_chunks}")

            # Process heading
            if 'heading' in chunk and chunk['heading']:
                if self.is_english_chunk(chunk['heading']):
                    english_count += 1
                else:
                    chunk['heading'] = self.translate_text(chunk['heading'])
                    translated_count += 1

            # Process text
            if 'text' in chunk and chunk['text']:
                if self.is_english_chunk(chunk['text']):
                    english_count += 1
                else:
                    chunk['text'] = self.translate_text(chunk['text'])
                    translated_count += 1

        # Save translated file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"  ✓ Completed: {english_count} English chunks, {translated_count} translated chunks")

        # Update global stats
        self.english_chunks_preserved += english_count
        self.chunks_translated += translated_count

        # Clear translator after each file to free memory
        self.clear_translator()

    def translate_documents(self):
        """
        Main method to translate all documents in input directory.
        """
        print("🚀 Starting document translation...")

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Find all JSON files
        json_files = []
        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith('.json'):
                    input_path = os.path.join(root, file)
                    rel_path = os.path.relpath(input_path, self.input_dir)
                    output_path = os.path.join(self.output_dir, rel_path)
                    json_files.append((input_path, output_path))

        total_files = len(json_files)
        print(f"📁 Found {total_files} JSON files to process")

        if total_files == 0:
            print("⚠️  No JSON files found in input directory")
            return

        # Process each file
        for i, (input_path, output_path) in enumerate(json_files, 1):
            print(f"\n[{i}/{total_files}]", end=" ")
            try:
                self.process_json_file(input_path, output_path)
            except Exception as e:
                print(f"  ✗ Error processing file: {str(e)[:100]}")
                # Copy original file on error
                try:
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    shutil.copy(input_path, output_path)
                    print(f"  📋 Copied original file instead")
                except Exception as copy_error:
                    print(f"  ✗ Failed to copy original: {copy_error}")

        # Final summary
        print(f"\n🎉 Translation Complete!")
        print(f"📊 Summary:")
        print(f"   • Total files processed: {total_files}")
        print(f"   • English chunks preserved: {self.english_chunks_preserved}")
        print(f"   • Chunks translated: {self.chunks_translated}")



class PostCleaner:
    """
    Advanced class to clean up translation artifacts in processed JSON documents.
    
    This cleaner handles complex cases including:
    1. Numerical pattern repetition (0.09.09.09...)
    2. Dollar sign and other symbol repetition ($$$$$)
    3. Quoted row markers and duplicate rows
    4. Nonsensical word repetitions (agglomeration, agitation...)
    5. Technical phrase repetition (material injury, material...)
    6. Mixed numerical and textual artifacts
    """
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        maintain_folder_structure: bool = True
    ):
        """
        Initialize the translation cleaner.
        
        Args:
            input_dir: Directory containing translated JSON files
            output_dir: Directory to save cleaned files
            maintain_folder_structure: Whether to maintain folder structure when saving
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.maintain_folder_structure = maintain_folder_structure
        
        # Counter for statistics
        self.stats = {
            "files_processed": 0,
            "text_chunks_cleaned": 0,
            "table_chunks_cleaned": 0,
            "artifacts_removed": 0,
            "chinese_chars_removed": 0,
            "excessive_punctuation_fixed": 0,
            "table_rows_fixed": 0,
            "repeated_phrases_removed": 0,
            "repeated_words_fixed": 0,
            "numerical_patterns_fixed": 0,
            "quoted_rows_fixed": 0,
            "symbol_repetition_fixed": 0,
            "special_patterns_fixed": 0
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def clean_all_documents(self):
        """Process all JSON files in the input directory recursively."""
        # Find all JSON files
        json_files = glob.glob(os.path.join(self.input_dir, "**/*.json"), recursive=True)
        print(f"Found {len(json_files)} JSON files to clean.")
        
        for file_path in json_files:
            self.clean_document(file_path)
        
        # Print statistics
        print(f"\nCleaning Complete:")
        print(f"  Files processed: {self.stats['files_processed']}")
        print(f"  Text chunks cleaned: {self.stats['text_chunks_cleaned']}")
        print(f"  Table chunks cleaned: {self.stats['table_chunks_cleaned']}")
        print(f"  Artifacts removed: {self.stats['artifacts_removed']}")
        print(f"  Chinese characters removed: {self.stats['chinese_chars_removed']}")
        print(f"  Excessive punctuation fixed: {self.stats['excessive_punctuation_fixed']}")
        print(f"  Table rows fixed: {self.stats['table_rows_fixed']}")
        print(f"  Repeated phrases removed: {self.stats['repeated_phrases_removed']}")
        print(f"  Repeated words fixed: {self.stats['repeated_words_fixed']}")
        print(f"  Numerical patterns fixed: {self.stats['numerical_patterns_fixed']}")
        print(f"  Quoted rows fixed: {self.stats['quoted_rows_fixed']}")
        print(f"  Symbol repetition fixed: {self.stats['symbol_repetition_fixed']}")
        print(f"  Special patterns fixed: {self.stats['special_patterns_fixed']}")
    
    def clean_document(self, file_path: str):
        """Clean a single JSON document."""
        rel_path = os.path.relpath(file_path, self.input_dir)
        print(f"Cleaning: {rel_path}")
        
        try:
            # Load the document
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Clean the document
            cleaned_data = self._process_document(data)
            
            # Determine output path
            if self.maintain_folder_structure:
                # Create subdirectories if needed
                output_path = os.path.join(self.output_dir, rel_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            else:
                # Flat structure - just use filename
                output_path = os.path.join(self.output_dir, os.path.basename(file_path))
            
            # Save the cleaned document
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
            
            self.stats["files_processed"] += 1
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    def _process_document(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a document by cleaning all its chunks."""
        if not isinstance(data, dict) or "chunks" not in data:
            return data
        
        # Clean each chunk
        for chunk in data["chunks"]:
            # Clean the heading
            if "heading" in chunk:
                chunk["heading"] = self._clean_text(chunk["heading"], is_heading=True)
            
            # Clean the text content
            if "text" in chunk:
                original_text = chunk["text"]
                is_table = self._is_table_chunk(chunk)
                
                if is_table:
                    chunk["text"] = self._clean_table_text(original_text)
                    self.stats["table_chunks_cleaned"] += 1
                else:
                    chunk["text"] = self._clean_text(original_text)
                    self.stats["text_chunks_cleaned"] += 1
                
                # Apply advanced pattern cleaning regardless of chunk type
                chunk["text"] = self._clean_advanced_patterns(chunk["text"])
        
        return data
    
    def _is_table_chunk(self, chunk: Dict[str, Any]) -> bool:
        """Determine if a chunk contains a table."""
        # Check if the heading mentions "table"
        if "heading" in chunk and re.search(r'table', chunk["heading"], re.IGNORECASE):
            return True
        
        # Check if the text contains table-like rows
        if "text" in chunk and re.search(r'Row \d+:', chunk["text"]):
            return True
        
        return False
    
    def _clean_quoted_rows(self, text: str) -> str:
        """
        Clean quoted row markers like "Row 5" "Row 6" "Row 6".
        """
        if not text:
            return text
        
        # Count occurrences before cleaning
        quoted_row_pattern = r'"Row \d+"'
        count_before = len(re.findall(quoted_row_pattern, text))
        
        # Fix consecutive quoted rows (e.g., "Row 6" "Row 6" "Row 7" "Row 7")
        text = re.sub(r'("Row \d+"\s*)(\1)+', r'\1', text)
        
        # Fix rows with too many quotes
        text = re.sub(r'"Row (\d+)"\s+"Row \1"', r'Row \1:', text)
        
        # Count occurrences after cleaning
        count_after = len(re.findall(quoted_row_pattern, text))
        self.stats["quoted_rows_fixed"] += (count_before - count_after)
        
        return text
    
    def _clean_numerical_patterns(self, text: str) -> str:
        """
        Clean numerical pattern repetition like 0.09.09.09.09.09...
        """
        # Find patterns of repeating digits with dots or other separators
        patterns_found = 0
        
        # Find decimal number patterns that repeat (like 0.09.09.09...)
        decimal_repetition = r'(\d+\.\d{1,3})(\.\d{1,3}){3,}'
        matches = re.findall(decimal_repetition, text)
        for match in matches:
            if match and match[0]:
                # Get the first part of the pattern
                base_pattern = match[0]
                # Find the full repeating pattern in the text
                full_pattern = re.escape(base_pattern) + r'(\.\d{1,3}){3,}'
                replacement = base_pattern  # Replace with just the first occurrence
                text = re.sub(full_pattern, replacement, text)
                patterns_found += 1
        
        # Another pattern: repeating decimals like 0.090.090.09...
        decimal_repetition2 = r'(\d+\.\d{2,3})(\d+\.\d{2,3})(\d+\.\d{2,3})+'
        text = re.sub(decimal_repetition2, r'\1', text)
        
        # Yet another pattern: isolated repeating numbers
        repeated_numbers = re.compile(r'(\d{1,3})(\1){3,}')
        text = re.sub(repeated_numbers, r'\1\1', text)
        
        self.stats["numerical_patterns_fixed"] += patterns_found
        return text
    
    def _clean_symbol_repetition(self, text: str) -> str:
        """
        Clean repetitive symbols like $$$$$$$$ or ######## that go beyond normal formatting.
        """
        symbol_patterns = {
            # Repeated $ signs
            r'\${5,}': '$$$',
            # Repeated # signs
            r'#{5,}': '###',
            # Repeated @ signs
            r'@{5,}': '@@@',
            # Repeated + signs
            r'\+{5,}': '+++',
            # Repeated * signs
            r'\*{5,}': '***',
            # Repeated = signs
            r'={5,}': '===',
        }
        
        count = 0
        for pattern, replacement in symbol_patterns.items():
            # Count matches before replacement
            matches = re.findall(pattern, text)
            count += len(matches)
            
            # Replace the repetitions
            text = re.sub(pattern, replacement, text)
        
        self.stats["symbol_repetition_fixed"] += count
        return text
    
    def _clean_special_phrase_repetition(self, text: str) -> str:
        """
        Clean specific phrase repetitions found in examples.
        """
        special_patterns = [
            # Abortion of information/commission repeating
            (r'(Abortion of (?:this information|the Commission)(?:\s+|,)){3,}', r'\1\1'),
            
            # Agglomeration/agitation repeating
            (r'(agglomeration|agitation)(?:\s+\1){3,}', r'\1 \1'),
            
            # Repeated "ag ag ag" sequences
            (r'(ag\s+){3,}', r'ag ag '),
            
            # material injury repetition
            (r'((?:material |)injury,?\s+){5,}', r'material injury, '),
            
            # "material, material, material" repetition
            (r'(material,?\s+){3,}', r'material, material '),
            
            # "etc, etc, etc" repetition
            (r'(etc(?:,|\.)?\s*){3,}', r'etc., etc.'),
            
            # "of the of the of the" repetition
            (r'(of the\s+){3,}', r'of the '),
            
            # "Row: Row: Row:" repetition
            (r'(Row:\s*){3,}', r'Row: '),
            
            # progressively/progressionlessly/progressiveness repeating
            (r'(progress(?:ion|ively|iveness)(?:\s+|,)){3,}', r'\1\1'),
        ]
        
        count = 0
        for pattern, replacement in special_patterns:
            # Count matches before replacement
            matches = len(re.findall(pattern, text))
            count += matches
            
            # Replace the repetitions
            text = re.sub(pattern, replacement, text)
        
        self.stats["special_patterns_fixed"] += count
        return text
    
    def _clean_repeating_row_markers(self, text: str) -> str:
        """
        Clean repetitive row markers, especially in tables.
        """
        # Fix sequences of repeating "Row X:" or "Row: Row: Row:"
        row_fixes = 0
        
        # Fix "Row X: Row X: Row X:" patterns
        row_pattern = r'(Row \d+:)\s*\1+'
        row_fixes += len(re.findall(row_pattern, text))
        text = re.sub(row_pattern, r'\1', text)
        
        # Fix "Row: Row: Row:" patterns
        row_colon_pattern = r'(Row:)\s*\1+'
        row_fixes += len(re.findall(row_colon_pattern, text))
        text = re.sub(row_colon_pattern, r'\1', text)
        
        # Fix "Row: Row: Row: Row: Row:" patterns without numbers
        row_pattern2 = r'(Row:\s+){3,}'
        row_fixes += len(re.findall(row_pattern2, text))
        text = re.sub(row_pattern2, r'Row: ', text)
        
        # Fix sequences with numbers like "Row: 27: Row:"
        row_pattern3 = r'Row:\s*\d+:\s*Row:'
        row_fixes += len(re.findall(row_pattern3, text))
        text = re.sub(row_pattern3, r'Row:', text)
        
        self.stats["table_rows_fixed"] += row_fixes
        return text
    
    def _clean_advanced_patterns(self, text: str) -> str:
        """
        Apply advanced pattern cleaning that works on all document types.
        """
        # Save original length for artifact counting
        original_length = len(text)
        
        # Apply all advanced cleaning methods
        text = self._clean_numerical_patterns(text)
        text = self._clean_symbol_repetition(text)
        text = self._clean_quoted_rows(text)
        text = self._clean_special_phrase_repetition(text)
        text = self._clean_repeating_row_markers(text)
        
        # Specialized pattern for the examples you provided:
        # Pattern with "0.09.09.09..." repeating (from the Discussion section)
        text = re.sub(r'0\.09\.09\.09\.09\.09\.09\.09\.09\.09\.09\.09\.09\.09\.09\.09\.09\.09\.09[\.09]*', 
                      r'0.09', text)
        
        # Count artifacts removed
        artifacts_removed = original_length - len(text)
        if artifacts_removed > 0:
            self.stats["artifacts_removed"] += artifacts_removed
        
        return text
    
    def _find_repeated_phrases(self, text: str, min_length: int = 3, max_length: int = 30) -> List[tuple]:
        """Find repeated phrases in text."""
        words = text.split()
        if len(words) < min_length * 2:  # Need at least 2 occurrences to find repetition
            return []
        
        # Try different phrase lengths
        repeated_phrases = []
        
        for phrase_len in range(min_length, min(max_length, len(words) // 2 + 1)):
            # Check each possible phrase of this length
            for i in range(len(words) - phrase_len + 1):
                phrase = ' '.join(words[i:i+phrase_len])
                
                # Count occurrences
                count = 0
                for j in range(i + phrase_len, len(words) - phrase_len + 1, phrase_len):
                    if ' '.join(words[j:j+phrase_len]) == phrase:
                        count += 1
                    else:
                        break
                
                # If phrase repeats, add it to our list
                if count > 0:
                    repeated_phrases.append((phrase, count + 1))
                    # Skip ahead to avoid finding sub-phrases of this repetition
                    i += (count + 1) * phrase_len - 1
        
        return repeated_phrases
    
    def _remove_repeated_phrases(self, text: str) -> str:
        """Remove repeated phrases, keeping just one instance."""
        if not text:
            return text
        
        # Find repeated phrases
        repeated_phrases = self._find_repeated_phrases(text)
        count = 0
        
        for phrase, occurrences in repeated_phrases:
            # Create pattern that matches exactly this phrase repeated multiple times
            pattern = re.escape(phrase) + r'(?:\s+' + re.escape(phrase) + r')+'
            
            # Replace with single instance
            new_text = re.sub(pattern, phrase, text)
            
            # Update count if replacement occurred
            if new_text != text:
                count += 1
                text = new_text
        
        self.stats["repeated_phrases_removed"] += count
        return text
    
    def _remove_repeated_single_words(self, text: str) -> str:
        """Remove long runs of the same word (e.g., 'no, no, no, no...')."""
        if not text:
            return text
        
        # Pattern for repeated words with optional punctuation
        pattern = r'\b(\w+(?:[,.;:]? |, |\. ))\1{2,}'
        
        # Find all matches
        matches = re.findall(pattern, text)
        
        # Replace each match with just two instances (e.g., "no, no")
        for match in matches:
            repeat_pattern = re.escape(match) + r'{3,}'  # 3+ occurrences
            text = re.sub(repeat_pattern, match + match, text)
            self.stats["repeated_words_fixed"] += 1
        
        return text
    
    def _clean_text(self, text: str, is_heading: bool = False) -> str:
        """
        Clean general text content.
        This enhanced version handles complex patterns.
        """
        if not text:
            return text
        
        original_length = len(text)
        
        # Step 1: Handle basic patterns
        
        # Remove Chinese/Japanese/Korean characters
        chinese_char_count = len(re.findall(r'[一-龥的]', text))
        text = re.sub(r'[一-龥的]', '', text)
        self.stats["chinese_chars_removed"] += chinese_char_count
        
        # Handle repeated ellipses and dots
        text = re.sub(r'\.{5,}', '...', text)  # Replace long runs of dots with ellipsis
        
        # Fix excessive repetitions of common words
        text = re.sub(r'(?:no,? ){3,}', 'no, no ', text)
        text = re.sub(r'(?:yes,? ){3,}', 'yes, yes ', text)
        text = re.sub(r'(?:not ){3,}', 'not not ', text)
        
        # Remove excessive punctuation
        punct_matches = len(re.findall(r'([!?.:;,\-_=\*\+#&\|\[\]\{\}\(\)<>])\1{3,}', text))
        text = re.sub(r'([!?.:;,\-_=\*\+#&\|\[\]\{\}\(\)<>])\1{3,}', r'\1', text)
        self.stats["excessive_punctuation_fixed"] += punct_matches
        
        # Remove excessive letter repetitions
        text = re.sub(r'([a-zA-Z])\1{3,}', r'\1', text)
        
        # Step 2: Handle repeated phrases
        if len(text.split()) > 5:  # Only for longer texts
            text = self._remove_repeated_phrases(text)
            text = self._remove_repeated_single_words(text)
        
        # Step 3: Final formatting adjustments
        
        # Fix spacing issues
        text = re.sub(r'\s+', ' ', text)
        
        # For headings only, apply specific rules
        if is_heading:
            # Remove table markers in headings
            text = re.sub(r'table underheading', 'table heading', text)
            
            # Clean up angle brackets in headings
            text = re.sub(r'<([^<>]*)>', r'\1', text)
            
            # Remove row markers in headings
            text = re.sub(r'Row \d+:\s*', '', text)
        
        # Update artifact counter
        artifacts_removed = original_length - len(text)
        if artifacts_removed > 0:
            self.stats["artifacts_removed"] += artifacts_removed
        
        return text.strip()
    
    def _clean_table_text(self, text: str) -> str:
        """
        Clean text specifically for table content.
        Enhanced to handle complex patterns in tables.
        """
        if not text:
            return text
        
        original_length = len(text)
        
        # Step 1: First run special patterns for tables
        
        # Fix duplicate row labels (Row 1:Row 1:Row 1:)
        row_fixes = 0
        row_fixes += len(re.findall(r'(Row \d+:)\s*\1+', text))
        text = re.sub(r'(Row \d+:)\s*\1+', r'\1', text)
        
        # Fix "Low N" to "Row N"
        row_fixes += len(re.findall(r'Low (\d+)', text))
        text = re.sub(r'Low (\d+)', r'Row \1:', text)
        
        # Fix row label format
        row_fixes += len(re.findall(r'Row (\d+)!!+', text))
        text = re.sub(r'Row (\d+)!!+', r'Row \1:', text)
        
        row_fixes += len(re.findall(r'Row (\d+):的', text))
        text = re.sub(r'Row (\d+):的', r'Row \1:', text)
        
        # Fix consecutive empty row numbers
        text = re.sub(r'(Row \d+:\s*\n\s*){3,}', r'Row 1:\n', text)
        
        self.stats["table_rows_fixed"] += row_fixes
        
        # Apply advanced pattern cleaning
        text = self._clean_advanced_patterns(text)
        
        # Clean repeated content within rows
        parts = re.split(r'(Row \d+:)', text)
        result_parts = []
        
        for i, part in enumerate(parts):
            if i % 2 == 0:  # Even parts are content between "Row N:" markers
                if part.strip():
                    # Clean the content of this row
                    cleaned_part = self._clean_text(part)  # Use standard text cleaning
                    if len(part.split()) > 5:  # Only for longer content
                        # Try to fix repeated phrases in this row
                        cleaned_part = self._remove_repeated_phrases(cleaned_part)
                    result_parts.append(cleaned_part)
                else:
                    result_parts.append(part)
            else:  # Odd parts are "Row N:" markers
                result_parts.append(part)
        
        text = ''.join(result_parts)
        
        # Special handling for common table artifacts
        
        # Repeated "Indication of AMM & " pattern
        text = re.sub(r'(Indication of AMM &\s+)+', r'Indication of AMM & ', text)
        
        # Remove consecutive duplicate items in comma-separated lists
        text = re.sub(r'([^,]+, )(\1)+', r'\1', text)
        
        # Fix row formatting
        
        # Remove empty rows
        text = re.sub(r'Row \d+:\s*(\n|$)', '', text)
        
        # Fix spacing issues
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Make sure rows are on new lines
        text = re.sub(r'(Row \d+:)(?!\n)', r'\1\n', text)
        
        # Update artifact counter
        artifacts_removed = original_length - len(text)
        if artifacts_removed > 0:
            self.stats["artifacts_removed"] += artifacts_removed
        
        return text.strip()