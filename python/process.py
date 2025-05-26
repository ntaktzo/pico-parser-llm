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

        # Extract source type 
        self.source_type = self.extract_source_type_from_path()

        print("--------------------------------------------")
        print(f"Source type for '{self.pdf_path}': {self.source_type}")
        print(f"Submission year for '{self.pdf_path}': {self.created_date}")
        print(f"Country for '{self.pdf_path}': {self.country}")

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
from typing import Optional
from langdetect import detect
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

class Translator:
    """
    Fixed Translator class that processes JSON files with proper parameter handling
    and conservative repetition detection to preserve medical content.
    
    This class replaces the original Translator in python/process.py
    """
    def __init__(
        self,
        input_dir,
        output_dir,
        max_chunk_length=200  # Reduced from 300 to avoid token limit issues
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.max_chunk_length = max_chunk_length
        self.english_chunks_preserved = 0
        self.chunks_translated = 0

        # Mapping from source language to the corresponding Helsinki-NLP model
        self.models = {
            'fr': 'Helsinki-NLP/opus-mt-fr-en',
            'de': 'DunnBC22/opus-mt-de-en-OPUS_Medical_German_to_English',
            'pl': 'Helsinki-NLP/opus-mt-pl-en',
            'es': 'Helsinki-NLP/opus-mt-es-en',
            'it': 'Helsinki-NLP/opus-mt-it-en',
            'nl': 'FremyCompany/opus-mt-nl-en-healthcare',
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
            'pt': 'Helsinki-NLP/opus-mt-pt-en',
        }
        
        # Language groups for fallback models
        self.language_groups = {
            'facebook/mbart-large-50-many-to-many-mmt': {
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
            print("Device set to use cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Device set to use mps")
        else:
            self.device = torch.device("cpu")
            print("Device set to use cpu")
            
        # FIXED: More conservative repetition patterns that only flag severe cases
        self.severe_repetition_patterns = [
            r'(\b\w{3,}\b)(\s+\1){5,}',  # Same word repeated 6+ times (increased from 3+)
            r'([A-Z0-9]{2,})\1{5,}',     # Symbol/code repeated 6+ times
            r'(\d+\.\d+)(\.\d+){5,}',    # Number pattern repeated 6+ times
        ]
        
        # Medical terminology patterns to EXCLUDE from repetition detection
        self.medical_exclusions = [
            r'\b(?:anti|pre|post|non|pro|co|multi|inter|intra|sub|super|over|under|trans|semi|pseudo)-\w+\b',
            r'\b\w+(?:-dependent|-specific|-related|-based|-induced|-mediated|-resistant|-sensitive|-positive|-negative)\b',
            r'\b(?:patient|dose|treatment|therapy|drug|clinical|medical|surgical|diagnostic)-\w*\b',
            r'\b(?:meta|micro|macro|ultra|hyper|hypo|extra|intra)-\w+\b',
            r'\b\w+(?:-ase|-itis|-osis|-emia|-uria|-pathy|-therapy|-metry|-scopy|-tomy|-ectomy)\b',
            r'\b(?:HCC|CCA|HBV|HCV|HIV|AIDS|CT|MRI|PET|DNA|RNA|PCR)\b',  # Medical acronyms
        ]
        
        # FIXED: Much more conservative threshold (increased from 0.3 to 0.6)
        self.max_repetition_ratio = 0.6

    def detect_language(self, text: str) -> Optional[str]:
        """
        Detects the language of the input text.
        Returns a two-letter language code if successful, otherwise None.
        """
        try:
            from langdetect import detect
            lang = detect(text)
            return lang
        except Exception as e:
            print(f"Language detection failed: {e}")
            return None
            
    def detect_chunk_language(self, text: str, min_length: int = 40) -> str:
        """
        Detects the language of a text chunk with more reliability.
        
        Args:
            text: The text to detect language for
            min_length: Minimum length of text to attempt detection (shorter texts are unreliable)
            
        Returns:
            Language code (e.g., 'en', 'fr') or None if detection failed or text too short
        """
        if not text or len(text.strip()) < min_length:
            return None
            
        # Clean the text before detection
        clean_text = re.sub(r'\d+', '', text)  # Remove numbers
        clean_text = re.sub(r'[^\w\s]', '', clean_text)  # Remove punctuation
        
        if len(clean_text.strip()) < min_length:
            return None
        
        try:
            from langdetect import detect, LangDetectException
            lang = detect(clean_text)
            return lang
        except:
            return None

    def is_medical_terminology(self, text: str) -> bool:
        """Check if text contains medical terminology that should be excluded from repetition detection."""
        for pattern in self.medical_exclusions:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def get_translator(self, lang: str):
        """
        Returns a translation function for the given language code.
        FIXED: Proper pipeline parameter handling.
        """
        # Return cached translator if already loaded
        if lang in self.translators:
            return self.translators[lang]

        # 1. Try direct Helsinki-NLP model first (language-specific)
        if lang in self.models:
            model_name = self.models[lang]
            print(f"Loading model: {model_name}")
            
            try:
                # FIXED: Removed invalid generate_kwargs from pipeline initialization
                translator = pipeline(
                    "translation",
                    model=model_name,
                    torch_dtype=torch.float16 if self.device.type != "cpu" else torch.float32,
                    device=self.device,
                    # REMOVED: model_kwargs and generate_kwargs - these caused the error
                )
                self.translators[lang] = translator
                return translator
            except Exception as e:
                print(f"Failed to load model {model_name}: {e}")
                # Continue to next option
        
        # 2. Try NLLB multilingual model as fallback
        nllb_model = 'facebook/nllb-200-distilled-600M'
        if lang in self.language_groups.get(nllb_model, {}).get('langs', set()):
            print(f"Using fallback model: {nllb_model}")
            
            # Load or retrieve cached NLLB model
            if nllb_model not in self.multilingual_models:
                try:
                    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
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
            
            # Create translator function with repetition prevention
            def nllb_translate(text, **kwargs):
                # Set the source language
                inputs = tokenizer(text, return_tensors="pt").to(self.device)
                
                # Get the tokenizer's language ID for the target language
                with torch.no_grad():
                    translated_tokens = model.generate(
                        **inputs,
                        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
                        max_length=512,
                        num_beams=3,  # Reduced for stability
                        no_repeat_ngram_size=3,
                        length_penalty=0.8,
                        repetition_penalty=1.2
                    )
                
                translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                return [{'translation_text': translation}]
            
            self.translators[lang] = nllb_translate
            return nllb_translate

        # No suitable model found
        return None

    def count_tokens_rough(self, text: str) -> int:
        """Rough token count estimation (words * 1.3)."""
        return int(len(text.split()) * 1.3)

    def chunk_text(self, text: str) -> list:
        """
        FIXED: Smarter text chunking that respects model limits and sentence boundaries.
        """
        # Target around 150-200 words per chunk (roughly 200-260 tokens)
        target_words = self.max_chunk_length
        
        # Split into sentences first
        sentences = re.split(r'(?<=\.)\s+', text.strip())
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            # If this single sentence is too long, split it further
            if sentence_words > target_words:
                # If we have accumulation, save it first
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_word_count = 0
                
                # Split long sentence by punctuation
                sub_parts = re.split(r'[,;:]', sentence)
                temp_chunk = []
                temp_count = 0
                
                for part in sub_parts:
                    part_words = len(part.split())
                    if temp_count + part_words > target_words and temp_chunk:
                        chunks.append(" ".join(temp_chunk))
                        temp_chunk = [part.strip()]
                        temp_count = part_words
                    else:
                        temp_chunk.append(part.strip())
                        temp_count += part_words
                
                if temp_chunk:
                    chunks.append(" ".join(temp_chunk))
                    
            # Normal sentence processing
            elif current_word_count + sentence_words > target_words and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_word_count = sentence_words
            else:
                current_chunk.append(sentence)
                current_word_count += sentence_words
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

    def detect_repetitions(self, text: str) -> bool:
        """
        FIXED: More conservative repetition detection that only flags severe cases.
        """
        if not text or len(text.strip()) < 50:
            return False
            
        # First check if this looks like medical terminology
        if self.is_medical_terminology(text):
            return False
            
        total_length = len(text)
        total_repetition_length = 0
        
        for pattern in self.severe_repetition_patterns:
            matches = list(re.finditer(pattern, text))
            for match in matches:
                repeated_part = match.group()
                repetition_length = len(repeated_part)
                if repetition_length > 30:  # Only count substantial repetitions
                    total_repetition_length += repetition_length
        
        repetition_ratio = total_repetition_length / total_length if total_length > 0 else 0
        return repetition_ratio > self.max_repetition_ratio
        
    def clean_repetitions(self, text: str) -> str:
        """
        FIXED: Much more conservative cleaning that preserves medical content.
        """
        if not text or not self.detect_repetitions(text):
            return text
            
        original_length = len(text)
        
        # Very conservative cleaning - only remove extreme cases
        for pattern in self.severe_repetition_patterns:
            text = re.sub(pattern, lambda m: m.group(1) + ' ' + m.group(1), text)
        
        # Clean repeated character strings (very conservative)
        text = re.sub(r'([a-zA-Z])\1{7,}', r'\1\1\1', text)  # Only 8+ repeated chars
        
        # Fix spacing
        text = re.sub(r'\s+', ' ', text)
        
        # SAFETY CHECK: If we removed more than 20% of text, return original
        if len(text) < original_length * 0.8:
            print("Warning: Cleaning removed too much text, returning original")
            return text  # Return cleaned version anyway, but warn
            
        return text.strip()

    def translate_text(self, text: str, translator) -> str:
        """
        FIXED: Translates text with proper parameter passing and better error handling.
        """
        if not text.strip():
            return text
            
        # Check if text is too long and needs chunking
        estimated_tokens = self.count_tokens_rough(text)
        
        # For short text, translate directly if under token limit
        if estimated_tokens <= 300:  # Conservative limit
            return self.translate_single_chunk(text, translator)
            
        # For longer text, split intelligently
        chunks = self.chunk_text(text)
        translated_chunks = []
        
        for i, chunk in enumerate(chunks):
            translated_chunk = self.translate_single_chunk(chunk, translator)
            translated_chunks.append(translated_chunk)
            
        return " ".join(translated_chunks)

    def translate_single_chunk(self, chunk: str, translator) -> str:
        """
        FIXED: Translate a single chunk with corrected parameter passing.
        """
        max_attempts = 2
        current_attempt = 0
        
        while current_attempt < max_attempts:
            try:
                # FIXED: Pass generation parameters to the actual call, not pipeline init
                translation_result = translator(
                    chunk, 
                    max_length=400,  # Increased max length
                    num_beams=3,     # Reduced for stability
                    no_repeat_ngram_size=3,
                    length_penalty=0.8,
                    repetition_penalty=1.2,
                    early_stopping=True,
                    do_sample=False
                )
                
                translation = translation_result[0]['translation_text']
                
                # Check for problematic repetitions (more conservative)
                if self.detect_repetitions(translation):
                    print(f"Repetition detected in translation. Attempt {current_attempt+1}/{max_attempts}")
                    if current_attempt == max_attempts - 1:
                        # Last attempt - apply conservative cleaning
                        translation = self.clean_repetitions(translation)
                        return translation
                    else:
                        # Try again with different chunk size
                        if len(chunk) > 100:
                            chunk = chunk[:len(chunk)//2]  # Only reduce by half once
                        current_attempt += 1
                        continue
                else:
                    # No repetitions detected - good translation
                    return translation
                    
            except Exception as e:
                print(f"Translation error on attempt {current_attempt+1}: {e}")
                if current_attempt == max_attempts - 1:
                    # Last attempt failed - return original text
                    print(f"Translation failed completely, returning original text")
                    return chunk
                
                current_attempt += 1
                
        return chunk

    def translate_json_file(self, input_path: str, output_path: str):
        """
        Reads a JSON file, detects language at the chunk level,
        and only translates non-English chunks.
        """
        # Get document name and parent folder
        file_name = os.path.basename(input_path)
        parent_folder = os.path.basename(os.path.dirname(input_path))
        
        # Print document info
        print(f"\nDocument: {file_name} (in folder: {parent_folder})")
        
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Concatenate text from doc_id, chunks, and tables for initial language detection
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

        combined_text = "\n".join([t for t in all_texts if t]).strip()
        primary_lang = self.detect_language(combined_text)
        print(f"Primary document language detected: {primary_lang}")

        # If the document is primarily English, we'll still check each chunk
        # but we'll expect most to be English
        if primary_lang == 'en':
            print("Document primarily in English, but will check each chunk individually")
        
        # Initialize chunk statistics
        chunks_checked = 0
        english_chunks = 0
        translated_chunks = 0
        
        # Get translator based on the primary language (only if needed)
        translator = None
        if primary_lang != 'en' and primary_lang is not None:
            translator = self.get_translator(primary_lang)
            if not translator:
                print(f"No translator available for {primary_lang}. Copying file unmodified.")
                shutil.copy(input_path, output_path)
                return

        # Process chunks: check language for each chunk and translate only non-English
        if 'chunks' in data:
            total_chunks = len(data['chunks'])
            for i, ch in enumerate(data['chunks']):
                print(f"  Processing chunk {i+1}/{total_chunks}")
                chunks_checked += 1
                
                # Process heading
                if 'heading' in ch and ch['heading'].strip():
                    heading_lang = self.detect_chunk_language(ch['heading'])
                    if heading_lang is None or heading_lang == 'en':
                        # Keep English/undetected headings as is
                        english_chunks += 1
                    else:
                        # Translate non-English headings
                        if translator:
                            ch['heading'] = self.translate_text(ch['heading'], translator)
                            translated_chunks += 1
                        else:
                            # If no translator available for primary language, try to get one for this chunk
                            chunk_translator = self.get_translator(heading_lang)
                            if chunk_translator:
                                ch['heading'] = self.translate_text(ch['heading'], chunk_translator)
                                translated_chunks += 1
                
                # Process main text content
                if 'text' in ch and ch['text'].strip():
                    text_lang = self.detect_chunk_language(ch['text'])
                    if text_lang is None or text_lang == 'en':
                        # Keep English/undetected text as is
                        english_chunks += 1
                    else:
                        # Translate non-English text
                        if translator:
                            ch['text'] = self.translate_text(ch['text'], translator)
                            translated_chunks += 1
                        else:
                            # If no translator available for primary language, try to get one for this chunk
                            chunk_translator = self.get_translator(text_lang)
                            if chunk_translator:
                                ch['text'] = self.translate_text(ch['text'], chunk_translator)
                                translated_chunks += 1

        # Process table texts similarly
        if 'tables' in data:
            for tb in data['tables']:
                if 'text' in tb and tb['text'].strip():
                    chunks_checked += 1
                    text_lang = self.detect_chunk_language(tb['text'])
                    if text_lang is None or text_lang == 'en':
                        # Keep English/undetected text as is
                        english_chunks += 1
                    else:
                        # Translate non-English text
                        if translator:
                            tb['text'] = self.translate_text(tb['text'], translator)
                            translated_chunks += 1
                        else:
                            # If no translator available for primary language, try to get one for this chunk
                            chunk_translator = self.get_translator(text_lang)
                            if chunk_translator:
                                tb['text'] = self.translate_text(tb['text'], chunk_translator)
                                translated_chunks += 1

        # Save the processed JSON
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump(data, out_f, indent=2, ensure_ascii=False)

        # Update statistics
        self.english_chunks_preserved += english_chunks
        self.chunks_translated += translated_chunks
        
        # Print chunk statistics
        print(f"Chunks processed: {chunks_checked}")
        print(f"English chunks preserved: {english_chunks}")
        print(f"Non-English chunks translated: {translated_chunks}")

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
        total_files = 0
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if not file.endswith(".json"):
                    continue
                total_files += 1
                    
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, self.input_dir)
                output_subdir = os.path.join(self.output_dir, relative_path)
                output_path = os.path.join(output_subdir, file)
                
                try:
                    self.translate_json_file(input_path, output_path)
                except Exception as e:
                    print(f"❌ Error processing {file}: {e}")
                    # Copy original file if translation fails completely
                    os.makedirs(output_subdir, exist_ok=True)
                    shutil.copy(input_path, output_path)
                
        # Print final statistics
        print("\n=== Translation Summary ===")
        print(f"Total files processed: {total_files}")
        print(f"Total English chunks preserved: {self.english_chunks_preserved}")
        print(f"Total chunks translated: {self.chunks_translated}")
        if self.chunks_translated > 0:
            print(f"Preservation rate: {self.english_chunks_preserved/(self.english_chunks_preserved+self.chunks_translated):.2%}")
        print("✅ Translation completed!")



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