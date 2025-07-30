import os
import re
import json
import statistics
import pdfplumber
from collections import defaultdict
import numpy as np
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

    def merge_hyphenated_words(self, lines: List[str]) -> List[str]:
        """
        Merge hyphenated words that are split across lines.
        This handles cases like 'signifi-' on one line and 'cant' on the next.
        """
        if not lines:
            return lines
            
        cleaned_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i].rstrip()
            
            # Check if current line ends with hyphen and there's a next line
            if i < len(lines) - 1 and line.endswith('-'):
                next_line = lines[i + 1].lstrip()
                
                # Only merge if the next line starts with a lowercase letter (continuation)
                # or if it's clearly a continuation (no punctuation at start)
                if (next_line and 
                    (next_line[0].islower() or 
                     not re.match(r'^[A-Z\d\(\[\{]', next_line))):
                    
                    # Remove hyphen and merge with next line
                    merged = line[:-1] + next_line
                    cleaned_lines.append(merged)
                    i += 2  # Skip the next line as it's been merged
                else:
                    # Keep the hyphen if it doesn't look like a word break
                    cleaned_lines.append(line)
                    i += 1
            else:
                cleaned_lines.append(line)
                i += 1
                
        return cleaned_lines

    def improve_paragraph_cohesion(self, lines: List[str]) -> List[str]:
        """
        Improve paragraph cohesion by joining lines that appear to be 
        continuation of the same sentence or paragraph.
        """
        if not lines:
            return lines
            
        cohesive_lines = []
        i = 0
        
        while i < len(lines):
            current_line = lines[i].strip()
            
            if not current_line:
                cohesive_lines.append(current_line)
                i += 1
                continue
            
            # Look ahead to see if we should merge with next line
            if i < len(lines) - 1:
                next_line = lines[i + 1].strip()
                
                # Conditions for merging:
                # 1. Current line doesn't end with sentence-ending punctuation
                # 2. Next line exists and isn't empty
                # 3. Next line starts with lowercase (likely continuation)
                # 4. Current line doesn't look like a heading or list item
                should_merge = (
                    current_line and next_line and
                    not re.search(r'[.!?:;]$', current_line) and
                    next_line[0].islower() and
                    not re.match(r'^\d+[\.\)]\s', current_line) and  # Not a numbered list
                    not re.match(r'^[•\-\*]\s', current_line) and   # Not a bullet list
                    len(current_line.split()) > 1  # Not a single word (likely heading)
                )
                
                if should_merge:
                    # Merge current line with next line
                    merged_line = current_line + ' ' + next_line
                    cohesive_lines.append(merged_line)
                    i += 2  # Skip the next line
                else:
                    cohesive_lines.append(current_line)
                    i += 1
            else:
                cohesive_lines.append(current_line)
                i += 1
                
        return cohesive_lines

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

    def extract_table_title_from_page(self, page, table_bbox=None):
            """
            Extract table title by looking for text patterns above or near the table.
            
            Args:
                page: pdfplumber page object
                table_bbox: Bounding box of the table (x0, top, x1, bottom) if available
                
            Returns:
                String containing the extracted table title or None if not found
            """
            try:
                # Extract all text with positioning information
                words = page.extract_words(x_tolerance=2, y_tolerance=2)
                if not words:
                    return None
                
                # If we have table bbox, focus search area above the table
                search_area_words = []
                if table_bbox:
                    x0, top, x1, bottom = table_bbox
                    # Look for titles in the area above the table (within 100 points)
                    for word in words:
                        word_bottom = word.get('bottom', word.get('y1', 0))
                        word_left = word.get('x0', 0)
                        word_right = word.get('x1', word.get('x0', 0))
                        
                        # Check if word is above table and horizontally aligned
                        if (word_bottom <= top and 
                            word_bottom >= top - 100 and  # Within 100 points above
                            word_right >= x0 - 20 and    # Allow some horizontal tolerance
                            word_left <= x1 + 20):
                            search_area_words.append(word)
                else:
                    # Search entire page if no bbox available
                    search_area_words = words
                
                # Group words into lines
                lines = self._group_words_into_lines(search_area_words)
                
                # Look for table title patterns
                for line in lines:
                    title = self._extract_title_from_line(line)
                    if title:
                        return title
                        
                return None
                
            except Exception as e:
                print(f"Error extracting table title: {e}")
                return None

    def _group_words_into_lines(self, words):
        """Group words into lines based on vertical position."""
        if not words:
            return []
            
        # Sort words by vertical position, then horizontal
        sorted_words = sorted(words, key=lambda w: (w.get('top', 0), w.get('x0', 0)))
        
        lines = []
        current_line = []
        current_y = None
        y_tolerance = 5
        
        for word in sorted_words:
            word_y = word.get('top', 0)
            
            if current_y is None or abs(word_y - current_y) <= y_tolerance:
                current_line.append(word)
                current_y = word_y
            else:
                if current_line:
                    line_text = ' '.join(w.get('text', '') for w in current_line)
                    lines.append(line_text.strip())
                current_line = [word]
                current_y = word_y
        
        # Add the last line
        if current_line:
            line_text = ' '.join(w.get('text', '') for w in current_line)
            lines.append(line_text.strip())
            
        return lines

    def _extract_title_from_line(self, line_text):
        """
        Extract table title from a line of text using pattern matching.
        
        Returns the cleaned title or None if no title pattern is found.
        """
        if not line_text or len(line_text.strip()) < 3:
            return None
            
        line = line_text.strip()
        
        # Table title patterns (case insensitive)
        title_patterns = [
            # Direct table references
            r'^Table\s+\d+[:\.\-\s]+(.+)$',
            r'^Table\s+[A-Z]?[:\.\-\s]+(.+)$',
            r'^Tabla\s+\d+[:\.\-\s]+(.+)$',  # Spanish
            r'^Tableau\s+\d+[:\.\-\s]+(.+)$',  # French
            r'^Tabelle\s+\d+[:\.\-\s]+(.+)$',  # German
            r'^Tabela\s+\d+[:\.\-\s]+(.+)$',  # Portuguese/Polish
            
            # Figure references that might be tables
            r'^Figure\s+\d+[:\.\-\s]+(.+)$',
            r'^Fig\.?\s+\d+[:\.\-\s]+(.+)$',
            r'^Figura\s+\d+[:\.\-\s]+(.+)$',  # Spanish/Italian
            
            # Appendix tables
            r'^Appendix\s+[A-Z]?[:\.\-\s]*Table\s*\d*[:\.\-\s]+(.+)$',
            r'^Annex\s+[A-Z]?[:\.\-\s]*Table\s*\d*[:\.\-\s]+(.+)$',
            
            # Summary/overview patterns
            r'^Summary\s+of\s+(.+)$',
            r'^Overview\s+of\s+(.+)$',
            r'^Comparison\s+of\s+(.+)$',
            r'^Results\s+of\s+(.+)$',
            
            # Clinical trial specific patterns
            r'^Efficacy\s+(.+)$',
            r'^Safety\s+(.+)$',
            r'^Adverse\s+Events?\s+(.+)$',
            r'^Patient\s+Characteristics\s+(.+)$',
            r'^Baseline\s+Characteristics\s+(.+)$',
            r'^Treatment\s+Outcomes?\s+(.+)$',
            
            # Generic patterns (more permissive)
            r'^(.+)\s+\(Table\)$',
            r'^(.+)\s+\(continued\)$',
        ]
        
        # Try each pattern
        for pattern in title_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                extracted_title = match.group(1).strip()
                # Clean up the extracted title
                cleaned_title = self._clean_table_title(extracted_title)
                if cleaned_title and len(cleaned_title) > 2:
                    return f"Table: {cleaned_title}"
        
        # Check if the entire line looks like a table title
        if self._line_looks_like_table_title(line):
            cleaned_title = self._clean_table_title(line)
            if cleaned_title and len(cleaned_title) > 2:
                return f"Table: {cleaned_title}"
                
        return None

    def _clean_table_title(self, title):
        """Clean and format the extracted table title."""
        if not title:
            return None
            
        # Remove common artifacts
        cleaned = title.strip()
        
        # Remove trailing punctuation marks except periods
        cleaned = re.sub(r'[,;:\-]+$', '', cleaned)
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove page numbers and references at the end
        cleaned = re.sub(r'\s+\d+\s*$', '', cleaned)
        
        # Remove common prefixes that might be artifacts
        prefixes_to_remove = [
            r'^Table\s+\d+[\.\:\-\s]*',
            r'^Figure\s+\d+[\.\:\-\s]*',
            r'^Fig\.?\s+\d+[\.\:\-\s]*',
        ]
        
        for prefix in prefixes_to_remove:
            cleaned = re.sub(prefix, '', cleaned, flags=re.IGNORECASE)
        
        return cleaned.strip()

    def _line_looks_like_table_title(self, line):
        """
        Determine if a line looks like it could be a table title based on content and structure.
        """
        if not line or len(line.strip()) < 5:
            return False
            
        line = line.strip()
        
        # Characteristics of table titles
        title_indicators = [
            # Contains medical/clinical keywords
            any(keyword in line.lower() for keyword in [
                'efficacy', 'safety', 'adverse', 'patient', 'treatment', 'outcome', 
                'baseline', 'characteristics', 'comparison', 'summary', 'overview',
                'results', 'analysis', 'endpoints', 'demographics', 'response',
                'survival', 'progression', 'toxicity', 'dose', 'study', 'trial'
            ]),
            
            # Reasonable length for a title (not too short, not too long)
            5 <= len(line.split()) <= 15,
            
            # Starts with capital letter
            line[0].isupper(),
            
            # Contains proper nouns or medical terms (indicated by capitals)
            sum(1 for word in line.split() if word[0].isupper()) >= 2,
            
            # Doesn't look like a sentence (no common sentence starters)
            not any(line.lower().startswith(starter) for starter in [
                'the ', 'this ', 'these ', 'those ', 'a ', 'an ', 'in ', 'at ', 
                'on ', 'for ', 'with ', 'by ', 'from ', 'to ', 'of '
            ])
        ]
        
        # Must meet at least 2 criteria
        return sum(title_indicators) >= 2

    def get_table_bounding_box(self, table_data, page):
        """
        Estimate the bounding box of a table based on its content.
        This is a rough estimation for title search purposes.
        """
        try:
            if not table_data or not page:
                return None
                
            # For now, return None to use page-wide search
            # This could be enhanced with more sophisticated bbox detection
            return None
            
        except Exception:
            return None

    def detect_complete_tables(self, pdf):
        """
        Enhanced table detection that only identifies genuine tabular data.
        Reduces false positives by validating table characteristics.
        """
        tables_info = []
        
        for page_num, page in enumerate(pdf.pages, start=1):
            # Step 1: Extract potential tables with stricter settings
            page_tables = self.extract_tables_strict(page)
            
            if not page_tables:
                continue
            
            # Step 2: Validate each potential table
            for table_idx, table_data in enumerate(page_tables, start=1):
                if not table_data or len(table_data) < 2:
                    continue
                
                # Step 3: Apply comprehensive table validation
                if not self.is_genuine_table(table_data, page, page_num):
                    print(f"    Rejected table candidate on page {page_num} (not a genuine table)")
                    continue
                
                # Step 4: Extract title only for validated tables
                table_bbox = self.get_table_bounding_box(table_data, page)
                extracted_title = self.extract_table_title_from_page(page, table_bbox)
                
                # Step 5: Convert validated table to narrative
                narrative_text = self.convert_table_to_narrative(table_data)
                
                if narrative_text.strip():
                    table_title = extracted_title if extracted_title else self.find_table_heading(page_num, table_idx)
                    print(f"    ✓ Validated genuine table on page {page_num}: '{table_title}'")
                    
                    tables_info.append({
                        "page": page_num,
                        "heading": table_title,
                        "text": narrative_text,
                        "table_type": "validated_table"
                    })
        
        return tables_info

    def extract_tables_strict(self, page):
        """
        Extract tables with stricter settings to reduce false positives.
        """
        try:
            # Use more conservative table detection settings
            tables = page.extract_tables(
                table_settings={
                    "vertical_strategy": "lines",      # Require actual lines
                    "horizontal_strategy": "lines",    # Require actual lines
                    "min_words_vertical": 3,           # Need multiple words per column
                    "min_words_horizontal": 2,         # Need multiple words per row
                    "keep_blank_chars": False,         # Remove blank chars
                    "text_tolerance": 3,               # Stricter text alignment
                    "intersection_tolerance": 3,       # Stricter intersection
                }
            )
            return tables
        except:
            # Fallback to default if strict settings fail
            return page.extract_tables()

    def is_genuine_table(self, table_data, page, page_num):
        """
        Comprehensive validation to determine if detected data is a genuine table.
        
        Returns True only if the data exhibits strong tabular characteristics.
        """
        if not table_data or len(table_data) < 3:  # Need at least header + 2 data rows
            return False
        
        # Clean the table data first
        cleaned_table = self.clean_table_data(table_data)
        if len(cleaned_table) < 3:
            return False
        
        # Validation criteria
        validations = {
            'structure': self._validate_table_structure(cleaned_table),
            'content': self._validate_table_content(cleaned_table),
            'visual': self._validate_table_visual_cues(table_data, page),
            'context': self._validate_table_context(cleaned_table, page_num),
            'data_patterns': self._validate_data_patterns(cleaned_table)
        }
        
        # Score each validation (0.0 to 1.0)
        scores = {k: v for k, v in validations.items()}
        
        # Weighted scoring - all criteria must be reasonably satisfied
        weights = {
            'structure': 0.25,    # Consistent column structure
            'content': 0.25,      # Appropriate content types
            'visual': 0.20,       # Visual table indicators
            'context': 0.15,      # Contextual clues
            'data_patterns': 0.15 # Data type patterns
        }
        
        total_score = sum(weights[k] * scores[k] for k in weights.keys())
        
        # Debug output
        print(f"      Table validation scores: {scores}")
        print(f"      Total score: {total_score:.3f}")
        
        # Require high confidence (0.7+) to accept as genuine table
        return total_score >= 0.7

    def _validate_table_structure(self, cleaned_table):
        """Validate structural consistency of the table."""
        if len(cleaned_table) < 3:
            return 0.0
        
        # Check column consistency
        expected_cols = len(cleaned_table[0])
        if expected_cols < 2:  # Tables need at least 2 columns
            return 0.0
        
        consistent_rows = 0
        for row in cleaned_table:
            if len(row) == expected_cols:
                consistent_rows += 1
        
        consistency_ratio = consistent_rows / len(cleaned_table)
        
        # Bonus for having reasonable number of columns (2-8 is typical)
        col_bonus = 1.0 if 2 <= expected_cols <= 8 else 0.5
        
        return consistency_ratio * col_bonus

    def _validate_table_content(self, cleaned_table):
        """Validate that content looks like tabular data."""
        if len(cleaned_table) < 2:
            return 0.0
        
        header_row = cleaned_table[0]
        data_rows = cleaned_table[1:]
        
        # Check if first row looks like headers
        header_score = 0.0
        if header_row:
            header_indicators = [
                # Short, descriptive headers
                all(len(cell.split()) <= 5 for cell in header_row if cell.strip()),
                # Contains typical header words
                any(word.lower() in ' '.join(header_row).lower() 
                    for word in ['name', 'value', 'result', 'type', 'dose', 'group', 'arm', 
                            'endpoint', 'outcome', 'parameter', 'characteristic', 'n=', 
                            'patient', 'treatment', 'study', 'baseline', 'safety', 'efficacy']),
                # Headers are different from data patterns
                not all(self.is_numeric(cell) for cell in header_row if cell.strip())
            ]
            header_score = sum(header_indicators) / len(header_indicators)
        
        # Check if data rows contain appropriate data
        data_score = 0.0
        if data_rows:
            numeric_columns = 0
            total_columns = len(header_row) if header_row else len(data_rows[0])
            
            for col_idx in range(total_columns):
                column_values = [row[col_idx] if col_idx < len(row) else "" 
                            for row in data_rows]
                column_values = [v for v in column_values if v.strip()]
                
                if column_values:
                    # Check if column contains mostly numeric data (good for tables)
                    numeric_count = sum(1 for v in column_values if self.is_numeric(v))
                    if numeric_count / len(column_values) > 0.5:
                        numeric_columns += 1
            
            # Tables should have at least one numeric column
            data_score = min(1.0, numeric_columns / max(total_columns, 1) + 0.3)
        
        return (header_score + data_score) / 2

    def _validate_table_visual_cues(self, table_data, page):
        """Look for visual cues that indicate a genuine table."""
        try:
            # Check for table-like visual elements in the original PDF
            visual_score = 0.0
            
            # Look for lines/borders (tables often have visible structure)
            lines = page.lines if hasattr(page, 'lines') else []
            if lines:
                # More lines might indicate table structure
                line_density = len(lines) / (page.width * page.height) * 10000
                visual_score += min(0.3, line_density / 10)
            
            # Look for consistent spacing patterns
            if table_data and len(table_data) > 2:
                # Check if rows have similar structure
                row_lengths = [len([cell for cell in row if cell and cell.strip()]) 
                            for row in table_data]
                if row_lengths:
                    length_consistency = 1.0 - (max(row_lengths) - min(row_lengths)) / max(max(row_lengths), 1)
                    visual_score += length_consistency * 0.4
            
            # Check for table-related words in surrounding text
            page_text = page.extract_text() or ""
            table_keywords = ['table', 'tableau', 'tabelle', 'tabela', 'figura', 'fig']
            keyword_found = any(keyword in page_text.lower() for keyword in table_keywords)
            if keyword_found:
                visual_score += 0.3
            
            return min(1.0, visual_score)
        
        except Exception:
            return 0.5  # Neutral score if visual analysis fails

    def _validate_table_context(self, cleaned_table, page_num):
        """Validate contextual clues about table likelihood."""
        context_score = 0.0
        
        # Check if we have a reasonable number of rows (3-50 is typical for most tables)
        row_count = len(cleaned_table)
        if 3 <= row_count <= 50:
            context_score += 0.4
        elif row_count > 50:
            context_score += 0.1  # Very large "tables" are often misdetected text
        
        # Check if column headers make sense
        if cleaned_table:
            first_row = cleaned_table[0]
            if any(len(cell.split()) > 10 for cell in first_row if cell.strip()):
                # Headers shouldn't be very long sentences
                context_score -= 0.3
            else:
                context_score += 0.3
        
        # Bonus for medical/clinical table indicators
        all_text = ' '.join(' '.join(row) for row in cleaned_table)
        medical_indicators = [
            'patient', 'dose', 'mg', 'efficacy', 'safety', 'adverse', 'endpoint',
            'baseline', 'treatment', 'placebo', 'n=', '%', 'ci', 'p-value', 'hr', 'or'
        ]
        
        medical_matches = sum(1 for indicator in medical_indicators 
                            if indicator in all_text.lower())
        if medical_matches >= 3:
            context_score += 0.3
        
        return max(0.0, min(1.0, context_score))

    def _validate_data_patterns(self, cleaned_table):
        """Validate that the data exhibits typical table patterns."""
        if len(cleaned_table) < 2:
            return 0.0
        
        pattern_score = 0.0
        
        # Check for mixed data types across columns (typical of tables)
        if len(cleaned_table[0]) >= 2:
            col_types = []
            for col_idx in range(len(cleaned_table[0])):
                column_data = [row[col_idx] if col_idx < len(row) else "" 
                            for row in cleaned_table[1:]]  # Skip header
                column_data = [cell for cell in column_data if cell.strip()]
                
                if column_data:
                    numeric_ratio = sum(1 for cell in column_data if self.is_numeric(cell)) / len(column_data)
                    if numeric_ratio > 0.7:
                        col_types.append('numeric')
                    elif numeric_ratio < 0.3:
                        col_types.append('text')
                    else:
                        col_types.append('mixed')
            
            # Good tables often have mix of text and numeric columns
            if 'numeric' in col_types and 'text' in col_types:
                pattern_score += 0.5
            elif 'numeric' in col_types:
                pattern_score += 0.3
        
        # Check for consistent formatting patterns
        consistent_patterns = 0
        total_patterns = 0
        
        for row in cleaned_table[1:]:  # Skip header
            for cell in row:
                if cell.strip():
                    total_patterns += 1
                    # Look for consistent patterns like percentages, measurements, etc.
                    if re.match(r'^\d+\.?\d*%?$', cell.strip()) or \
                    re.match(r'^\d+\.?\d*\s*(mg|%|ml|cm|mm)$', cell.strip().lower()) or \
                    re.match(r'^\d+/\d+$', cell.strip()) or \
                    re.match(r'^\d+\.?\d*\s*\(\d+\.?\d*[-–]\d+\.?\d*\)$', cell.strip()):
                        consistent_patterns += 1
        
        if total_patterns > 0:
            pattern_consistency = consistent_patterns / total_patterns
            pattern_score += pattern_consistency * 0.5
        
        return min(1.0, pattern_score)

    def is_numeric(self, text):
        """Enhanced numeric detection for medical/scientific data."""
        if not text or not text.strip():
            return False
        
        text = text.strip()
        
        # Common numeric patterns in medical/scientific tables
        numeric_patterns = [
            r'^\d+\.?\d*$',                          # Simple numbers: 123, 12.5
            r'^\d+\.?\d*%$',                         # Percentages: 45.2%
            r'^\d+/\d+$',                            # Fractions: 15/20
            r'^[<>=≤≥]\s*\d+\.?\d*$',               # Comparisons: <0.05, ≥50
            r'^\d+\.?\d*\s*\([^)]+\)$',             # With confidence intervals: 1.2 (0.8-1.6)
            r'^\d+\.?\d*\s*(mg|ml|cm|mm|%|kg)$',    # With units: 50mg, 2.5ml
            r'^n\s*=\s*\d+$',                       # Sample sizes: n=150
            r'^\d+\.?\d*[-–]\d+\.?\d*$',            # Ranges: 10-20, 1.5–2.8
            r'^\d{1,3}(,\d{3})*(\.\d+)?$',          # Comma-separated: 1,234.56
        ]
        
        return any(re.match(pattern, text, re.IGNORECASE) for pattern in numeric_patterns)

    def merge_related_tables(self, page_tables, page):
        """
        Merge tables that appear to be parts of the same logical table.
        This helps when a table is split across areas or columns.
        """
        if len(page_tables) <= 1:
            return page_tables
        
        merged = []
        used_indices = set()
        
        for i, table1 in enumerate(page_tables):
            if i in used_indices:
                continue
                
            # Start with this table
            combined_table = list(table1) if table1 else []
            used_indices.add(i)
            
            # Look for tables that might be continuations
            for j, table2 in enumerate(page_tables[i+1:], start=i+1):
                if j in used_indices or not table2:
                    continue
                
                # Check if table2 might be a continuation of table1
                if self.tables_appear_related(table1, table2):
                    # Merge table2 into combined_table
                    combined_table.extend(table2)
                    used_indices.add(j)
            
            if combined_table:
                merged.append(combined_table)
        
        return merged

    def tables_appear_related(self, table1, table2):
        """
        Determine if two tables appear to be parts of the same logical table.
        """
        if not table1 or not table2:
            return False
        
        # Check if they have similar column structures
        if len(table1[0]) == len(table2[0]):
            # Similar number of columns - might be continuation
            return True
        
        # Check if one appears to be a header and the other data
        if len(table1) == 1 and len(table2) > 1:
            return True
        
        return False

    def convert_table_to_narrative(self, table_data):
        """
        Convert a table into narrative text that preserves the relationships.
        Each row becomes a descriptive sentence or bullet point.
        """
        if not table_data or len(table_data) == 0:
            return ""
        
        # Clean and prepare the table
        cleaned_table = self.clean_table_data(table_data)
        
        if len(cleaned_table) == 0:
            return ""
        
        # Try to identify headers
        headers = self.identify_table_headers(cleaned_table)
        
        if headers:
            # Convert each data row to narrative
            narrative_parts = []
            
            # Add header information
            narrative_parts.append(f"Table contains the following columns: {', '.join(headers)}")
            
            # Process data rows
            data_rows = cleaned_table[1:] if len(cleaned_table) > 1 else []
            
            for row_idx, row in enumerate(data_rows, 1):
                row_description = self.create_row_description(headers, row, row_idx)
                if row_description:
                    narrative_parts.append(row_description)
            
            return "\n".join(narrative_parts)
        else:
            # No clear headers - treat as a simple data table
            return self.convert_headerless_table(cleaned_table)

    def clean_table_data(self, table_data):
        """
        Clean and standardize table data.
        """
        cleaned = []
        
        for row in table_data:
            if not row:
                continue
            
            cleaned_row = []
            for cell in row:
                if cell is None:
                    cell_text = ""
                else:
                    # Clean cell content
                    cell_text = re.sub(r'\s+', ' ', str(cell)).strip()
                    
                cleaned_row.append(cell_text)
            
            # Only include rows that have some content
            if any(cell.strip() for cell in cleaned_row):
                cleaned.append(cleaned_row)
        
        return cleaned

    def identify_table_headers(self, cleaned_table):
        """
        Try to identify table headers from the first row(s).
        """
        if not cleaned_table:
            return None
        
        first_row = cleaned_table[0]
        
        # Check if first row looks like headers
        header_indicators = [
            # Check for typical header words
            any(word.lower() in cell.lower() for word in ['name', 'type', 'value', 'result', 'outcome', 'dose', 'drug', 'treatment', 'group', 'arm', 'study', 'n=', 'patient', 'endpoint', 'efficacy', 'safety', 'adverse', 'event'] for cell in first_row if cell),
            
            # Check if cells are short (typical of headers)
            all(len(cell.split()) <= 4 for cell in first_row if cell),
            
            # Check if subsequent rows have different content patterns
            len(cleaned_table) > 1 and any(
                self.is_numeric(cell) for cell in cleaned_table[1] if cell
            )
        ]
        
        if any(header_indicators):
            return [cell if cell else f"Column_{i+1}" for i, cell in enumerate(first_row)]
        
        return None

    def is_numeric(self, text):
        """
        Check if text contains numeric data (percentages, numbers, etc.)
        """
        if not text:
            return False
        
        # Check for common numeric patterns
        numeric_patterns = [
            r'\d+\.?\d*%',  # Percentages
            r'\d+\.?\d*',   # Numbers
            r'\d+/\d+',     # Fractions
            r'[<>=]?\s*\d+\.?\d*',  # Comparison operators with numbers
            r'\d+\.\d+\s*\([^)]+\)',  # Numbers with confidence intervals
        ]
        
        return any(re.search(pattern, text) for pattern in numeric_patterns)

    def create_row_description(self, headers, row, row_idx):
        """
        Create a descriptive sentence for a table row.
        """
        if not headers or not row:
            return ""
        
        # Pair headers with values
        pairs = []
        for i, (header, value) in enumerate(zip(headers, row)):
            if value and value.strip():
                # Clean header and value
                clean_header = header.strip().rstrip(':')
                clean_value = value.strip()
                
                pairs.append(f"{clean_header}: {clean_value}")
        
        if not pairs:
            return ""
        
        # Create a descriptive sentence
        if len(pairs) == 1:
            return f"Row {row_idx}: {pairs[0]}"
        elif len(pairs) == 2:
            return f"Row {row_idx}: {pairs[0]} and {pairs[1]}"
        else:
            # Multiple pairs - create a structured description
            return f"Row {row_idx}: {', '.join(pairs[:-1])}, and {pairs[-1]}"

    def convert_headerless_table(self, cleaned_table):
        """
        Convert a table without clear headers into narrative form.
        """
        narrative_parts = []
        
        for row_idx, row in enumerate(cleaned_table, 1):
            # Filter out empty cells
            non_empty_cells = [cell for cell in row if cell and cell.strip()]
            
            if non_empty_cells:
                if len(non_empty_cells) == 1:
                    narrative_parts.append(f"Row {row_idx}: {non_empty_cells[0]}")
                else:
                    # Join multiple cells with descriptive text
                    cell_desc = ", ".join(f"'{cell}'" for cell in non_empty_cells)
                    narrative_parts.append(f"Row {row_idx} contains: {cell_desc}")
        
        return "\n".join(narrative_parts)

    def extract_text_by_columns(self, page):
        """
        Extract text from a page considering column layout.
        Returns text organized by columns and preserving reading order.
        Enhanced with hyphenation and paragraph cohesion improvements.
        """
        num_columns, column_boundaries = self.detect_columns(page)
        
        if num_columns == 1:
            # For single column, extract text and apply improvements
            column_text = page.extract_text()
            if column_text:
                lines = column_text.split('\n')
                # Apply hyphenation merging
                lines = self.merge_hyphenated_words(lines)
                # Apply paragraph cohesion
                lines = self.improve_paragraph_cohesion(lines)
                return ['\n'.join(lines)]
            return [""]
        
        # For multi-column layout, extract text for each column separately
        column_texts = []
        
        for i in range(num_columns):
            left_bound = column_boundaries[i]
            right_bound = column_boundaries[i+1]
            
            # Extract text only within this column's boundaries
            column_area = (left_bound, 0, right_bound, page.height)
            column_text = page.crop(column_area).extract_text()
            
            if column_text and column_text.strip():
                lines = column_text.split('\n')
                # Apply hyphenation merging
                lines = self.merge_hyphenated_words(lines)
                # Apply paragraph cohesion
                lines = self.improve_paragraph_cohesion(lines)
                column_texts.append('\n'.join(lines))
        
        return column_texts

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

    def extract_preliminary_chunks(self):
        """
        Main function that:
        1. Extracts text and identifies headings from each page (skipping footnotes, boilerplate, etc.).
        2. Stores headings per page in self.page_headings_map.
        3. Extracts tables using enhanced table detection and narrative conversion.
        4. Returns a dictionary with enhanced text processing including hyphenation and paragraph cohesion.
        Enhanced with better error handling for problematic PDFs and improved table handling.
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
                        
                        # Extract text column by column with enhanced processing
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

                # Now extract tables with enhanced detection and narrative conversion
                tables_info = self.detect_complete_tables(pdf)

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
        Enhanced with hyphenation and paragraph cohesion improvements.
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
                        
                        # Apply text improvements to fallback method too
                        lines = text.split('\n')
                        lines = self.merge_hyphenated_words(lines)
                        lines = self.improve_paragraph_cohesion(lines)
                        improved_text = '\n'.join(lines)
                        
                        # Check for potential section headings
                        lines = improved_text.split('\n')
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
                            current_chunk_text = [improved_text]
                            current_heading = potential_heading
                            start_page = page_num
                        else:
                            # Continue with current chunk
                            current_chunk_text.append(improved_text)
                        
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
from datetime import datetime  # ADD THIS LINE
from typing import Optional, List, Dict, Any, Tuple
from langdetect import detect
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import copy

class Translator:
    """
    Enhanced Translator class with document-level hierarchical model selection optimized for medical data,
    robust CUDA handling, medical term preservation, intelligent quality detection, and adaptive translation chunking.
    """

    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.english_chunks_preserved = 0
        self.chunks_translated = 0

        self.processing_start_time = None
        self.tier_attempts = {}  # Track tier attempts per document
        self.translation_decisions = {}  # Track translation decisions

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

        # Updated hierarchical model configuration with verified models
        self.model_tiers = {
            1: {  # Tier 1: Fast models for initial translation
                'helsinki_models': {
                    'fr': 'Helsinki-NLP/opus-mt-fr-en',
                    'de': 'Helsinki-NLP/opus-mt-de-en',
                    'es': 'Helsinki-NLP/opus-mt-es-en',
                    'it': 'Helsinki-NLP/opus-mt-it-en',
                    'nl': 'Helsinki-NLP/opus-mt-nl-en',
                    'pl': 'Helsinki-NLP/opus-mt-pl-en',
                    'pt': 'Helsinki-NLP/opus-mt-pt-en',
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
                    'lt': 'Helsinki-NLP/opus-mt-lt-en',
                    'el': 'Helsinki-NLP/opus-mt-el-en',
                    'ro': 'Helsinki-NLP/opus-mt-ro-en',
                    'tr': 'Helsinki-NLP/opus-mt-tr-en',
                },
                'fallback': 'facebook/nllb-200-distilled-600M',
                'description': 'Fast Helsinki models with NLLB fallback'
            },
            2: {  # Tier 2: Higher quality models for retranslation if needed
                'nllb_large': 'facebook/nllb-200-1.3B',
                'nllb_distilled': 'facebook/nllb-200-distilled-1.3B',
                'fallback': 'facebook/nllb-200-distilled-600M',
                'description': 'Large NLLB models for improved quality'
            },
            3: {  # Tier 3: Best available models (if implemented)
                'nllb_xl': 'facebook/nllb-200-3.3B',
                'm2m100_large': 'facebook/m2m100_1.2B',
                'fallback': 'facebook/nllb-200-1.3B',
                'description': 'Highest quality models'
            }
        }

        # Language mapping for NLLB models
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

        # Document-level quality thresholds for tier escalation
        self.document_quality_thresholds = {
            'tier_1_to_2_threshold': 0.65,  # If avg quality < 0.65, try Tier 2
            'minimum_acceptable_quality': 0.50,  # Absolute minimum before giving up
            'medical_preservation_threshold': 0.70,  # Critical for medical docs
            'repetition_penalty_threshold': 0.80,  # High penalty for repetition issues
            'language_detection_threshold': 0.70,  # Must be reasonably detected as English
        }

        # Translation chunking parameters
        self.translation_chunk_params = {
            'max_tokens': 400,  # Maximum tokens per translation chunk
            'overlap_ratio': 0.1,  # Overlap ratio for context preservation
            'min_chunk_size': 50,  # Minimum characters for a chunk
            'quality_threshold': 0.7,  # Threshold for rechunking complex content
        }

        # Performance tracking for adaptive model selection
        self.model_performance = {}
        
        # Translation artifact patterns for cleaning (medical-focused)
        self.translation_artifact_patterns = [
            # Numerical patterns
            r'(\d+\.\d+\.\d+\.)+\d+',  # Repeated decimal patterns
            r'(\d+\s*mg\s*){3,}',      # Repeated dosage
            r'(\d+\s*%\s*){3,}',       # Repeated percentages
            
            # Punctuation and symbols
            r'([!?.,:;-])\1{3,}',      # Repeated punctuation
            r'(\$\s*){3,}',           # Repeated currency symbols
            
            # Medical phrase repetition
            r'(\w+\s+)\1{3,}',         # General repeated words
            r'((?:clinical trial|study|patient|treatment)\s+){3,}',
            r'((?:primary endpoint|secondary endpoint|adverse event)\s+){3,}',
            r'(p[<=]\d+\.\d+\s*){3,}', # Repeated p-values
            r'(CI:\s*\d+\.\d+-\d+\.\d+\s*){3,}',  # Repeated confidence intervals
            r'(HR:\s*\d+\.\d+\s*){3,}',  # Repeated hazard ratios
            r'(OR:\s*\d+\.\d+\s*){3,}',  # Repeated odds ratios
        ]

        # Medical terms that should be preserved even if they appear repetitive
        self.medical_exclusions = [
            r'dose-dose\s+(?:escalation|reduction)',
            r'first-line.*second-line.*third-line',
            r'pre-treatment.*post-treatment',
            r'primary.*secondary.*tertiary',
            r'grade\s+1.*grade\s+2.*grade\s+3',
            r'phase\s+I.*phase\s+II.*phase\s+III',
        ]

        # Robust CUDA setup
        self.use_cuda = False
        self.device = "cpu"

        if torch.cuda.is_available():
            try:
                if torch.cuda.is_initialized():
                    torch.cuda.empty_cache()
                    gc.collect()

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
                try:
                    torch.cuda.empty_cache()
                except:
                    pass
                gc.collect()
                self.device = "cpu"
        else:
            print("⚠️  CUDA not available, using CPU")

        # Current loaded translator info
        self.current_translator = None
        self.current_language = None
        self.current_tier = None
        self.current_model_name = None

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text. Uses a simple approximation.
        For more accuracy, could use specific tokenizer, but this is efficient.
        """
        # Rough estimation: ~4 characters per token for medical text
        return len(text) // 4

    def adaptive_chunk_for_translation(self, text: str, max_tokens: int = None) -> List[Dict[str, Any]]:
        """
        Intelligently chunk large text blocks for optimal translation.
        Respects semantic boundaries and medical context.
        
        Returns list of chunk dictionaries with text and metadata.
        """
        if max_tokens is None:
            max_tokens = self.translation_chunk_params['max_tokens']
            
        # If already small enough, return as-is
        if self.count_tokens(text) <= max_tokens:
            return [{'text': text, 'is_sub_chunk': False, 'chunk_index': 0}]
        
        print(f"      🔨 Adaptive chunking needed (estimated {self.count_tokens(text)} tokens > {max_tokens})")
        
        # Split on natural boundaries (ordered by preference)
        boundaries = [
            r'\n\n•\s+',              # Bullet points
            r'\n\n[A-Z][^a-z]*\n',    # Section headers
            r'\.\s+•\s+',             # Sentence + bullet
            r'\n\n',                  # Paragraph breaks
            r'\.\s+[A-Z]',            # Sentence boundaries
            r'[.!?]\s+',              # Any sentence end
            r'[;:]\s+',               # Semi-colon/colon breaks
        ]
        
        chunks = self._split_by_boundaries(text, boundaries, max_tokens)
        
        # Add metadata to chunks
        chunk_data = []
        for i, chunk_text in enumerate(chunks):
            chunk_data.append({
                'text': chunk_text,
                'is_sub_chunk': len(chunks) > 1,
                'chunk_index': i,
                'total_sub_chunks': len(chunks)
            })
        
        print(f"      ✓ Created {len(chunks)} translation sub-chunks")
        return chunk_data

    def _split_by_boundaries(self, text: str, boundaries: List[str], max_tokens: int) -> List[str]:
        """
        Split text by boundaries while respecting token limits and preserving medical context.
        """
        # Try each boundary pattern in order of preference
        for boundary_pattern in boundaries:
            chunks = re.split(boundary_pattern, text)
            
            # If we got useful splits, process them
            if len(chunks) > 1:
                final_chunks = []
                current_chunk = ""
                
                for chunk in chunks:
                    chunk = chunk.strip()
                    if not chunk:
                        continue
                    
                    # Check if adding this chunk would exceed token limit
                    potential_chunk = current_chunk + ("\n\n" if current_chunk else "") + chunk
                    
                    if self.count_tokens(potential_chunk) <= max_tokens:
                        current_chunk = potential_chunk
                    else:
                        # Save current chunk if it exists
                        if current_chunk:
                            final_chunks.append(current_chunk)
                        
                        # Start new chunk
                        if self.count_tokens(chunk) <= max_tokens:
                            current_chunk = chunk
                        else:
                            # Chunk is still too large, try next boundary level recursively
                            if boundary_pattern != boundaries[-1]:  # Not the last boundary
                                remaining_boundaries = boundaries[boundaries.index(boundary_pattern) + 1:]
                                sub_chunks = self._split_by_boundaries(chunk, remaining_boundaries, max_tokens)
                                final_chunks.extend(sub_chunks)
                                current_chunk = ""
                            else:
                                # Last resort: force split by characters
                                sub_chunks = self._force_split_by_chars(chunk, max_tokens)
                                final_chunks.extend(sub_chunks)
                                current_chunk = ""
                
                # Add remaining chunk
                if current_chunk:
                    final_chunks.append(current_chunk)
                
                # If we achieved reasonable chunking, return
                if all(self.count_tokens(chunk) <= max_tokens for chunk in final_chunks):
                    return final_chunks
        
        # Fallback: force split by character count
        return self._force_split_by_chars(text, max_tokens)

    def _force_split_by_chars(self, text: str, max_tokens: int) -> List[str]:
        """
        Force split text by character count as last resort.
        Tries to split at word boundaries when possible.
        """
        max_chars = max_tokens * 4  # Rough character estimate
        chunks = []
        
        while text:
            if len(text) <= max_chars:
                chunks.append(text)
                break
            
            # Find a good split point near the limit
            split_point = max_chars
            
            # Try to split at word boundary
            while split_point > max_chars * 0.8 and split_point < len(text):
                if text[split_point].isspace():
                    break
                split_point -= 1
            
            # If no good word boundary found, split at sentence
            if split_point <= max_chars * 0.8:
                split_point = max_chars
                while split_point > max_chars * 0.6 and split_point < len(text):
                    if text[split_point] in '.!?':
                        split_point += 1
                        break
                    split_point -= 1
            
            # Extract chunk
            chunk = text[:split_point].strip()
            if chunk:
                chunks.append(chunk)
            
            text = text[split_point:].strip()
        
        return chunks

    def preserve_medical_context(self, chunk: str) -> bool:
        """
        Ensure drug names and their contexts stay together.
        """
        drug_context_patterns = [
            r'(osimertinib|crizotinib|alectinib).*?(?=\n\n|\.|$)',
            r'(EGFR|ALK|ROS-1).*?mutation.*?(?=\n\n|\.|$)',
            r'(first-line|second-line).*?treatment.*?(?=\n\n|\.|$)'
        ]
        return any(re.search(pattern, chunk, re.IGNORECASE) for pattern in drug_context_patterns)

    def should_rechunk_for_translation(self, chunk: str, quality_threshold: float = None) -> bool:
        """
        Determine if a chunk is too complex for reliable translation.
        """
        if quality_threshold is None:
            quality_threshold = self.translation_chunk_params['quality_threshold']
            
        complexity_indicators = {
            'token_count': self.count_tokens(chunk),
            'ocr_artifacts': len(re.findall(r'\(cid:\d+\)', chunk)),
            'reference_density': len(re.findall(r'\[\d+(?:,\s*\d+)*\]', chunk)) / max(len(chunk.split()), 1),
            'context_switches': len(re.findall(r'(EGFR|ALK|ROS-1|BRAF)', chunk)),
            'formatting_changes': len(re.findall(r'\n\s*•|\n\n[A-Z]', chunk))
        }
        
        complexity_score = self._calculate_complexity_score(complexity_indicators)
        return complexity_score > quality_threshold

    def _calculate_complexity_score(self, indicators: Dict[str, float]) -> float:
        """
        Calculate complexity score based on various indicators.
        """
        weights = {
            'token_count': 0.3,
            'ocr_artifacts': 0.25,
            'reference_density': 0.2,
            'context_switches': 0.15,
            'formatting_changes': 0.1
        }
        
        # Normalize indicators
        normalized = {}
        normalized['token_count'] = min(1.0, indicators['token_count'] / 600)  # 600 tokens = 1.0
        normalized['ocr_artifacts'] = min(1.0, indicators['ocr_artifacts'] / 5)  # 5 artifacts = 1.0
        normalized['reference_density'] = min(1.0, indicators['reference_density'] * 10)  # 0.1 density = 1.0
        normalized['context_switches'] = min(1.0, indicators['context_switches'] / 3)  # 3 switches = 1.0
        normalized['formatting_changes'] = min(1.0, indicators['formatting_changes'] / 5)  # 5 changes = 1.0
        
        # Calculate weighted score
        score = sum(weights[key] * normalized[key] for key in weights.keys())
        return score

    def merge_translated_chunks(self, translated_sub_chunks: List[str], original_chunk_text: str) -> str:
        """
        Merge translated sub-chunks back into a coherent whole.
        Handles overlaps and ensures medical context coherence.
        """
        if len(translated_sub_chunks) == 1:
            return translated_sub_chunks[0]
        
        # Simple concatenation with intelligent spacing
        merged = []
        
        for i, sub_chunk in enumerate(translated_sub_chunks):
            sub_chunk = sub_chunk.strip()
            if not sub_chunk:
                continue
                
            # Add appropriate spacing between chunks
            if merged:
                # Check if we need paragraph break or just space
                last_chunk = merged[-1]
                if (last_chunk.endswith('.') or last_chunk.endswith('!') or last_chunk.endswith('?') or
                    sub_chunk[0].isupper()):
                    # Likely new sentence or paragraph
                    if '\n\n' in original_chunk_text:
                        merged.append('\n\n')
                    else:
                        merged.append(' ')
                else:
                    merged.append(' ')
            
            merged.append(sub_chunk)
        
        result = ''.join(merged)
        
        # Clean up any spacing issues
        result = re.sub(r'\n\n\n+', '\n\n', result)  # Max 2 newlines
        result = re.sub(r'  +', ' ', result)  # Max 1 space
        
        return result.strip()

    def check_model_availability(self, model_name: str) -> bool:
        """Check if a model is available and can be loaded."""
        try:
            # Try to load the tokenizer first (lightweight check)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            return True
        except Exception as e:
            print(f"    ✗ Model {model_name} not available: {str(e)[:50]}")
            return False

    def get_available_models_for_tier(self, tier: int, language: str) -> List[str]:
        """Get list of available models for a tier and language, checking availability."""
        available_models = []
        
        if tier == 1:
            # Check Helsinki model first
            helsinki_models = self.model_tiers[1]['helsinki_models']
            if language in helsinki_models:
                model_name = helsinki_models[language]
                if self.check_model_availability(model_name):
                    available_models.append(model_name)
            
            # Always add NLLB fallback if language is supported
            if language in self.nllb_lang_mapping:
                fallback_model = self.model_tiers[1]['fallback']
                if self.check_model_availability(fallback_model):
                    available_models.append(fallback_model)
                    
        elif tier == 2:
            if language in self.nllb_lang_mapping:
                # Try distilled large first
                if 'nllb_distilled' in self.model_tiers[2]:
                    model_name = self.model_tiers[2]['nllb_distilled']
                    if self.check_model_availability(model_name):
                        available_models.append(model_name)
                
                # Try regular large
                if 'nllb_large' in self.model_tiers[2]:
                    model_name = self.model_tiers[2]['nllb_large']
                    if self.check_model_availability(model_name):
                        available_models.append(model_name)
                
                # Fallback
                if 'fallback' in self.model_tiers[2]:
                    fallback_model = self.model_tiers[2]['fallback']
                    if self.check_model_availability(fallback_model):
                        available_models.append(fallback_model)
                        
        elif tier == 3:
            if language in self.nllb_lang_mapping:
                # Try XL NLLB first
                if 'nllb_xl' in self.model_tiers[3]:
                    model_name = self.model_tiers[3]['nllb_xl']
                    if self.check_model_availability(model_name):
                        available_models.append(model_name)
                
                # Try M2M-100
                if 'm2m100_large' in self.model_tiers[3]:
                    model_name = self.model_tiers[3]['m2m100_large']
                    if self.check_model_availability(model_name):
                        available_models.append(model_name)
                
                # Fallback
                if 'fallback' in self.model_tiers[3]:
                    fallback_model = self.model_tiers[3]['fallback']
                    if self.check_model_availability(fallback_model):
                        available_models.append(fallback_model)
        
        return available_models

    def assess_translation_quality(self, original_text: str, translated_text: str, language: str) -> Dict[str, float]:
        """
        Assess the quality of a translation using multiple metrics optimized for medical content.
        """
        if not translated_text.strip():
            return {'overall': 0.0, 'repetition': 0.0, 'coherence': 0.0, 'language': 0.0, 'medical': 0.0}
        
        # 1. Repetition detection (critical for medical accuracy)
        repetition_score = 1.0 - (1.0 if self.detect_repetition(translated_text) else 0.0)
        
        # 2. Length coherence (medical translations should be reasonably similar in length)
        len_ratio = len(translated_text) / max(len(original_text), 1)
        length_score = 1.0 if 0.4 <= len_ratio <= 2.5 else 0.6 if 0.2 <= len_ratio <= 4.0 else 0.3
        
        # 3. Language detection (should be English)
        try:
            detected_lang = detect(translated_text)
            language_score = 1.0 if detected_lang == 'en' else 0.2
        except:
            language_score = 0.5  # Neutral if detection fails
        
        # 4. Medical term preservation (critical for medical data)
        medical_score = self.check_medical_term_preservation(original_text, translated_text)
        
        # 5. Artifact detection (medical texts should be clean)
        artifact_count = self.count_artifacts(translated_text)
        word_count = len(translated_text.split())
        artifact_score = max(0.0, 1.0 - (artifact_count / max(word_count, 1)) * 5)
        
        # 6. Medical coherence (check for medical-specific issues)
        coherence_score = self.assess_medical_coherence(translated_text)
        
        # Overall weighted score (medical preservation weighted heavily)
        weights = {
            'repetition': 0.25,   # High weight - repetition ruins medical accuracy
            'length': 0.15,       # Moderate weight
            'language': 0.15,     # Moderate weight
            'medical': 0.30,      # Highest weight - critical for medical data
            'artifacts': 0.10,    # Low weight but still important
            'coherence': 0.05     # Basic sanity check
        }
        
        scores = {
            'repetition': repetition_score,
            'length': length_score,
            'language': language_score,
            'medical': medical_score,
            'artifacts': artifact_score,
            'coherence': coherence_score
        }
        
        overall_score = sum(weights[k] * scores[k] for k in weights.keys())
        scores['overall'] = overall_score
        
        return scores

    def assess_document_quality(self, translated_data: dict, original_data: dict, language: str) -> Dict[str, float]:
        """
        Assess the overall quality of a translated document.
        
        Returns comprehensive quality metrics for the entire document.
        """
        if 'chunks' not in translated_data or 'chunks' not in original_data:
            return {'overall': 0.0, 'chunk_count': 0}
        
        chunk_qualities = []
        total_original_text = ""
        total_translated_text = ""
        
        # Analyze each chunk and collect overall statistics
        for orig_chunk, trans_chunk in zip(original_data['chunks'], translated_data['chunks']):
            # Assess heading if present
            if 'heading' in orig_chunk and 'heading' in trans_chunk:
                if orig_chunk['heading'] and trans_chunk['heading']:
                    heading_quality = self.assess_translation_quality(
                        orig_chunk['heading'], trans_chunk['heading'], language
                    )
                    chunk_qualities.append(heading_quality)
                    total_original_text += " " + orig_chunk['heading']
                    total_translated_text += " " + trans_chunk['heading']
            
            # Assess text if present
            if 'text' in orig_chunk and 'text' in trans_chunk:
                if orig_chunk['text'] and trans_chunk['text']:
                    text_quality = self.assess_translation_quality(
                        orig_chunk['text'], trans_chunk['text'], language
                    )
                    chunk_qualities.append(text_quality)
                    total_original_text += " " + orig_chunk['text']
                    total_translated_text += " " + trans_chunk['text']
        
        if not chunk_qualities:
            return {'overall': 0.0, 'chunk_count': 0}
        
        # Calculate aggregate metrics
        metrics = ['overall', 'repetition', 'language', 'medical', 'artifacts', 'coherence']
        aggregate_scores = {}
        
        for metric in metrics:
            scores = [quality[metric] for quality in chunk_qualities if metric in quality]
            aggregate_scores[metric] = sum(scores) / len(scores) if scores else 0.0
        
        # Document-level assessment
        doc_level_quality = self.assess_translation_quality(
            total_original_text.strip(), total_translated_text.strip(), language
        )
        
        # Combine chunk-level and document-level assessments
        final_scores = {}
        for metric in metrics:
            chunk_score = aggregate_scores.get(metric, 0.0)
            doc_score = doc_level_quality.get(metric, 0.0)
            # Weight chunk-level assessment more heavily (70%) than document-level (30%)
            final_scores[metric] = (chunk_score * 0.7) + (doc_score * 0.3)
        
        final_scores['chunk_count'] = len(chunk_qualities)
        final_scores['avg_chunk_quality'] = aggregate_scores.get('overall', 0.0)
        final_scores['doc_level_quality'] = doc_level_quality.get('overall', 0.0)
        
        return final_scores

    def should_retranslate_with_higher_tier(self, quality_scores: Dict[str, float]) -> bool:
        """
        Determine if a document should be retranslated with a higher tier model.
        
        Uses a comprehensive quality assessment approach:
        1. Overall quality threshold
        2. Critical component thresholds (medical preservation, repetition, language detection)
        3. Weighted decision making
        """
        overall_quality = quality_scores.get('overall', 0.0)
        medical_score = quality_scores.get('medical', 0.0)
        repetition_score = quality_scores.get('repetition', 0.0)
        language_score = quality_scores.get('language', 0.0)
        
        # Primary threshold check
        if overall_quality < self.document_quality_thresholds['tier_1_to_2_threshold']:
            print(f"    📉 Overall quality {overall_quality:.3f} below threshold {self.document_quality_thresholds['tier_1_to_2_threshold']}")
            return True
        
        # Critical component checks
        if medical_score < self.document_quality_thresholds['medical_preservation_threshold']:
            print(f"    🏥 Medical preservation {medical_score:.3f} below threshold {self.document_quality_thresholds['medical_preservation_threshold']}")
            return True
            
        if repetition_score < self.document_quality_thresholds['repetition_penalty_threshold']:
            print(f"    🔄 Repetition issues detected {repetition_score:.3f} below threshold {self.document_quality_thresholds['repetition_penalty_threshold']}")
            return True
            
        if language_score < self.document_quality_thresholds['language_detection_threshold']:
            print(f"    🌐 Language detection {language_score:.3f} below threshold {self.document_quality_thresholds['language_detection_threshold']}")
            return True
        
        return False

    def compare_translation_quality(self, quality_1: Dict[str, float], quality_2: Dict[str, float]) -> int:
        """
        Compare two translation quality assessments and return the better one.
        
        Returns:
            1 if quality_1 is better
            2 if quality_2 is better
            1 if tie (default to first)
        """
        # Primary comparison: overall quality
        if abs(quality_1['overall'] - quality_2['overall']) > 0.05:  # 5% difference threshold
            return 1 if quality_1['overall'] > quality_2['overall'] else 2
        
        # Secondary comparison: medical preservation (critical for medical docs)
        med_diff = quality_1.get('medical', 0) - quality_2.get('medical', 0)
        if abs(med_diff) > 0.1:  # 10% difference threshold for medical
            return 1 if med_diff > 0 else 2
        
        # Tertiary comparison: repetition issues
        rep_diff = quality_1.get('repetition', 0) - quality_2.get('repetition', 0)
        if abs(rep_diff) > 0.1:
            return 1 if rep_diff > 0 else 2
        
        # Quaternary: language detection
        lang_diff = quality_1.get('language', 0) - quality_2.get('language', 0)
        if abs(lang_diff) > 0.1:
            return 1 if lang_diff > 0 else 2
        
        # Default to first translation if very close
        return 1

    def assess_medical_coherence(self, text: str) -> float:
        """Assess if the translated text maintains medical coherence."""
        nonsense_patterns = [
            r'patient patient patient',
            r'treatment treatment treatment',
            r'study study study',
            r'(\w+)\s+\1\s+\1\s+\1',  # Any word repeated 4+ times
        ]
        
        penalties = 0
        for pattern in nonsense_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                penalties += 1
        
        return max(0.0, 1.0 - (penalties * 0.3))

    def check_medical_term_preservation(self, original: str, translated: str) -> float:
        """Check if medical terms were properly preserved."""
        original_lower = original.lower()
        translated_lower = translated.lower()
        
        preserved_count = 0
        total_terms = 0
        
        for term in self.preserve_terms:
            if term.lower() in original_lower:
                total_terms += 1
                if term.lower() in translated_lower:
                    preserved_count += 1
        
        return preserved_count / max(total_terms, 1)

    def count_artifacts(self, text: str) -> int:
        """Count translation artifacts in text."""
        # Check medical exclusions first
        for exclusion_pattern in self.medical_exclusions:
            if re.search(exclusion_pattern, text, re.IGNORECASE):
                return 0
        
        artifact_count = 0
        for pattern in self.translation_artifact_patterns:
            matches = re.findall(pattern, text)
            artifact_count += len(matches)
        
        return artifact_count

    def load_helsinki_model(self, model_name: str, language: str) -> Optional[Any]:
        """Load Helsinki model with error handling."""
        try:
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
                gc.collect()

            translator = pipeline(
                "translation",
                model=model_name,
                device=self.device,
                torch_dtype=torch.float32,
                trust_remote_code=False,
            )

            # Test the translator
            test_result = translator("Hello world", max_length=50)
            
            print(f"    ✓ Helsinki model loaded successfully on {self.device}")
            return translator

        except Exception as e:
            if self.device.startswith("cuda"):
                print(f"    ⚠️  CUDA failed, trying CPU: {str(e)[:50]}")
                try:
                    translator = pipeline(
                        "translation",
                        model=model_name,
                        device="cpu",
                        torch_dtype=torch.float32,
                        trust_remote_code=False,
                    )
                    test_result = translator("Hello world", max_length=50)
                    self.device = "cpu"
                    print(f"    ✓ Helsinki model loaded successfully on CPU")
                    return translator
                except Exception as cpu_error:
                    print(f"    ✗ Helsinki model failed on CPU: {str(cpu_error)[:50]}")
                    return None
            else:
                raise e

    def load_nllb_model(self, model_name: str, language: str) -> Optional[Any]:
        """Load NLLB model with error handling and medical-optimized parameters."""
        try:
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
                gc.collect()

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                trust_remote_code=False,
            )

            if self.device.startswith("cuda"):
                model = model.to(self.device)

            def nllb_translate(text, generation_params=None, **kwargs):
                try:
                    src_lang = self.nllb_lang_mapping[language]
                    tgt_lang = 'eng_Latn'

                    max_input_length = min(512, generation_params.get('max_length', 400) + 50) if generation_params else 256
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_input_length)
                    if self.device.startswith("cuda"):
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    # Medical-optimized generation parameters
                    gen_kwargs = {
                        'forced_bos_token_id': tokenizer.convert_tokens_to_ids(tgt_lang),
                        'max_length': generation_params.get('max_length', 256) if generation_params else 256,
                        'num_beams': generation_params.get('num_beams', 3) if generation_params else 3,
                        'length_penalty': generation_params.get('length_penalty', 1.0) if generation_params else 1.0,
                        'do_sample': generation_params.get('do_sample', False) if generation_params else False,
                        'no_repeat_ngram_size': generation_params.get('no_repeat_ngram_size', 3) if generation_params else 3,
                        'repetition_penalty': generation_params.get('repetition_penalty', 1.2) if generation_params else 1.2,
                    }
                    
                    # Filter out unsupported parameters
                    if generation_params and 'temperature' in generation_params:
                        del generation_params['temperature']
                    if 'early_stopping' in gen_kwargs:
                        del gen_kwargs['early_stopping']

                    with torch.no_grad():
                        translated_tokens = model.generate(**inputs, **gen_kwargs)

                    translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                    return [{'translation_text': translation}]

                except Exception as e:
                    print(f"      NLLB translation error: {str(e)[:50]}")
                    return [{'translation_text': text}]

            # Test the translator
            test_result = nllb_translate("Hello world")
            print(f"    ✓ NLLB model loaded successfully on {self.device}")
            return nllb_translate

        except Exception as e:
            print(f"    ✗ NLLB model failed: {str(e)[:50]}")
            return None

    def load_translator_for_tier(self, language: str, tier: int) -> Optional[Any]:
        """Load the appropriate translator for the given language and tier with robust fallback."""
        print(f"    Loading translator for language: {language} (Tier {tier})")
        
        # Get available models for this tier and language
        available_models = self.get_available_models_for_tier(tier, language)
        
        if not available_models:
            print(f"    No available models for language {language} at tier {tier}")
            return None
        
        # Try each available model in order
        for model_name in available_models:
            print(f"    Trying model: {model_name}")
            
            try:
                # Determine model type and load accordingly
                if 'helsinki' in model_name.lower():
                    translator = self.load_helsinki_model(model_name, language)
                elif 'nllb' in model_name.lower():
                    translator = self.load_nllb_model(model_name, language)
                else:
                    print(f"    Unknown model type: {model_name}")
                    continue
                
                if translator:
                    self.current_model_name = model_name
                    return translator
                    
            except Exception as e:
                print(f"    Failed to load {model_name}: {str(e)[:100]}")
                continue
        
        print(f"    ✗ No translator could be loaded for language: {language} at tier {tier}")
        return None

    def load_translator_for_language(self, language: str, target_tier: int = 1):
        """Load the appropriate translator for the given language, starting at the target tier."""
        if self.current_language == language and self.current_tier == target_tier and self.current_translator:
            return self.current_translator

        # Clear previous translator
        self.clear_translator()

        print(f"  🔄 Loading translator for language: {language} (Tier {target_tier})")

        # Try to load the model for the target tier
        translator = self.load_translator_for_tier(language, target_tier)
        
        if translator:
            self.current_translator = translator
            self.current_language = language
            self.current_tier = target_tier
            print(f"    ✓ Successfully loaded translator for {language}")
            return translator
        else:
            # Try fallback to other tiers if target tier failed
            for fallback_tier in [1, 2, 3]:
                if fallback_tier == target_tier:
                    continue
                print(f"    Trying fallback to Tier {fallback_tier}")
                translator = self.load_translator_for_tier(language, fallback_tier)
                if translator:
                    self.current_translator = translator
                    self.current_language = language
                    self.current_tier = fallback_tier
                    print(f"    ✓ Successfully loaded fallback translator (Tier {fallback_tier})")
                    return translator
            
            print(f"    ✗ No translator available for language: {language}")
            return None

    def preserve_medical_terms(self, text: str) -> tuple[str, dict]:
        """Replace medical terms with placeholders before translation."""
        preserved = {}
        modified_text = text
        
        for i, term in enumerate(self.preserve_terms):
            if term.lower() in text.lower():
                placeholder = f"__MEDICAL_TERM_{i}__"
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
            r'\|.*\|.*\|',
            r'^\s*\d+\.\d+\s+\d+\.\d+',
            text.count('|') > 5,
            text.count('Row') > 3
        ]
        return any(re.search(pattern, text) if isinstance(pattern, str) else pattern 
                   for pattern in table_indicators)

    def clean_translation_artifacts(self, text: str) -> str:
        """Clean common translation artifacts while preserving medical terminology."""
        for exclusion_pattern in self.medical_exclusions:
            if re.search(exclusion_pattern, text, re.IGNORECASE):
                return text
        
        cleaned_text = text
        for pattern in self.translation_artifact_patterns:
            cleaned_text = re.sub(pattern, r'\1', cleaned_text)
        
        return cleaned_text

    def detect_document_language(self, text: str) -> Optional[str]:
        """Detect the primary language of a document."""
        if not text or len(text.strip()) < 20:
            return None

        try:
            clean_text = ' '.join(text.split()[:200])
            detected_lang = detect(clean_text)
            print(f"    Detected language: {detected_lang}")
            return detected_lang
        except Exception:
            print("    Language detection failed")
            return None

    def is_english_chunk(self, text: str) -> bool:
        """Quick check if a chunk is in English."""
        if not text or len(text.strip()) < 10:
            return True

        text_lower = text.lower()
        english_words = [
            ' the ', ' and ', ' of ', ' to ', ' a ', ' in ', ' is ', ' it ', ' you ', ' that ',
            ' he ', ' was ', ' for ', ' on ', ' are ', ' as ', ' with ', ' his ', ' they ',
            ' i ', ' at ', ' be ', ' this ', ' have ', ' from ', ' or ', ' one ', ' had ',
            ' by ', ' word ', ' but ', ' not ', ' what ', ' all ', ' were ', ' we '
        ]

        english_count = sum(1 for word in english_words if word in f' {text_lower} ')
        total_words = len(text.split())

        if total_words > 5:
            english_ratio = english_count / total_words
            return english_ratio > 0.1

        return False

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for sequential translation."""
        medical_abbrevs = r'(?:Dr|Mr|Mrs|Ms|Prof|vs|etc|i\.e|e\.g|cf|approx|max|min|Fig|Tab|Ref|Vol|No|pg|pp|PFS|OS|ORR|DCR|CI|HR|OR|RR|AE|SAE|ECOG|BCLC|HCC|NSCLC|KRAS|G12C)'
        
        protected_text = re.sub(f'({medical_abbrevs})\\.', r'\1__PERIOD__', text, flags=re.IGNORECASE)
        sentences = re.split(r'[.!?]+\s+', protected_text)
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

    def get_generation_params(self, tier: int = 1) -> dict:
        """Get generation parameters optimized for each tier."""
        base_params = {
            'max_length': 400,
            'truncation': True,
            'no_repeat_ngram_size': 3,
            'repetition_penalty': 1.2,
            'do_sample': False,
            'num_beams': 3,
        }
        
        # Tier-specific optimizations
        if tier >= 2:
            base_params.update({
                'num_beams': 4,
                'length_penalty': 1.1,
                'repetition_penalty': 1.25,
                'no_repeat_ngram_size': 4,
            })
        
        return base_params

    def translate_single_chunk(self, text: str, translator, tier: int = 1) -> str:
        """
        Translate a single chunk of text with tier-appropriate parameters and adaptive chunking.
        This method now incorporates the translation-specific chunking strategy.
        """
        if not text.strip() or not translator:
            return text

        try:
            # First, check if the chunk needs adaptive chunking for translation
            if self.should_rechunk_for_translation(text) or self.count_tokens(text) > self.translation_chunk_params['max_tokens']:
                print(f"      🔨 Applying adaptive translation chunking")
                
                # Create translation-specific sub-chunks
                translation_chunks = self.adaptive_chunk_for_translation(text, self.translation_chunk_params['max_tokens'])
                
                # Translate each sub-chunk
                translated_sub_chunks = []
                for chunk_data in translation_chunks:
                    sub_chunk_text = chunk_data['text']
                    
                    # Preserve medical terms for this sub-chunk
                    protected_text, preserved_terms = self.preserve_medical_terms(sub_chunk_text)
                    
                    # Get generation parameters for this tier
                    gen_params = self.get_generation_params(tier)
                    
                    # Translate based on translator type
                    if hasattr(translator, 'model'):
                        # Helsinki pipeline
                        pipeline_params = {k: v for k, v in gen_params.items() 
                                         if k in ['max_length', 'truncation', 'no_repeat_ngram_size', 
                                                 'repetition_penalty', 'do_sample', 'num_beams']}
                        result = translator(protected_text, **pipeline_params)
                    else:
                        # NLLB custom function
                        result = translator(protected_text, generation_params=gen_params)
                    
                    translated_sub_text = result[0]['translation_text']
                    
                    # Restore medical terms for this sub-chunk
                    final_sub_text = self.restore_medical_terms(translated_sub_text, preserved_terms)
                    
                    translated_sub_chunks.append(final_sub_text)
                
                # Merge the translated sub-chunks back together
                final_text = self.merge_translated_chunks(translated_sub_chunks, text)
                
                print(f"      ✓ Merged {len(translated_sub_chunks)} translation sub-chunks")
                
            else:
                # Standard single-chunk translation (no adaptive chunking needed)
                # Preserve medical terms
                protected_text, preserved_terms = self.preserve_medical_terms(text)
                
                # Get generation parameters for this tier
                gen_params = self.get_generation_params(tier)
                
                # Translate based on translator type
                if hasattr(translator, 'model'):
                    # Helsinki pipeline
                    pipeline_params = {k: v for k, v in gen_params.items() 
                                     if k in ['max_length', 'truncation', 'no_repeat_ngram_size', 
                                             'repetition_penalty', 'do_sample', 'num_beams']}
                    result = translator(protected_text, **pipeline_params)
                else:
                    # NLLB custom function
                    result = translator(protected_text, generation_params=gen_params)
                
                translated_text = result[0]['translation_text']
                
                # Restore medical terms
                final_text = self.restore_medical_terms(translated_text, preserved_terms)
            
            # Clean translation artifacts
            cleaned_text = self.clean_translation_artifacts(final_text)
            
            return cleaned_text

        except Exception as e:
            print(f"      Translation error: {str(e)[:50]}")
            return text

    def translate_document_with_tier(self, data: dict, language: str, tier: int) -> Tuple[dict, Dict[str, float], Dict[str, Any]]:
        """
        Translate an entire document with a specific tier model.
        
        Returns:
            Tuple of (translated_data, quality_scores, tier_metadata)
        """
        print(f"  📝 Translating document with Tier {tier}")
        
        tier_start_time = datetime.now()
        
        # Load translator for this tier
        translator = self.load_translator_for_language(language, tier)
        if not translator:
            print(f"    ✗ No translator available for Tier {tier}")
            return data, {'overall': 0.0}, {
                'tier': tier,
                'model_loaded': False,
                'processing_time_seconds': 0,
                'model_name': None
            }
        
        # Make a deep copy to avoid modifying original data
        translated_data = copy.deepcopy(data)
        
        if 'chunks' not in translated_data:
            print(f"    No chunks found in document")
            return translated_data, {'overall': 0.0}, {
                'tier': tier,
                'model_loaded': True,
                'processing_time_seconds': (datetime.now() - tier_start_time).total_seconds(),
                'model_name': self.current_model_name,
                'chunks_found': False
            }
        
        total_chunks = len(translated_data['chunks'])
        translated_count = 0
        english_count = 0
        
        print(f"    Processing {total_chunks} chunks with Tier {tier}...")
        
        for i, chunk in enumerate(translated_data['chunks']):
            if i % 20 == 0 or i == total_chunks - 1:
                print(f"      Chunk {i+1}/{total_chunks}")
            
            # Process heading
            if 'heading' in chunk and chunk['heading']:
                if self.is_english_chunk(chunk['heading']):
                    english_count += 1
                else:
                    chunk['heading'] = self.translate_single_chunk(
                        chunk['heading'], translator, tier
                    )
                    translated_count += 1
            
            # Process text
            if 'text' in chunk and chunk['text']:
                if self.is_english_chunk(chunk['text']):
                    english_count += 1
                else:
                    chunk['text'] = self.translate_single_chunk(
                        chunk['text'], translator, tier
                    )
                    translated_count += 1
        
        processing_time = (datetime.now() - tier_start_time).total_seconds()
        
        print(f"    ✓ Tier {tier} translation complete: {english_count} English chunks, {translated_count} translated chunks")
        
        # Assess document quality
        quality_scores = self.assess_document_quality(translated_data, data, language)
        
        print(f"    📊 Tier {tier} Quality Assessment:")
        print(f"      Overall: {quality_scores['overall']:.3f}")
        print(f"      Medical: {quality_scores.get('medical', 0):.3f}")
        print(f"      Repetition: {quality_scores.get('repetition', 0):.3f}")
        print(f"      Language: {quality_scores.get('language', 0):.3f}")
        
        # Capture tier metadata
        tier_metadata = {
            'tier': tier,
            'model_loaded': True,
            'model_name': self.current_model_name,
            'processing_time_seconds': processing_time,
            'chunks_found': True,
            'total_chunks': total_chunks,
            'chunks_translated': translated_count,
            'chunks_english': english_count,
            'quality_scores': quality_scores
        }
        
        return translated_data, quality_scores, tier_metadata

    def clear_translator(self):
        """Clear current translator and free memory."""
        self.current_translator = None
        self.current_language = None
        self.current_tier = None
        self.current_model_name = None

        # Safe memory cleanup
        gc.collect()
        if self.use_cuda and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except:
                pass

    def process_json_file(self, input_path: str, output_path: str):
        """
        Process a single JSON file with document-level hierarchical translation.
        Enhanced with comprehensive metadata tracking.
        """
        file_name = os.path.basename(input_path)
        print(f"\n📄 Processing: {file_name}")
        
        # Start tracking processing time
        self.processing_start_time = datetime.now()

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

        # Initialize translation metadata
        translation_metadata = {
            "processing_timestamp": self.processing_start_time.isoformat(),
            "source_file": file_name,
            "detected_language": document_language,
            "was_translation_needed": False,
            "tier_attempts": [],
            "final_tier_used": None,
            "final_model_used": None,
            "retranslation_occurred": False,
            "quality_comparison": {},
            "final_quality_scores": {},
            "chunks_translated": 0,
            "chunks_preserved_english": 0,
            "total_processing_time_seconds": 0,
            "translation_strategy": "hierarchical_document_level"
        }

        if not document_language or document_language == 'en':
            print(f"  📋 Document is English, copying without translation")
            translation_metadata.update({
                "translation_decision": "no_translation_needed_english",
                "total_processing_time_seconds": (datetime.now() - self.processing_start_time).total_seconds()
            })
            
            # Add metadata even for English documents
            data["_translation_metadata"] = translation_metadata
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return

        # Check if we have translation capabilities for this language
        tier_1_available = len(self.get_available_models_for_tier(1, document_language)) > 0
        tier_2_available = len(self.get_available_models_for_tier(2, document_language)) > 0
        
        translation_metadata.update({
            "was_translation_needed": True,
            "tier_1_available": tier_1_available,
            "tier_2_available": tier_2_available
        })
        
        if not tier_1_available and not tier_2_available:
            print(f"  📋 No translation models available for language {document_language}, copying original file")
            translation_metadata.update({
                "translation_decision": "no_models_available",
                "total_processing_time_seconds": (datetime.now() - self.processing_start_time).total_seconds()
            })
            
            # Add metadata even when no translation is possible
            data["_translation_metadata"] = translation_metadata
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return

        # Translate with Tier 1 (if available)
        best_translation = None
        best_quality = {'overall': 0.0}
        tier_1_metadata = None
        tier_2_metadata = None
        
        if tier_1_available:
            print(f"  🎯 Starting with Tier 1 translation")
            tier_1_translation, tier_1_quality, tier_1_metadata = self.translate_document_with_tier(
                data, document_language, 1
            )
            translation_metadata["tier_attempts"].append(tier_1_metadata)
            best_translation = tier_1_translation
            best_quality = tier_1_quality
            
            # Check if Tier 1 quality is sufficient
            if not self.should_retranslate_with_higher_tier(tier_1_quality):
                print(f"  ✅ Tier 1 quality sufficient (overall: {tier_1_quality['overall']:.3f})")
                final_translation = tier_1_translation
                final_quality = tier_1_quality
                translation_metadata.update({
                    "translation_decision": "tier_1_sufficient",
                    "final_tier_used": 1,
                    "final_model_used": tier_1_metadata["model_name"],
                    "retranslation_occurred": False
                })
            else:
                print(f"  📈 Tier 1 quality insufficient, trying Tier 2")
                
                # Try Tier 2 if available
                if tier_2_available:
                    self.clear_translator()  # Free Tier 1 memory
                    tier_2_translation, tier_2_quality, tier_2_metadata = self.translate_document_with_tier(
                        data, document_language, 2
                    )
                    translation_metadata["tier_attempts"].append(tier_2_metadata)
                    
                    # Compare translations and choose the best
                    if self.compare_translation_quality(tier_2_quality, tier_1_quality) == 1:
                        print(f"  🏆 Tier 2 translation is better (overall: {tier_2_quality['overall']:.3f} vs {tier_1_quality['overall']:.3f})")
                        final_translation = tier_2_translation
                        final_quality = tier_2_quality
                        translation_metadata.update({
                            "translation_decision": "tier_2_better_than_tier_1",
                            "final_tier_used": 2,
                            "final_model_used": tier_2_metadata["model_name"],
                            "retranslation_occurred": True,
                            "quality_comparison": {
                                "tier_1_overall": tier_1_quality['overall'],
                                "tier_2_overall": tier_2_quality['overall'],
                                "improvement": tier_2_quality['overall'] - tier_1_quality['overall']
                            }
                        })
                    else:
                        print(f"  🥈 Tier 1 translation is still better, keeping it")
                        final_translation = tier_1_translation
                        final_quality = tier_1_quality
                        translation_metadata.update({
                            "translation_decision": "tier_1_better_than_tier_2",
                            "final_tier_used": 1,
                            "final_model_used": tier_1_metadata["model_name"],
                            "retranslation_occurred": True,
                            "quality_comparison": {
                                "tier_1_overall": tier_1_quality['overall'],
                                "tier_2_overall": tier_2_quality['overall'],
                                "tier_1_chosen_despite_tier_2": True
                            }
                        })
                else:
                    print(f"  ⚠️  Tier 2 not available, keeping Tier 1 result")
                    final_translation = tier_1_translation
                    final_quality = tier_1_quality
                    translation_metadata.update({
                        "translation_decision": "tier_1_only_tier_2_unavailable",
                        "final_tier_used": 1,
                        "final_model_used": tier_1_metadata["model_name"],
                        "retranslation_occurred": False
                    })
        else:
            # No Tier 1 available, go directly to Tier 2
            print(f"  🎯 Tier 1 not available, starting with Tier 2")
            final_translation, final_quality, tier_2_metadata = self.translate_document_with_tier(
                data, document_language, 2
            )
            translation_metadata["tier_attempts"].append(tier_2_metadata)
            translation_metadata.update({
                "translation_decision": "tier_2_only_tier_1_unavailable",
                "final_tier_used": 2,
                "final_model_used": tier_2_metadata["model_name"],
                "retranslation_occurred": False
            })

        # Count final statistics
        translated_count = 0
        english_count = 0
        
        if 'chunks' in final_translation:
            for chunk in final_translation['chunks']:
                if 'heading' in chunk and chunk['heading']:
                    if self.is_english_chunk(chunk['heading']):
                        english_count += 1
                    else:
                        translated_count += 1
                
                if 'text' in chunk and chunk['text']:
                    if self.is_english_chunk(chunk['text']):
                        english_count += 1
                    else:
                        translated_count += 1

        # Complete metadata
        total_processing_time = (datetime.now() - self.processing_start_time).total_seconds()
        translation_metadata.update({
            "final_quality_scores": final_quality,
            "chunks_translated": translated_count,
            "chunks_preserved_english": english_count,
            "total_processing_time_seconds": total_processing_time,
            "processing_completed_timestamp": datetime.now().isoformat()
        })

        # Add metadata to the final translation
        final_translation["_translation_metadata"] = translation_metadata

        # Save the translation with metadata
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_translation, f, indent=2, ensure_ascii=False)

        print(f"  ✓ Final result: {english_count} English chunks, {translated_count} translated chunks")
        print(f"  📊 Final quality: {final_quality['overall']:.3f}")
        print(f"  ⏱️  Processing time: {total_processing_time:.2f}s")

        # Update global stats
        self.english_chunks_preserved += english_count
        self.chunks_translated += translated_count

        # Clear translator after each file to free memory
        self.clear_translator()

    def translate_documents(self):
        """
        Main method to translate all documents with document-level hierarchical selection 
        and adaptive translation chunking.
        """
        print("🚀 Starting document translation with enhanced adaptive chunking...")
        print("📋 Strategy:")
        print("   • Try Tier 1 first (if available) for entire document")
        print("   • Apply adaptive chunking during translation for complex content")
        print("   • Assess document-level quality")
        print("   • If quality insufficient, try Tier 2 and compare")
        print("   • Keep the best translation")
        print("   • Final document maintains original PDF chunk structure")

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
        print(f"   • Adaptive chunking applied when needed for translation quality")
        print(f"   • Original PDF document structure maintained in final output")