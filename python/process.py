import os
import re
import json
import statistics
import pdfplumber
from collections import defaultdict
import numpy as np
from typing import Dict, Any, Optional, List


class TableDetector:
    """
    Enhanced table detection that works across languages and document types
    without hardcoded language-specific patterns
    """
    
    def __init__(self, pdf_processor):
        """Initialize the detector with a PDFProcessor instance."""
        self.pdf_processor = pdf_processor
        
        # Universal patterns that indicate structured tabular data
        self.strong_table_patterns = [
            # Multiple aligned numeric values
            r'^\s*\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+',
            r'^\s*\d+,\d+\s+\d+,\d+\s+\d+,\d+',  # European decimal format
            
            # Multiple percentages or statistical measures
            r'^\s*\d+%\s+\d+%\s+\d+%',
            r'^\s*[<>=≤≥]\s*\d+[\.,]\d+\s+[<>=≤≥]\s*\d+[\.,]\d+',
            
            # Multiple fractions or ratios
            r'^\s*\d+/\d+\s+\d+/\d+\s+\d+/\d+',
            r'^\s*\d+:\d+\s+\d+:\d+\s+\d+:\d+',
            
            # Clear tabular separators
            r'\|\s*[^|]+\s*\|\s*[^|]+\s*\|',  # Pipe separators
            r'^\s*[^\t]+\t[^\t]+\t[^\t]+',     # Tab separators
            
            # Generated table content (from PDF extraction)
            r'Row\s+\d+.*:.*Row\s+\d+.*:',
            r'Column\s+\d+.*:.*Column\s+\d+.*:',
        ]
        
        # Pharmaceutical domain patterns
        self.pharmaceutical_patterns = [
            # Currency patterns (Euro symbols)
            r'€\s*\d+[\.,]\d+',
            r'\d+[\.,]\d+\s*€',
            
            # Medical dosage patterns
            r'\d+\s*mg(?:/m²)?(?:\s|$)',
            r'\d+\s*μg(?:\s|$)',
            r'\d+\s*ng(?:\s|$)',
            r'\d+\s*pg(?:\s|$)',
            r'\d+\s*IU(?:\s|$)',
            
            # Pharmaceutical abbreviations
            r'\b(?:FCT|CIS|TAB|AMP|SC|PIS)\b',
            r'\b(?:mg|ml|kg|mmol|μg|ng|pg|IU)\b',
            
            # Treatment cycle patterns
            r'\d+\s*x\s*(?:daily|per\s+day)',
            r'per\s+\d+-day\s+cycle',
            r'every\s+\d+\s+days',
            r'continuously',
            
            # Treatment designations
            r'designation\s+of\s+(?:the\s+)?therapy',
            r'treatment\s+mode',
            r'treatment\s+costs',
            r'appropriate\s+comparator\s+therapy',
        ]
        
        # Table title patterns
        self.table_title_patterns = [
            r'^\s*Table\s+\d+',
            r'^\s*Designation\s+of\s+the\s+therapy',
            r'^\s*Treatment\s+(?:costs|mode|schedule)',
            r'^\s*Medicinal\s+product\s+to\s+be\s+assessed',
            r'^\s*Appropriate\s+comparator\s+therapy',
            r'^\s*Consumption:?',
            r'^\s*Costs:?',
        ]
        
        # Patterns that strongly suggest prose (universal across languages)
        self.prose_patterns = [
            # Sentence-like structures with conjunctions/connectors
            r'\b\w{2,}\s+(?:and|or|but|however|therefore|moreover|furthermore|nevertheless|additionally)\s+\w{2,}',
            r'\b\w{2,}\s+(?:et|ou|mais|cependant|donc|de plus|néanmoins|également)\s+\w{2,}',  # French
            r'\b\w{2,}\s+(?:und|oder|aber|jedoch|daher|außerdem|dennoch|zusätzlich)\s+\w{2,}',  # German
            r'\b\w{2,}\s+(?:y|o|pero|sin embargo|por lo tanto|además|no obstante|también)\s+\w{2,}',  # Spanish
            r'\b\w{2,}\s+(?:e|o|ma|tuttavia|pertanto|inoltre|tuttavia|anche)\s+\w{2,}',  # Italian
            r'\b\w{2,}\s+(?:i|lub|ale|jednak|dlatego|ponadto|niemniej|również)\s+\w{2,}',  # Polish
            
            # Long sentences with punctuation
            r'[.!?]\s+[A-ZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞŸ][a-zàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ]{2,}',
            
            # Phrases with articles and prepositions (common in prose)
            r'\b(?:the|a|an|in|on|at|by|for|with|from|to|of|as)\s+\w{2,}\s+\w{2,}',  # English
            r'\b(?:le|la|les|un|une|des|dans|sur|avec|pour|par|de|du|des)\s+\w{2,}\s+\w{2,}',  # French
            r'\b(?:der|die|das|ein|eine|in|auf|mit|für|durch|von|zu|bei)\s+\w{2,}\s+\w{2,}',  # German
            r'\b(?:el|la|los|las|un|una|en|con|por|para|de|del|desde)\s+\w{2,}\s+\w{2,}',  # Spanish
            r'\b(?:il|la|lo|gli|le|un|una|in|con|per|da|di|del|sulla)\s+\w{2,}\s+\w{2,}',  # Italian
        ]


    def enhanced_table_validation(self, table_data, page, page_num, sensitivity_level=1) -> bool:
        """
        Enhanced table validation with graduated sensitivity levels.
        Level 1: Standard validation (threshold 0.6)
        Level 2: Relaxed validation (threshold 0.5) 
        Level 3: Medical-specific validation (threshold 0.4 with domain patterns)
        """
        if not table_data or len(table_data) < 3:
            return False
        
        cleaned_table = self.pdf_processor.clean_table_data(table_data)
        if len(cleaned_table) < 3:
            return False
        
        # Core validation checks
        structure_score = self._check_table_structure(cleaned_table)
        content_score = self._check_table_content(cleaned_table)
        visual_score = 1.0 if self.has_explicit_table_structure(page) else 0.0
        
        # Apply sensitivity-based adjustments
        if sensitivity_level == 3:
            # Medical domain boost for level 3
            medical_score = self._check_medical_domain_patterns(cleaned_table)
            overall_score = (structure_score * 0.3) + (content_score * 0.3) + (visual_score * 0.1) + (medical_score * 0.3)
            threshold = 0.4
        elif sensitivity_level == 2:
            # Relaxed threshold for level 2
            overall_score = (structure_score * 0.4) + (content_score * 0.4) + (visual_score * 0.2)
            threshold = 0.5
        else:
            # Standard validation for level 1
            overall_score = (structure_score * 0.4) + (content_score * 0.4) + (visual_score * 0.2)
            threshold = 0.6
        
        return overall_score > threshold

    def _check_medical_domain_patterns(self, cleaned_table) -> float:
        """Check for medical domain-specific table patterns focusing on dosages and pricing."""
        if not cleaned_table:
            return 0.0
        
        score = 0.0
        all_cells = []
        
        # Collect all cell content
        for row in cleaned_table:
            for cell in row:
                if cell and cell.strip():
                    all_cells.append(cell.strip().lower())
        
        if not all_cells:
            return 0.0
        
        cell_text = ' '.join(all_cells)
        
        # Dosage patterns (high priority)
        dosage_patterns = [
            r'\d+\s*mg(?:/m²)?(?:\s|$)',
            r'\d+\s*μg(?:\s|$)',
            r'\d+\s*ml(?:\s|$)',
            r'\d+\s*(?:mg|μg|ml|g)\s*(?:daily|per\s+day|twice\s+daily)',
            r'cycle\s+\d+',
            r'\d+\s*x\s*daily',
            r'every\s+\d+\s+(?:days|weeks)',
        ]
        
        # Pricing patterns (high priority)
        pricing_patterns = [
            r'€\s*\d+(?:[.,]\d+)?',
            r'\d+(?:[.,]\d+)?\s*€',
            r'\$\s*\d+(?:[.,]\d+)?',
            r'cost(?:s)?',
            r'price(?:s)?',
            r'treatment\s+cost',
        ]
        
        # Medical table indicators (medium priority)
        medical_indicators = [
            r'designation\s+of\s+therapy',
            r'medicinal\s+product',
            r'appropriate\s+comparator',
            r'consumption',
            r'treatment\s+(?:mode|schedule)',
            r'therapeutic\s+indication',
        ]
        
        # Count pattern matches
        dosage_matches = sum(1 for pattern in dosage_patterns if re.search(pattern, cell_text, re.IGNORECASE))
        pricing_matches = sum(1 for pattern in pricing_patterns if re.search(pattern, cell_text, re.IGNORECASE))
        medical_matches = sum(1 for pattern in medical_indicators if re.search(pattern, cell_text, re.IGNORECASE))
        
        # Weight scoring based on priority
        if dosage_matches > 0:
            score += 0.4
        if pricing_matches > 0:
            score += 0.4
        if medical_matches > 0:
            score += 0.2
        
        # Bonus for combination of patterns
        if dosage_matches > 0 and pricing_matches > 0:
            score += 0.2
        
        return min(score, 1.0)

    def _check_table_structure(self, cleaned_table) -> float:
        """Check basic structural characteristics of potential table."""
        if not cleaned_table:
            return 0.0
        
        # 1. Reasonable dimensions
        rows = len(cleaned_table)
        cols = len(cleaned_table[0]) if cleaned_table[0] else 0
        
        if not (3 <= rows <= 50 and 2 <= cols <= 10):
            return 0.0
        
        # 2. Column consistency (most rows should have similar column count)
        col_counts = [len(row) for row in cleaned_table]
        most_common_cols = max(set(col_counts), key=col_counts.count)
        consistency = col_counts.count(most_common_cols) / len(col_counts)
        
        if consistency < 0.7:
            return 0.0
        
        # 3. Content density (not too sparse, not too dense)
        total_cells = sum(len(row) for row in cleaned_table)
        filled_cells = sum(1 for row in cleaned_table for cell in row if cell and cell.strip())
        density = filled_cells / max(total_cells, 1)
        
        if not (0.3 <= density <= 0.95):
            return 0.0
        
        return 1.0

    def _check_table_content(self, cleaned_table) -> float:
        """Check if content looks like typical table data."""
        if not cleaned_table:
            return 0.0
        
        score = 0.0
        total_cells = 0
        
        # Collect all non-empty cells
        all_cells = []
        for row in cleaned_table:
            for cell in row:
                if cell and cell.strip():
                    all_cells.append(cell.strip())
                    total_cells += 1
        
        if total_cells < 6:  # Too few cells to assess
            return 0.0
        
        # 1. Check for numeric content (common in tables)
        numeric_cells = sum(1 for cell in all_cells if self._is_numeric_like(cell))
        numeric_ratio = numeric_cells / total_cells
        
        if numeric_ratio > 0.3:  # Good amount of numeric data
            score += 0.4
        elif numeric_ratio > 0.1:  # Some numeric data
            score += 0.2
        
        # 2. Check cell length consistency (tables have concise cells)
        short_cells = sum(1 for cell in all_cells if len(cell.split()) <= 5)
        short_ratio = short_cells / total_cells
        
        if short_ratio > 0.7:  # Most cells are short
            score += 0.3
        elif short_ratio > 0.5:  # Many cells are short
            score += 0.15
        
        # 3. Check for table-like patterns
        table_patterns = 0
        text_sample = ' '.join(all_cells[:20])  # Sample for pattern checking
        
        # Currency, percentages, measurements
        if re.search(r'[€$£¥]\s*\d+|(\d+[.,]\d*\s*[%€$£¥])', text_sample):
            table_patterns += 1
        
        # Statistical patterns
        if re.search(r'[<>=≤≥]\s*\d+|p\s*[<>=]\s*\d', text_sample):
            table_patterns += 1
        
        # Medical/scientific units
        if re.search(r'\d+\s*(mg|ml|kg|%|mm|cm|years?|days?)\b', text_sample, re.IGNORECASE):
            table_patterns += 1
        
        if table_patterns > 0:
            score += 0.3
        
        return min(score, 1.0)


    def _group_words_into_lines(self, words, y_tolerance=3):
        """Group extracted words into lines based on their vertical position."""
        if not words:
            return []

        sorted_words = sorted(words, key=lambda w: w.get("top", 0))

        lines = []
        current_line = []
        current_top = None

        for word in sorted_words:
            top = word.get("top", 0)
            if current_top is None or abs(top - current_top) <= y_tolerance:
                current_line.append(word)
                if current_top is None:
                    current_top = top
            else:
                lines.append(sorted(current_line, key=lambda w: w.get("x0", 0)))
                current_line = [word]
                current_top = top

        if current_line:
            lines.append(sorted(current_line, key=lambda w: w.get("x0", 0)))

        return lines


    def find_table_title(self, page, table_region=None):
        """Locate a nearby heading that likely serves as the table title."""
        try:
            # Extract words with positioning
            words = page.extract_words(x_tolerance=3, y_tolerance=3)
            if not words:
                return None
            
            # Get lines of text
            lines = self._group_words_into_lines(words)
            if not lines:
                return None
            
            # If we have table region info, look above it
            search_lines = lines
            if table_region:
                # Look for title in lines above the table region
                table_top = min(word.get('top', 0) for line in table_region for word in line if isinstance(word, dict))
                search_lines = [line for line in lines if any(word.get('top', 0) < table_top - 10 for word in line)]
            
            # Look for table title patterns in reverse order (closest to table first)
            for line in reversed(search_lines):
                line_text = ' '.join([w.get('text', '') for w in line if w.get('text')])
                
                # Check for table title patterns
                for pattern in self.table_title_patterns:
                    if re.search(pattern, line_text, re.IGNORECASE):
                        return line_text.strip()
                
                # Check for pharmaceutical table indicators
                pharma_indicators = ['treatment', 'therapy', 'medicinal', 'costs', 'dosage', 'consumption']
                if any(indicator in line_text.lower() for indicator in pharma_indicators):
                    # Make sure it's not too long (likely not a title if > 100 chars)
                    if len(line_text) <= 100:
                        return line_text.strip()
            
            # Fallback: look for any short line with key pharmaceutical terms
            for line in reversed(search_lines[-10:]):  # Last 10 lines before table
                line_text = ' '.join([w.get('text', '') for w in line if w.get('text')])
                if (len(line_text.split()) <= 10 and 
                    any(term in line_text.lower() for term in ['table', 'designation', 'treatment', 'costs', 'therapy'])):
                    return line_text.strip()
            
            return None
            
        except Exception as e:
            print(f"      Error finding table title: {e}")
            return None

    def _estimate_table_region(self, table_data):
        """Estimate the region occupied by a table for proximity detection."""
        # Simple region estimation - in practice this could be more sophisticated
        return {
            "estimated_rows": len(table_data) if table_data else 0,
            "estimated_cols": len(table_data[0]) if table_data and table_data[0] else 0
        }

    def _create_table_metadata(self, table_data, table_title, extraction_method, page_num):
        """Create comprehensive metadata for detected tables."""
        metadata = {
            "original_rows": len(table_data) if table_data else 0,
            "extraction_method": extraction_method,
            "has_title": bool(table_title),
            "page": page_num
        }
        
        # Add medical-specific metadata
        if table_data:
            cleaned_table = self.pdf_processor.clean_table_data(table_data)
            all_text = ' '.join([' '.join(row) for row in cleaned_table if row])
            
            metadata.update({
                "contains_dosage": bool(re.search(r'\d+\s*(?:mg|μg|ml)', all_text, re.IGNORECASE)),
                "contains_pricing": bool(re.search(r'[€$]\s*\d+|cost|price', all_text, re.IGNORECASE)),
                "contains_medication": bool(re.search(r'sorafenib|lenvatinib|treatment|therapy', all_text, re.IGNORECASE)),
                "table_title": table_title if table_title else None,
                "narrative_length": 0  # Will be updated after narrative creation
            })
        
        return metadata

    def convert_table_to_narrative(self, table_data, table_title=None):
        """Convert table to hybrid format with narrative and structured metadata."""
        if not table_data or len(table_data) == 0:
            return ""
        
        cleaned_table = self.pdf_processor.clean_table_data(table_data)
        if len(cleaned_table) == 0:
            return ""
        
        # Generate narrative (existing logic)
        narrative_parts = []
        
        if table_title:
            narrative_parts.append(f"Table Title: {table_title}")
            narrative_parts.append("")
        
        headers = self.identify_table_headers(cleaned_table)
        
        if headers:
            narrative_parts.append(f"Table contains the following columns: {', '.join(headers)}")
            data_rows = cleaned_table[1:] if len(cleaned_table) > 1 else []
            
            for row_idx, row in enumerate(data_rows, 1):
                row_description = self.create_row_description(headers, row, row_idx)
                if row_description:
                    narrative_parts.append(row_description)
        else:
            title_part = f"Table Title: {table_title}\n\n" if table_title else ""
            narrative_content = title_part + self.convert_headerless_table(cleaned_table)
            return narrative_content
        
        narrative_text = "\n".join(narrative_parts)
        
        return narrative_text


    def _classify_table_content(self, text):
        """Classify the primary content type of the table."""
        text_lower = text.lower()
        
        if re.search(r'cost|price|€|\$', text_lower):
            return "pricing"
        elif re.search(r'\d+\s*(?:mg|μg|ml)', text_lower):
            return "dosage"
        elif re.search(r'patient|study|trial', text_lower):
            return "clinical_data"
        elif re.search(r'treatment|therapy|medicinal', text_lower):
            return "treatment_info"
        else:
            return "general"
    
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

    def is_numeric(self, text: str) -> bool:
        """Check whether a text string represents a numeric value."""
        if not text or not isinstance(text, str):
            return False
        
        text = text.strip()
        if not text:
            return False
        
        # Remove common non-numeric characters that might appear in numbers
        cleaned = text.replace(',', '').replace(' ', '')
        
        try:
            # Try to convert to float
            float(cleaned)
            return True
        except (ValueError, TypeError):
            pass
        
        # Check for percentage
        if text.endswith('%'):
            try:
                float(text[:-1].replace(',', '').replace(' ', ''))
                return True
            except (ValueError, TypeError):
                pass
        
        # Check for simple patterns like "< 0.001" or "> 100"
        try:
            if re.match(r'^[<>=≤≥]\s*[\d.,]+$', text):
                return True
        except (TypeError, re.error):
            pass
        
        # Check for ranges like "1.2-3.4"
        try:
            if re.match(r'^\d+\.?\d*[-–]\d+\.?\d*$', text):
                return True
        except (TypeError, re.error):
            pass
        
        return False
    
    def _is_numeric_like(self, text: str) -> bool:
        """Simple check if text represents numeric data."""
        if not text:
            return False
        
        # Pure numbers
        if re.match(r'^\d+([.,]\d+)?$', text):
            return True
        
        # Numbers with units/symbols
        if re.match(r'^\d+([.,]\d+)?\s*[%€$£¥]$', text):
            return True
        
        # Ranges
        if re.match(r'^\d+([.,]\d+)?\s*[-–]\s*\d+([.,]\d+)?$', text):
            return True
        
        # Comparison operators
        if re.match(r'^[<>=≤≥]\s*\d+([.,]\d+)?$', text):
            return True

        return False

    def has_explicit_table_structure(self, page) -> bool:
        """Check for a clear table grid on the page."""
        try:
            lines = page.lines if hasattr(page, 'lines') else []
            if len(lines) < 6:
                return False

            horizontal_lines = []
            vertical_lines = []

            for line in lines:
                x1, y1, x2, y2 = line.get('x0', 0), line.get('y0', 0), line.get('x1', 0), line.get('y1', 0)
                width = abs(x2 - x1)
                height = abs(y2 - y1)

                if width > 100 and height < 3:
                    horizontal_lines.append(line)
                elif height > 30 and width < 3:
                    vertical_lines.append(line)

            if len(horizontal_lines) >= 3 and len(vertical_lines) >= 2:
                return self._check_line_intersections(horizontal_lines, vertical_lines)

            return False
        except Exception:
            return False

    def _check_line_intersections(self, h_lines, v_lines) -> bool:
        """Simple intersection check for grid formation."""
        try:
            intersections = 0

            for h_line in h_lines[:5]:
                h_y = h_line.get('y0', 0)
                h_x1, h_x2 = h_line.get('x0', 0), h_line.get('x1', 0)

                for v_line in v_lines[:4]:
                    v_x = v_line.get('x0', 0)
                    v_y1, v_y2 = v_line.get('y0', 0), v_line.get('y1', 0)

                    if (min(h_x1, h_x2) <= v_x <= max(h_x1, h_x2) and
                            min(v_y1, v_y2) <= h_y <= max(v_y1, v_y2)):
                        intersections += 1

            expected_min = min(len(h_lines), 5) * min(len(v_lines), 4) * 0.3
            return intersections >= expected_min
        except Exception:
            return False


    def extract_tables_ultra_strict(self, page):
        """Extract tables with strict visual settings."""
        try:
            explicit_v_lines = self.detect_vertical_table_lines_strict(page)
            explicit_h_lines = self.detect_horizontal_table_lines_strict(page)

            if explicit_v_lines and explicit_h_lines:
                tables = page.extract_tables(
                    table_settings={
                        "vertical_strategy": "explicit",
                        "horizontal_strategy": "explicit",
                        "explicit_vertical_lines": explicit_v_lines,
                        "explicit_horizontal_lines": explicit_h_lines,
                        "min_words_vertical": 3,
                        "min_words_horizontal": 2,
                        "keep_blank_chars": False,
                        "text_tolerance": 2,
                        "intersection_tolerance": 2,
                    }
                )
                if tables:
                    return tables

            tables = page.extract_tables(
                table_settings={
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "min_words_vertical": 4,
                    "min_words_horizontal": 3,
                    "keep_blank_chars": False,
                    "text_tolerance": 1,
                    "intersection_tolerance": 1,
                }
            )

            return tables if tables else []
        except Exception:
            return []

    def detect_vertical_table_lines_strict(self, page) -> List[float]:
        """Detect vertical lines with relaxed criteria."""
        try:
            lines = page.lines if hasattr(page, 'lines') else []
            vertical_lines = []

            for line in lines:
                x1, y1, x2, y2 = line.get('x0', 0), line.get('y0', 0), line.get('x1', 0), line.get('y1', 0)
                line_width = abs(x2 - x1)
                line_height = abs(y2 - y1)

                if line_height > 25 and line_width < 6 and line_height > line_width * 8:
                    vertical_lines.append(x1)

            unique_lines = []
            for x in sorted(vertical_lines):
                if not unique_lines or abs(x - unique_lines[-1]) > 5:
                    unique_lines.append(x)

            return unique_lines
        except Exception:
            return []

    def detect_horizontal_table_lines_strict(self, page) -> List[float]:
        """Detect horizontal lines with relaxed criteria."""
        try:
            lines = page.lines if hasattr(page, 'lines') else []
            horizontal_lines = []

            for line in lines:
                x1, y1, x2, y2 = line.get('x0', 0), line.get('y0', 0), line.get('x1', 0), line.get('y1', 0)
                line_width = abs(x2 - x1)
                line_height = abs(y2 - y1)

                if line_width > 40 and line_height < 6 and line_width > line_height * 12:
                    horizontal_lines.append(y1)

            unique_lines = []
            for y in sorted(horizontal_lines):
                if not unique_lines or abs(y - unique_lines[-1]) > 3:
                    unique_lines.append(y)

            return unique_lines
        except Exception:
            return []


    def detect_complete_tables(self, pdf):
        """Enhanced multi-pass table detection with graduated sensitivity."""
        tables_info = []
        detected_regions = []  # Track regions where tables were found

        print("  🔍 Multi-pass table detection with graduated sensitivity...")

        for page_num, page in enumerate(pdf.pages, start=1):
            page_tables = []
            
            # Pass 1: Standard detection (sensitivity level 1)
            if self.has_explicit_table_structure(page):
                visual_tables = self.extract_tables_ultra_strict(page)
                if visual_tables:
                    page_tables.extend(visual_tables)

            if not page_tables:
                try:
                    standard_tables = page.extract_tables()
                    if standard_tables:
                        page_tables.extend(standard_tables)
                except Exception as e:
                    print(f"      Page {page_num}: Standard extraction failed: {e}")

            # Validate Pass 1 results
            pass_1_tables = []
            for table_idx, table_data in enumerate(page_tables):
                if self.enhanced_table_validation(table_data, page, page_num, sensitivity_level=1):
                    table_title = self.find_table_title(page)
                    narrative_text = self.convert_table_to_narrative(table_data, table_title)
                    
                    if narrative_text.strip():
                        heading = table_title if table_title else f"Table {table_idx + 1} on page {page_num}"
                        
                        table_info = {
                            "page": page_num,
                            "heading": heading,
                            "text": narrative_text,
                            "table_type": "multi_pass_validated_table",
                            "table_metadata": self._create_table_metadata(table_data, table_title, "pass_1_standard", page_num)
                        }
                        
                        pass_1_tables.append(table_info)
                        # Track detected region for pass 2
                        detected_regions.append((page_num, self._estimate_table_region(table_data)))

            # Pass 2: Relaxed detection near existing tables (sensitivity level 2)
            if detected_regions:
                additional_tables = []
                for table_idx, table_data in enumerate(page_tables):
                    # Skip if already validated in pass 1
                    already_found = any(t["page"] == page_num for t in pass_1_tables)
                    if not already_found and self.enhanced_table_validation(table_data, page, page_num, sensitivity_level=2):
                        table_title = self.find_table_title(page)
                        narrative_text = self.convert_table_to_narrative(table_data, table_title)
                        
                        if narrative_text.strip():
                            heading = table_title if table_title else f"Table {table_idx + 1} on page {page_num}"
                            
                            table_info = {
                                "page": page_num,
                                "heading": heading,
                                "text": narrative_text,
                                "table_type": "multi_pass_validated_table",
                                "table_metadata": self._create_table_metadata(table_data, table_title, "pass_2_relaxed", page_num)
                            }
                            
                            additional_tables.append(table_info)

                pass_1_tables.extend(additional_tables)

            # Pass 3: Medical domain-specific detection (sensitivity level 3)
            medical_tables = []
            for table_idx, table_data in enumerate(page_tables):
                # Skip if already found in previous passes
                already_found = any(t["page"] == page_num for t in pass_1_tables)
                if not already_found and self.enhanced_table_validation(table_data, page, page_num, sensitivity_level=3):
                    table_title = self.find_table_title(page)
                    narrative_text = self.convert_table_to_narrative(table_data, table_title)
                    
                    if narrative_text.strip():
                        heading = table_title if table_title else f"Medical Table {table_idx + 1} on page {page_num}"
                        
                        table_info = {
                            "page": page_num,
                            "heading": heading,
                            "text": narrative_text,
                            "table_type": "multi_pass_validated_table", 
                            "table_metadata": self._create_table_metadata(table_data, table_title, "pass_3_medical", page_num)
                        }
                        
                        medical_tables.append(table_info)

            pass_1_tables.extend(medical_tables)
            
            if pass_1_tables:
                print(f"      Page {page_num}: Found {len(pass_1_tables)} tables via multi-pass detection")
            
            tables_info.extend(pass_1_tables)

        print(f"  ✅ Multi-pass detection complete: {len(tables_info)} validated tables")
        return tables_info


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
        """Initialize processor and gather basic metadata."""
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

        # Initialize table detection class
        self.table_detector = TableDetector(self)

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

    def extract_source_type_from_path(self):
        """Identifies whether this is an HTA submission or clinical guideline."""
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
                    not re.search(r'[.!?:;]', current_line) and
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
        if raw_lines:
            current_group = raw_lines[0]

            for next_line in raw_lines[1:]:
                if not current_group:
                    current_group = next_line
                    continue
                    
                current_avg_size = statistics.mean([w['size'] for w in current_group if 'size' in w])
                next_avg_size = statistics.mean([w['size'] for w in next_line if 'size' in w])
                current_bold = all('bold' in w.get('fontname', '').lower() for w in current_group)
                next_bold = all('bold' in w.get('fontname', '').lower() for w in next_line)
                
                # Calculate vertical gap safely
                current_bottom = max(w.get('bottom', 0) for w in current_group)
                next_top = min(w.get('top', 0) for w in next_line)
                vertical_gap = next_top - current_bottom

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
            top_pos = min(lw.get('top', 0) for lw in line_words)
            bottom_pos = max(lw.get('bottom', 0) for lw in line_words)

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

    def extract_preliminary_chunks(self):
        """
        Main function that:
        1. Extracts text and identifies headings from each page (skipping footnotes, boilerplate, etc.).
        2. Stores headings per page in self.page_headings_map.
        3. Extracts tables using enhanced multi-pass detection and hybrid metadata conversion.
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
                        for column_text in column_texts:
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

                # Now extract tables with enhanced multi-pass detection and hybrid metadata conversion
                tables_info = self.table_detector.detect_complete_tables(pdf)

                # Insert each table as its own chunk with hybrid metadata
                for tinfo in tables_info:
                    pg = tinfo["page"]
                    heading_for_table = tinfo["heading"]
                    table_text = tinfo["text"]
                    
                    # Get enhanced metadata from table detection
                    table_metadata = tinfo.get("table_metadata", {})
                    
                    # Update narrative length in metadata now that we have the text
                    if "narrative_length" in table_metadata:
                        table_metadata["narrative_length"] = len(table_text)
                    
                    # Create enhanced table chunk with hybrid metadata support
                    table_chunk = {
                        "heading": heading_for_table,
                        "text": table_text,
                        "start_page": pg,
                        "end_page": pg,
                        "table_type": tinfo["table_type"],
                        "table_metadata": table_metadata
                    }
                    
                    # Add structured metadata for hybrid approach
                    if table_metadata.get("contains_dosage") or table_metadata.get("contains_pricing"):
                        # This table contains medical data, add extraction hints
                        extraction_hints = {
                            "key_numbers": self._extract_key_numbers(table_text),
                            "key_terms": self._extract_key_medical_terms(table_text),
                            "relationships": self._identify_data_relationships(table_metadata)
                        }
                        table_metadata["extraction_hints"] = extraction_hints
                    
                    chunks.append(table_chunk)

                # Create final document structure with enhanced table summary
                final_structure = {
                    "doc_id": self.doc_id,
                    "created_date": self.created_date,
                    "country": self.country,
                    "source_type": self.source_type,
                    "chunks": chunks
                }
                
                # Add enhanced table summary with hybrid metadata insights
                if tables_info:
                    table_summary = {
                        "total_tables_found": len(tables_info),
                        "tables_by_page": {},
                        "table_storage_info": "Tables stored as individual chunks with hybrid metadata (narrative + structured)",
                        "medical_table_insights": {
                            "pricing_tables": sum(1 for t in tables_info if t.get("table_metadata", {}).get("contains_pricing", False)),
                            "dosage_tables": sum(1 for t in tables_info if t.get("table_metadata", {}).get("contains_dosage", False)),
                            "medication_tables": sum(1 for t in tables_info if t.get("table_metadata", {}).get("contains_medication", False)),
                            "multi_pass_detection_summary": {
                                "pass_1_standard": sum(1 for t in tables_info if t.get("table_metadata", {}).get("extraction_method") == "pass_1_standard"),
                                "pass_2_relaxed": sum(1 for t in tables_info if t.get("table_metadata", {}).get("extraction_method") == "pass_2_relaxed"),
                                "pass_3_medical": sum(1 for t in tables_info if t.get("table_metadata", {}).get("extraction_method") == "pass_3_medical")
                            }
                        }
                    }
                    
                    for tinfo in tables_info:
                        page_num = tinfo["page"]
                        if page_num not in table_summary["tables_by_page"]:
                            table_summary["tables_by_page"][page_num] = []
                        
                        page_table_info = {
                            "heading": tinfo["heading"],
                            "narrative_length": len(tinfo["text"]),
                            "extraction_method": tinfo.get("table_metadata", {}).get("extraction_method", "unknown"),
                            "original_rows": tinfo.get("table_metadata", {}).get("original_rows", "unknown"),
                            "primary_content_type": tinfo.get("table_metadata", {}).get("primary_content_type", "general"),
                            "contains_medical_data": bool(
                                tinfo.get("table_metadata", {}).get("contains_dosage") or 
                                tinfo.get("table_metadata", {}).get("contains_pricing") or 
                                tinfo.get("table_metadata", {}).get("contains_medication")
                            )
                        }
                        
                        table_summary["tables_by_page"][page_num].append(page_table_info)
                    
                    final_structure["_table_detection_summary"] = table_summary
                    
                    # Report enhanced table storage and detection insights
                    medical_count = table_summary["medical_table_insights"]["pricing_tables"] + table_summary["medical_table_insights"]["dosage_tables"]
                    print(f"  📍 Table storage: {len(tables_info)} tables stored as chunks with hybrid metadata")
                    print(f"  🏥 Medical table insights: {medical_count} tables contain dosage/pricing data")
                    print(f"  📊 Multi-pass detection: P1={table_summary['medical_table_insights']['multi_pass_detection_summary']['pass_1_standard']}, "
                            f"P2={table_summary['medical_table_insights']['multi_pass_detection_summary']['pass_2_relaxed']}, "
                            f"P3={table_summary['medical_table_insights']['multi_pass_detection_summary']['pass_3_medical']}")
                    print(f"  📋 Enhanced metadata added to '_table_detection_summary' field")

                return final_structure

        except Exception as e:
            print(f"Error reading {self.pdf_path}: {e}")
            # Try fallback method if primary extraction fails
            return self.extract_using_fallback()

    def _extract_key_numbers(self, text):
        """Extract key numerical values for extraction hints."""
        numbers = re.findall(r'\d+(?:[.,]\d+)?(?:\s*[€$%])?', text)
        return numbers[:10]  # Limit to first 10 to avoid bloat

    def _extract_key_medical_terms(self, text):
        """Extract key medical terms for extraction hints."""
        medical_terms = []
        common_terms = ['sorafenib', 'lenvatinib', 'treatment', 'therapy', 'dose', 'cost', 'patient', 'study']
        text_lower = text.lower()
        
        for term in common_terms:
            if term in text_lower:
                medical_terms.append(term)
        
        return medical_terms

    def _identify_data_relationships(self, metadata):
        """Identify potential data relationships based on metadata."""
        relationships = []
        
        if metadata.get("contains_dosage") and metadata.get("contains_medication"):
            relationships.append("drug-dose")
        
        if metadata.get("contains_pricing") and metadata.get("contains_medication"):
            relationships.append("drug-cost")
        
        if metadata.get("contains_dosage"):
            relationships.append("dose-frequency")
        
        if metadata.get("contains_comparisons"):
            relationships.append("comparative-analysis")
        
        return relationships


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
        """Process all PDFs in a folder and save extracted JSON data."""
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
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from langdetect import detect
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import copy

class Translator:
    """
    Enhanced Translator class with simplified medical term preservation,
    improved chunking strategy, and better quality assessment.
    """

    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.english_chunks_preserved = 0
        self.chunks_translated = 0
        self.total_translation_start_time = None  

        self.processing_start_time = None
        self.tier_attempts = {}
        self.translation_decisions = {}

        # Simplified medical term whitelist - only truly critical terms
        self.critical_medical_terms = {
            # Drug names that should never be translated
            'drug_names': [
                'sotorasib', 'lumykras', 'pembrolizumab', 'keytruda', 'atezolizumab',
                'nivolumab', 'durvalumab', 'ipilimumab', 'bevacizumab', 'cetuximab',
                'panitumumab', 'trastuzumab', 'pertuzumab', 'ramucirumab',
                'nintedanib', 'osimertinib', 'erlotinib', 'gefitinib', 'afatinib',
                'crizotinib', 'alectinib', 'brigatinib', 'lorlatinib',
                'pemetrexed', 'docetaxel', 'paclitaxel', 'carboplatin', 'cisplatin',
                'gemcitabine', 'vinorelbine', 'etoposide'
            ],
            # Medical abbreviations
            'abbreviations': [
                'NSCLC', 'NDRP', 'SCLC', 'EGFR', 'ALK', 'ROS1', 'KRAS', 'BRAF',
                'PD-L1', 'TMB', 'MSI', 'MMR', 'HER2', 'MET', 'RET', 'NTRK',
                'PFS', 'OS', 'ORR', 'DCR', 'DOR', 'TTR', 'TTP', 'QoL',
                'ECOG', 'KPS', 'PS', 'CR', 'PR', 'SD', 'PD',
                'AE', 'SAE', 'TEAE', 'CTCAE', 'NCI', 'WHO', 'RECIST',
                'ITT', 'PP', 'mITT', 'FAS', 'SS'
            ],
            # Regulatory/organizational terms
            'regulatory': [
                'FDA', 'EMA', 'NICE', 'SMC', 'G-BA', 'HAS', 'AIFA', 'TGA', 'PMDA',
                'CADTH', 'IQWIG', 'HTA', 'QALY', 'ICER', 'ICUR'
            ]
        }

        # Model tiers - favor tier 2 for medical documents
        self.model_tiers = {
            1: {
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
            2: {
                'nllb_xl': 'facebook/nllb-200-3.3B',
                'nllb_large': 'facebook/nllb-200-1.3B',
                'fallback': 'facebook/nllb-200-distilled-1.3B',
                'description': 'Highest quality models - preferred for medical documents'
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

        # Updated quality thresholds - lower threshold to prefer tier 2
        self.quality_thresholds = {
            'tier_1_to_2_threshold': 0.50,  # Lower threshold to prefer tier 2
            'minimum_acceptable_quality': 0.40,
            'medical_preservation_threshold': 0.70,  # Higher expectation for medical terms
        }

        # Improved chunking parameters for medical documents
        self.chunk_params = {
            'max_chars': 2500,  # Increased for better context
            'overlap_chars': 150,
            'min_chunk_chars': 300,
        }

        # Table detection patterns
        self.table_patterns = [
            r'(?:Row|Column)\s+\d+[:\s]',
            r'Table\s+(?:Title|contains|shows)',
            r'\|\s*[^|\n]+\s*\|\s*[^|\n]+\s*\|',
            r'€\s*\d+(?:[.,]\d+)*',
            r'\d+(?:[.,]\d+)*\s*€',
        ]

        # CUDA setup
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

    def extract_medical_terms_database(self, text: str) -> Dict[str, str]:
        """Simplified medical term extraction using whitelist approach."""
        if not text:
            return {}
            
        term_db = {}
        term_counter = 0
        
        # Combine all critical terms
        all_terms = []
        for category, terms in self.critical_medical_terms.items():
            all_terms.extend(terms)
        
        # Sort by length (longest first) to avoid partial matches
        all_terms.sort(key=len, reverse=True)
        
        text_lower = text.lower()
        
        for term in all_terms:
            term_lower = term.lower()
            # Simple case-insensitive search
            if term_lower in text_lower:
                # Find actual case-sensitive matches
                pattern = r'\b' + re.escape(term) + r'\b'
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if match not in term_db.values():
                        term_id = f"MEDTERM{term_counter:03d}"
                        term_db[term_id] = match
                        term_counter += 1
        
        return term_db

    def apply_medical_term_protection(self, text: str, term_db: Dict[str, str]) -> str:
        """Apply simplified term protection."""
        if not text or not term_db:
            return text
            
        protected_text = text
        
        # Sort terms by length (longest first) to avoid partial replacements
        sorted_terms = sorted(term_db.items(), key=lambda x: len(x[1]), reverse=True)
        
        for term_id, term in sorted_terms:
            # Use word boundaries for better matching
            pattern = r'\b' + re.escape(term) + r'\b'
            protected_text = re.sub(pattern, term_id, protected_text, flags=re.IGNORECASE)
        
        return protected_text

    def restore_medical_terms(self, text: str, term_db: Dict[str, str]) -> str:
        """Restore terms with validation."""
        if not text or not term_db:
            return text
            
        restored_text = text
        unreplaced_count = 0
        
        for term_id, term in term_db.items():
            if term_id in restored_text:
                restored_text = restored_text.replace(term_id, term)
            else:
                unreplaced_count += 1
        
        # Log if restoration failed
        if unreplaced_count > 0:
            print(f"      ⚠️  {unreplaced_count} medical terms failed to restore")
        
        return restored_text

    def count_tokens_accurately(self, text: str) -> int:
        """Simplified token counting."""
        if not text:
            return 0
        
        # Simple approximation: 4 characters per token on average
        return len(text) // 4

    def split_text_into_chunks(self, text: str) -> List[str]:
        """Improved text splitting with better sentence boundary detection."""
        if not text:
            return []
            
        max_chars = self.chunk_params['max_chars']
        
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        remaining_text = text
        
        while remaining_text:
            if len(remaining_text) <= max_chars:
                chunks.append(remaining_text)
                break
            
            # Find the best split point - prioritize sentence boundaries
            split_point = max_chars
            
            # Look for sentence endings first
            for i in range(max_chars, max_chars // 2, -1):
                if i >= len(remaining_text):
                    continue
                    
                char = remaining_text[i]
                # Strong sentence boundaries
                if char in '.!?' and i + 1 < len(remaining_text):
                    next_char = remaining_text[i + 1]
                    if next_char.isspace() or next_char.isupper():
                        split_point = i + 1
                        break
                # Paragraph breaks
                elif char == '\n' and remaining_text[i:i+2] == '\n\n':
                    split_point = i + 2
                    break
            
            # If no good sentence boundary found, look for word boundaries
            if split_point == max_chars:
                for i in range(max_chars, max_chars // 2, -1):
                    if i >= len(remaining_text):
                        continue
                    if remaining_text[i].isspace():
                        split_point = i
                        break
            
            chunk = remaining_text[:split_point].strip()
            if chunk:
                chunks.append(chunk)
            
            remaining_text = remaining_text[split_point:].strip()
        
        return [chunk for chunk in chunks if chunk.strip()]

    def adaptive_chunk_for_translation(self, text: str) -> List[str]:
        """Simplified chunking for medical documents."""
        if not text:
            return []
            
        # Check if chunking is needed
        if self.count_tokens_accurately(text) <= 500 and len(text) <= self.chunk_params['max_chars']:
            return [text]
        
        print(f"      🔨 Chunking needed for large text ({len(text)} chars)")
        
        chunks = self.split_text_into_chunks(text)
        
        print(f"      ✓ Created {len(chunks)} chunks with improved sentence boundaries")
        return chunks

    def is_table_content(self, text: str) -> bool:
        """Check if text is table content."""
        if not text:
            return False
            
        for pattern in self.table_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # Check for structural indicators
        line_count = len(text.split('\n'))
        pipe_density = text.count('|') / max(len(text), 1)
        numeric_density = len(re.findall(r'\d+(?:\.\d+)?', text)) / max(len(text.split()), 1)
        
        return (line_count >= 5 and pipe_density > 0.05) or numeric_density > 0.3

    def assess_translation_quality(self, original_text: str, translated_text: str, language: str) -> Dict[str, float]:
        """Enhanced quality assessment with medical focus and English quality checks."""
        if not translated_text.strip():
            return {'overall': 0.0, 'english_quality': 0.0, 'medical_preservation': 0.0, 'has_placeholders': 1.0}
        
        # Check for unreplaced placeholders
        placeholder_score = 1.0
        if re.search(r'MEDTERM\d+', translated_text) or re.search(r'\[MED\d+\]', translated_text):
            placeholder_score = 0.0
            print(f"      ⚠️  Found unreplaced placeholders in translation")
        
        # Basic English quality checks
        english_quality = 1.0
        try:
            detected_lang = detect(translated_text)
            if detected_lang != 'en':
                english_quality *= 0.3
        except:
            english_quality *= 0.5
        
        # Check for basic English patterns
        if not re.search(r'\b(?:the|and|of|to|a|in|is|it|that|for|on|with|as)\b', translated_text.lower()):
            english_quality *= 0.5
        
        # Check for excessive repetition
        words = translated_text.split()
        if len(words) > 10:
            unique_words = set(words)
            if len(unique_words) / len(words) < 0.5:
                english_quality *= 0.5
        
        # Length ratio check
        len_ratio = len(translated_text) / max(len(original_text), 1)
        length_score = 1.0 if 0.5 <= len_ratio <= 2.5 else 0.5
        
        # Medical term preservation check (simplified)
        orig_terms = self.extract_medical_terms_database(original_text)
        medical_preservation = 1.0
        if orig_terms:
            preserved_count = 0
            for term in orig_terms.values():
                if term.lower() in translated_text.lower():
                    preserved_count += 1
            medical_preservation = preserved_count / len(orig_terms)
        
        # Overall weighted score
        overall_score = (
            placeholder_score * 0.30 +  # Critical: no unreplaced placeholders
            english_quality * 0.35 +     # Important: good English
            medical_preservation * 0.25 + # Important: preserve medical terms
            length_score * 0.10           # Basic: reasonable length
        )
        
        return {
            'overall': overall_score,
            'english_quality': english_quality,
            'medical_preservation': medical_preservation,
            'has_placeholders': 1.0 - placeholder_score,
            'length_score': length_score
        }

    def assess_document_quality(self, translated_data: dict, original_data: dict, language: str) -> Dict[str, float]:
        """Document-level quality assessment."""
        if 'chunks' not in translated_data or 'chunks' not in original_data:
            return {'overall': 0.0, 'chunk_count': 0}
        
        chunk_qualities = []
        
        for orig_chunk, trans_chunk in zip(original_data['chunks'], translated_data['chunks']):
            if 'text' in orig_chunk and 'text' in trans_chunk:
                if orig_chunk['text'] and trans_chunk['text']:
                    text_quality = self.assess_translation_quality(
                        orig_chunk['text'], trans_chunk['text'], language
                    )
                    chunk_qualities.append(text_quality)
        
        if not chunk_qualities:
            return {'overall': 0.0, 'chunk_count': 0}
        
        # Calculate averages
        metrics = ['overall', 'english_quality', 'medical_preservation', 'has_placeholders', 'length_score']
        final_scores = {}
        
        for metric in metrics:
            scores = [quality[metric] for quality in chunk_qualities if metric in quality]
            final_scores[metric] = sum(scores) / len(scores) if scores else 0.0
        
        final_scores['chunk_count'] = len(chunk_qualities)
        
        return final_scores

    def should_retranslate_with_higher_tier(self, quality_scores: Dict[str, float]) -> bool:
        """Updated logic to prefer tier 2 for medical documents."""
        overall_quality = quality_scores.get('overall', 0.0)
        medical_score = quality_scores.get('medical_preservation', 0.0)
        has_placeholders = quality_scores.get('has_placeholders', 0.0)
        
        # Always retranslate if placeholders weren't restored
        if has_placeholders > 0.1:
            print(f"    📉 Unreplaced placeholders detected - retranslating")
            return True
        
        if overall_quality < self.quality_thresholds['tier_1_to_2_threshold']:
            print(f"    📉 Overall quality {overall_quality:.3f} below threshold")
            return True
        
        if medical_score < self.quality_thresholds['medical_preservation_threshold']:
            print(f"    🏥 Medical preservation {medical_score:.3f} below threshold")
            return True
        
        return False

    def compare_translation_quality(self, quality_1: Dict[str, float], quality_2: Dict[str, float]) -> int:
        """Compare two quality scores with focus on medical preservation."""
        # Strongly penalize unreplaced placeholders
        if quality_1.get('has_placeholders', 0) < quality_2.get('has_placeholders', 0):
            return 1
        elif quality_1.get('has_placeholders', 0) > quality_2.get('has_placeholders', 0):
            return 2
        
        # Compare overall quality
        if abs(quality_1['overall'] - quality_2['overall']) > 0.05:
            return 1 if quality_1['overall'] > quality_2['overall'] else 2
        
        # Secondary: medical preservation
        med_diff = quality_1.get('medical_preservation', 0) - quality_2.get('medical_preservation', 0)
        if abs(med_diff) > 0.08:
            return 1 if med_diff > 0 else 2
        
        return 1

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
            ' he ', ' was ', ' for ', ' on ', ' are ', ' as ', ' with ', ' his ', ' they '
        ]

        english_count = sum(1 for word in english_words if word in f' {text_lower} ')
        total_words = len(text.split())

        if total_words > 5:
            english_ratio = english_count / total_words
            return english_ratio > 0.1

        return False

    def check_model_availability(self, model_name: str) -> bool:
        """Check if a model is available."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            return True
        except Exception as e:
            print(f"    ✗ Model {model_name} not available: {str(e)[:50]}")
            return False

    def get_available_models_for_tier(self, tier: int, language: str) -> List[str]:
        """Get available models for a tier and language."""
        available_models = []
        
        if tier == 1:
            helsinki_models = self.model_tiers[1]['helsinki_models']
            if language in helsinki_models:
                model_name = helsinki_models[language]
                if self.check_model_availability(model_name):
                    available_models.append(model_name)
            
            if language in self.nllb_lang_mapping:
                fallback_model = self.model_tiers[1]['fallback']
                if self.check_model_availability(fallback_model):
                    available_models.append(fallback_model)
                    
        elif tier == 2:
            if language in self.nllb_lang_mapping:
                for model_key in ['nllb_xl', 'nllb_large', 'fallback']:
                    if model_key in self.model_tiers[2]:
                        model_name = self.model_tiers[2][model_key]
                        if self.check_model_availability(model_name):
                            available_models.append(model_name)
        
        return available_models

    def load_helsinki_model(self, model_name: str, language: str) -> Optional[Any]:
        """Load Helsinki model."""
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
                    self.device = "cpu"
                    print(f"    ✓ Helsinki model loaded successfully on CPU")
                    return translator
                except Exception as cpu_error:
                    print(f"    ✗ Helsinki model failed on CPU: {str(cpu_error)[:50]}")
                    return None
            else:
                raise e

    def load_nllb_model(self, model_name: str, language: str) -> Optional[Any]:
        """Load NLLB model."""
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

            def nllb_translate(text, generation_params=None):
                try:
                    tgt_lang = 'eng_Latn'
                    max_input_length = 600  # Increased for better context
                    
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_input_length)
                    if self.device.startswith("cuda"):
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    gen_kwargs = {
                        'forced_bos_token_id': tokenizer.convert_tokens_to_ids(tgt_lang),
                        'max_length': 400,  # Increased for better translations
                        'num_beams': 4,     # More beams for better quality
                        'length_penalty': 1.0,
                        'do_sample': False,
                        'no_repeat_ngram_size': 3,
                        'repetition_penalty': 1.2,
                    }

                    with torch.no_grad():
                        translated_tokens = model.generate(**inputs, **gen_kwargs)

                    translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                    return [{'translation_text': translation}]

                except Exception as e:
                    print(f"      NLLB translation error: {str(e)[:50]}")
                    return [{'translation_text': text}]

            print(f"    ✓ NLLB model loaded successfully on {self.device}")
            return nllb_translate

        except Exception as e:
            print(f"    ✗ NLLB model failed: {str(e)[:50]}")
            return None

    def load_translator_for_tier(self, language: str, tier: int) -> Optional[Any]:
        """Load translator for given language and tier."""
        print(f"    Loading translator for language: {language} (Tier {tier})")
        
        available_models = self.get_available_models_for_tier(tier, language)
        
        if not available_models:
            print(f"    No available models for language {language} at tier {tier}")
            return None
        
        for model_name in available_models:
            print(f"    Trying model: {model_name}")
            
            try:
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

    def load_translator_for_language(self, language: str, target_tier: int = 2):  # Changed default to tier 2
        """Load translator for given language - now defaults to tier 2 for medical docs."""
        if self.current_language == language and self.current_tier == target_tier and self.current_translator:
            return self.current_translator

        self.clear_translator()

        print(f"  🔄 Loading translator for language: {language} (Tier {target_tier})")

        translator = self.load_translator_for_tier(language, target_tier)
        
        if translator:
            self.current_translator = translator
            self.current_language = language
            self.current_tier = target_tier
            print(f"    ✓ Successfully loaded translator for {language}")
            return translator
        else:
            # Try other tier as fallback
            fallback_tier = 1 if target_tier == 2 else 2
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

    def translate_single_chunk(self, text: str, translator, tier: int = 1) -> str:
        """Translate a single chunk using simplified medical term preservation."""
        if not text.strip() or not translator:
            return text

        try:
            # Extract medical terms and create simple database
            term_db = self.extract_medical_terms_database(text)
            
            # Apply protection if terms found
            protected_text = self.apply_medical_term_protection(text, term_db) if term_db else text
            
            # Check if chunking is needed
            chunks = self.adaptive_chunk_for_translation(protected_text)
            
            if len(chunks) == 1:
                # Single chunk translation
                gen_params = {
                    'max_length': 600,  # Increased for better translations
                    'truncation': True,
                    'no_repeat_ngram_size': 3,
                    'repetition_penalty': 1.1,  # Reduced for more natural text
                    'do_sample': False,
                    'num_beams': 4,  # More beams for quality
                }
                
                if hasattr(translator, 'model'):
                    result = translator(protected_text, **gen_params)
                else:
                    result = translator(protected_text, generation_params=gen_params)
                
                translated_text = result[0]['translation_text']
            else:
                # Multi-chunk translation
                translated_chunks = []
                for chunk in chunks:
                    gen_params = {
                        'max_length': 500,
                        'truncation': True,
                        'no_repeat_ngram_size': 3,
                        'repetition_penalty': 1.1,
                        'do_sample': False,
                        'num_beams': 4,
                    }
                    
                    if hasattr(translator, 'model'):
                        result = translator(chunk, **gen_params)
                    else:
                        result = translator(chunk, generation_params=gen_params)
                    
                    translated_chunks.append(result[0]['translation_text'])
                
                translated_text = ' '.join(translated_chunks)
                print(f"      ✓ Merged {len(translated_chunks)} chunks")
            
            # Restore medical terms
            final_text = self.restore_medical_terms(translated_text, term_db) if term_db else translated_text
            
            return final_text.strip()

        except Exception as e:
            print(f"      Translation error: {str(e)[:50]}")
            return text

    def translate_document_with_tier(self, data: dict, language: str, tier: int) -> Tuple[dict, Dict[str, float], Dict[str, Any]]:
        """Translate document with specific tier."""
        print(f"  📝 Translating document with Tier {tier}")
        
        tier_start_time = datetime.now()
        
        translator = self.load_translator_for_language(language, tier)
        if not translator:
            print(f"    ✗ No translator available for Tier {tier}")
            return data, {'overall': 0.0}, {
                'tier': tier,
                'model_loaded': False,
                'processing_time_seconds': 0,
                'model_name': None
            }
        
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
        table_count = 0
        
        print(f"    Processing {total_chunks} chunks with simplified medical term preservation...")
        
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
                    if self.is_table_content(chunk['text']):
                        table_count += 1
                    
                    chunk['text'] = self.translate_single_chunk(
                        chunk['text'], translator, tier
                    )
                    translated_count += 1
        
        processing_time = (datetime.now() - tier_start_time).total_seconds()
        
        print(f"    ✓ Tier {tier} translation complete:")
        print(f"      English chunks: {english_count}")
        print(f"      Translated chunks: {translated_count}")
        print(f"      Table chunks: {table_count}")
        
        # Quality assessment
        quality_scores = self.assess_document_quality(translated_data, data, language)
        
        print(f"    📊 Tier {tier} Quality Assessment:")
        print(f"      Overall: {quality_scores['overall']:.3f}")
        print(f"      English Quality: {quality_scores.get('english_quality', 0):.3f}")
        print(f"      Medical Preservation: {quality_scores.get('medical_preservation', 0):.3f}")
        print(f"      Unreplaced Placeholders: {quality_scores.get('has_placeholders', 0):.3f}")
        
        tier_metadata = {
            'tier': tier,
            'model_loaded': True,
            'model_name': self.current_model_name,
            'processing_time_seconds': processing_time,
            'chunks_found': True,
            'total_chunks': total_chunks,
            'chunks_translated': translated_count,
            'chunks_english': english_count,
            'table_chunks_processed': table_count,
            'quality_scores': quality_scores
        }
        
        return translated_data, quality_scores, tier_metadata

    def clear_translator(self):
        """Clear current translator and free memory."""
        self.current_translator = None
        self.current_language = None
        self.current_tier = None
        self.current_model_name = None

        gc.collect()
        if self.use_cuda and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except:
                pass

    def process_json_file(self, input_path: str, output_path: str):
        """Process a single JSON file."""
        file_name = os.path.basename(input_path)
        print(f"\n📄 Processing: {file_name}")
        
        self.processing_start_time = datetime.now()

        try:
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ✗ Error loading JSON: {e}")
            return

        # Language detection
        all_text_parts = []
        if 'doc_id' in data:
            all_text_parts.append(str(data['doc_id']))

        if 'chunks' in data:
            for chunk in data['chunks']:
                if 'heading' in chunk and chunk['heading']:
                    all_text_parts.append(chunk['heading'])
                if 'text' in chunk and chunk['text']:
                    all_text_parts.append(chunk['text'])

        combined_text = ' '.join(all_text_parts)
        document_language = self.detect_document_language(combined_text)

        # Translation metadata
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
            "table_chunks_processed": 0,
            "total_processing_time_seconds": 0,
            "translation_strategy": "simplified_medical_optimized",
            "medical_preservation_used": "whitelist_approach",
            "table_content_detected": sum(1 for chunk in data.get('chunks', []) 
                                        if self.is_table_content(chunk.get('text', '')))
        }

        if not document_language or document_language == 'en':
            print(f"  📋 Document is English, copying without translation")
            translation_metadata.update({
                "translation_decision": "no_translation_needed_english",
                "total_processing_time_seconds": (datetime.now() - self.processing_start_time).total_seconds()
            })
            
            data["_translation_metadata"] = translation_metadata
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return

        # Check model availability - start with tier 2 for medical documents
        tier_2_available = len(self.get_available_models_for_tier(2, document_language)) > 0
        tier_1_available = len(self.get_available_models_for_tier(1, document_language)) > 0
        
        translation_metadata.update({
            "was_translation_needed": True,
            "tier_1_available": tier_1_available,
            "tier_2_available": tier_2_available
        })
        
        if not tier_1_available and not tier_2_available:
            print(f"  📋 No translation models available for language {document_language}")
            translation_metadata.update({
                "translation_decision": "no_models_available",
                "total_processing_time_seconds": (datetime.now() - self.processing_start_time).total_seconds()
            })
            
            data["_translation_metadata"] = translation_metadata
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return

        # Start with tier 2 for medical documents
        if tier_2_available:
            print(f"  🎯 Starting with Tier 2 translation (preferred for medical docs)")
            tier_2_translation, tier_2_quality, tier_2_metadata = self.translate_document_with_tier(
                data, document_language, 2
            )
            translation_metadata["tier_attempts"].append(tier_2_metadata)
            
            if not self.should_retranslate_with_higher_tier(tier_2_quality):
                print(f"  ✅ Tier 2 quality sufficient (overall: {tier_2_quality['overall']:.3f})")
                final_translation = tier_2_translation
                final_quality = tier_2_quality
                translation_metadata.update({
                    "translation_decision": "tier_2_sufficient",
                    "final_tier_used": 2,
                    "final_model_used": tier_2_metadata["model_name"],
                    "retranslation_occurred": False
                })
            else:
                print(f"  📈 Tier 2 quality insufficient, trying Tier 1 as fallback")
                
                if tier_1_available:
                    self.clear_translator()
                    tier_1_translation, tier_1_quality, tier_1_metadata = self.translate_document_with_tier(
                        data, document_language, 1
                    )
                    translation_metadata["tier_attempts"].append(tier_1_metadata)
                    
                    if self.compare_translation_quality(tier_1_quality, tier_2_quality) == 1:
                        print(f"  🏆 Tier 1 translation is better")
                        final_translation = tier_1_translation
                        final_quality = tier_1_quality
                        translation_metadata.update({
                            "translation_decision": "tier_1_better_than_tier_2",
                            "final_tier_used": 1,
                            "final_model_used": tier_1_metadata["model_name"],
                            "retranslation_occurred": True,
                            "quality_comparison": {
                                "tier_2_overall": tier_2_quality['overall'],
                                "tier_1_overall": tier_1_quality['overall'],
                                "improvement": tier_1_quality['overall'] - tier_2_quality['overall']
                            }
                        })
                    else:
                        print(f"  🥈 Tier 2 translation is still better")
                        final_translation = tier_2_translation
                        final_quality = tier_2_quality
                        translation_metadata.update({
                            "translation_decision": "tier_2_better_than_tier_1",
                            "final_tier_used": 2,
                            "final_model_used": tier_2_metadata["model_name"],
                            "retranslation_occurred": True
                        })
                else:
                    print(f"  ⚠️  Tier 1 not available")
                    final_translation = tier_2_translation
                    final_quality = tier_2_quality
                    translation_metadata.update({
                        "translation_decision": "tier_2_only_tier_1_unavailable",
                        "final_tier_used": 2,
                        "final_model_used": tier_2_metadata["model_name"],
                        "retranslation_occurred": False
                    })
        else:
            print(f"  🎯 Tier 2 not available, using Tier 1")
            final_translation, final_quality, tier_1_metadata = self.translate_document_with_tier(
                data, document_language, 1
            )
            translation_metadata["tier_attempts"].append(tier_1_metadata)
            translation_metadata.update({
                "translation_decision": "tier_1_only_tier_2_unavailable",
                "final_tier_used": 1,
                "final_model_used": tier_1_metadata["model_name"],
                "retranslation_occurred": False
            })

        # Final statistics
        translated_count = 0
        english_count = 0
        table_count = 0
        
        if 'chunks' in final_translation:
            for chunk in final_translation['chunks']:
                if 'text' in chunk and chunk['text']:
                    if self.is_table_content(chunk['text']):
                        table_count += 1
                
                for field in ['heading', 'text']:
                    if field in chunk and chunk[field]:
                        if self.is_english_chunk(chunk[field]):
                            english_count += 1
                        else:
                            translated_count += 1

        total_processing_time = (datetime.now() - self.processing_start_time).total_seconds()
        translation_metadata.update({
            "final_quality_scores": final_quality,
            "chunks_translated": translated_count,
            "chunks_preserved_english": english_count,
            "table_chunks_processed": table_count,
            "total_processing_time_seconds": total_processing_time,
            "processing_completed_timestamp": datetime.now().isoformat()
        })

        final_translation["_translation_metadata"] = translation_metadata

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_translation, f, indent=2, ensure_ascii=False)

        print(f"  ✓ Final result: {english_count} English, {translated_count} translated, {table_count} table chunks")
        print(f"  📊 Final quality: {final_quality['overall']:.3f}")
        print(f"  ⏱️  Processing time: {total_processing_time:.2f}s")

        self.english_chunks_preserved += english_count
        self.chunks_translated += translated_count
        self.clear_translator()

    def process_batch_files(self, file_batch: List[Tuple[str, str]]) -> None:
        """Process a batch of files."""
        print(f"  📦 Processing batch of {len(file_batch)} files")
        
        for input_path, output_path in file_batch:
            try:
                self.process_json_file(input_path, output_path)
            except Exception as e:
                print(f"  ✗ Batch processing error for {os.path.basename(input_path)}: {str(e)[:100]}")
                try:
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    shutil.copy(input_path, output_path)
                    print(f"  📋 Copied original file instead")
                except Exception as copy_error:
                    print(f"  ✗ Failed to copy original: {copy_error}")

    def translate_documents(self):
        """Main translation method with total runtime tracking."""
        print("🚀 Starting improved medical document translation...")
        print("📋 Improved Strategy:")
        print("   • Simplified medical term preservation with whitelist")
        print("   • Improved chunking with larger sizes and better boundaries")
        print("   • Enhanced quality assessment with placeholder detection")
        print("   • Tier 2 models preferred for medical documents")
        print("   • Better English quality validation")

        # Start total runtime tracking
        self.total_translation_start_time = datetime.now()
        print(f"🕐 Translation started at: {self.total_translation_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        os.makedirs(self.output_dir, exist_ok=True)

        # Find files
        json_files = []
        for root, _, files in os.walk(self.input_dir):
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

        # Group files by directory for batch processing
        file_groups = {}
        for input_path, output_path in json_files:
            dir_key = os.path.basename(os.path.dirname(input_path))
            if dir_key not in file_groups:
                file_groups[dir_key] = []
            file_groups[dir_key].append((input_path, output_path))

        # Process groups
        for group_dir, group_files in file_groups.items():
            print(f"\n📦 Processing batch from {group_dir}")
            self.process_batch_files(group_files)

        # Calculate total runtime
        total_translation_end_time = datetime.now()
        total_runtime_seconds = (total_translation_end_time - self.total_translation_start_time).total_seconds()
        total_runtime_minutes = total_runtime_seconds / 60
        total_runtime_hours = total_runtime_minutes / 60

        # Summary with runtime information
        print(f"\n🎉 Improved Medical Translation Complete!")
        print(f"📊 Summary:")
        print(f"   • Total files processed: {total_files}")
        print(f"   • English chunks preserved: {self.english_chunks_preserved}")
        print(f"   • Chunks translated: {self.chunks_translated}")
        print(f"   • Simplified medical term preservation used")
        print(f"   • Improved chunking and quality assessment applied")
        print(f"\n⏱️  Runtime Summary:")
        print(f"   • Start time: {self.total_translation_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   • End time: {total_translation_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   • Total runtime: {total_runtime_seconds:.2f} seconds")
        print(f"   • Total runtime: {total_runtime_minutes:.2f} minutes")
        if total_runtime_hours >= 1:
            print(f"   • Total runtime: {total_runtime_hours:.2f} hours")
        if total_files > 0:
            avg_time_per_file = total_runtime_seconds / total_files
            print(f"   • Average time per file: {avg_time_per_file:.2f} seconds")
            

