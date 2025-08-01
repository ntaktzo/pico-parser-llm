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
    Enhanced Translator class with document-level hierarchical model selection optimized for medical data,
    robust CUDA handling, dynamic term preservation, intelligent quality detection, and adaptive translation chunking.
    """

    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.english_chunks_preserved = 0
        self.chunks_translated = 0

        self.processing_start_time = None
        self.tier_attempts = {}
        self.translation_decisions = {}

        # Dynamic pattern-based preservation system
        self.preservation_patterns = {
            'drug_names': [
                r'\b[A-Z][a-z]+(?:mab|nib|rib|ib|zumab|tinib)\b',  # Common drug suffixes
                r'\b[A-Z][a-z]*[0-9]+[A-Z]*\b',  # Alphanumeric compounds
            ],
            'medical_abbreviations': [
                r'\b[A-Z]{2,6}\b(?:\s*[G0-9][0-9A-Z]*)?',  # 2-6 letter abbreviations with optional suffixes
                r'\bp\.[A-Z][0-9]+[A-Z]\b',  # Mutation patterns like p.G12C
            ],
            'dosage_patterns': [
                r'\d+\.?\d*\s*(?:mg|μg|ng|pg|IU|mL|L)(?:/m²|/kg|/day)?',
                r'\d+\s*x\s*(?:daily|per\s+day|\d+\s*times)',
                r'every\s+\d+\s+(?:days?|weeks?|hours?)',
                r'\d+/\d+\s*days?',
            ],
            'statistical_measures': [
                r'HR\s*(?:\(95%\s*CI[:\s]*\d+\.\d+[-–,;\s]*\d+\.\d+\)|\d+\.\d+)',
                r'OR\s*(?:\(95%\s*CI[:\s]*\d+\.\d+[-–,;\s]*\d+\.\d+\)|\d+\.\d+)',
                r'RR\s*(?:\(95%\s*CI[:\s]*\d+\.\d+[-–,;\s]*\d+\.\d+\)|\d+\.\d+)',
                r'p\s*[<>=≤≥]\s*\d+\.\d+',
                r'95%\s*CI[:\s]*\d+\.\d+[-–,;\s]*\d+\.\d+',
            ],
            'regulatory_bodies': [
                r'\b(?:EMA|FDA|NICE|SMC|G-BA|HAS|AIFA|TGA|PMDA|CADTH)\b',
                r'\b(?:European\s+Medicines\s+Agency|Federal\s+Joint\s+Committee)\b',
            ],
            'cross_references': [
                r'(?:Section|Chapter|Table|Figure|Annex|Appendix)\s+\d+(?:\.\d+)*',
                r'(?:resolution|decision|determination)\s+(?:of\s+)?\d{1,2}\s+\w+\s+\d{4}',
                r'Article\s+\d+(?:\.\d+)*',
                r'paragraph\s+\d+(?:\.\d+)*',
            ],
            'technical_terms': [
                r'\b(?:RECIST|ECOG|BCLC|Child-Pugh|mRECIST)\b',
                r'\b(?:progression-free\s+survival|overall\s+survival)\b',
                r'\b(?:hazard\s+ratio|odds\s+ratio|relative\s+risk)\b',
            ]
        }

        # Context patterns that indicate when medical terms should be translated
        self.translation_context_patterns = {
            'disease_names': [
                r'(?:cancer|carcinoma|tumor|tumour)(?:\s+of\s+the)?',
                r'(?:lung|liver|breast|prostate|colorectal)',
                r'(?:hepatocellular|non-small\s+cell|small\s+cell)',
            ],
            'anatomical_parts': [
                r'(?:lung|liver|kidney|brain|heart|stomach)',
                r'(?:chest|abdomen|pelvis|thorax)',
            ],
            'general_medical': [
                r'(?:treatment|therapy|medication|drug)',
                r'(?:patient|subject|participant)',
                r'(?:study|trial|investigation)',
            ]
        }

        # Table structure preservation patterns
        self.table_indicators = {
            'strong_patterns': [
                r'(?:Row|Column)\s+\d+[:\s]',
                r'Table\s+(?:Title|contains|shows)',
                r'\|\s*[^|\n]+\s*\|\s*[^|\n]+\s*\|',
                r'^\s*\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+',
            ],
            'cost_tables': [
                r'€\s*\d+(?:[.,]\d+)*',
                r'\d+(?:[.,]\d+)*\s*€',
                r'(?:cost|price|expense)\s*[:\s]',
            ],
            'clinical_tables': [
                r'(?:treatment|therapy|drug)\s+(?:arm|group)',
                r'(?:primary|secondary)\s+endpoint',
                r'(?:efficacy|safety)\s+(?:analysis|results)',
            ]
        }

        # Enhanced model configuration with medical focus
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
                'nllb_large': 'facebook/nllb-200-1.3B',
                'nllb_distilled': 'facebook/nllb-200-distilled-1.3B',
                'fallback': 'facebook/nllb-200-distilled-600M',
                'description': 'Large NLLB models for improved quality'
            },
            3: {
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

        # Enhanced quality thresholds with medical focus
        self.document_quality_thresholds = {
            'tier_1_to_2_threshold': 0.65,
            'minimum_acceptable_quality': 0.50,
            'medical_preservation_threshold': 0.75,
            'quantitative_data_threshold': 0.80,
            'regulatory_content_threshold': 0.85,
            'repetition_penalty_threshold': 0.80,
            'language_detection_threshold': 0.70,
        }

        # Medical-optimized chunking parameters
        self.translation_chunk_params = {
            'max_tokens': 350,  # Reduced for better medical concept preservation
            'overlap_ratio': 0.15,  # Increased for better context preservation
            'min_chunk_size': 75,
            'quality_threshold': 0.7,
            'medical_concept_boundary_bonus': 50,  # Extra tokens for medical concepts
        }

        # Batch processing parameters
        self.batch_params = {
            'max_batch_size': 3,  # Conservative for memory management
            'similar_language_batching': True,
            'max_concurrent_translations': 2,
        }

        # Enhanced artifact detection patterns
        self.translation_artifact_patterns = [
            r'(\d+\.\d+\.\d+\.)+\d+',
            r'(\d+\s*mg\s*){3,}',
            r'(\d+\s*%\s*){3,}',
            r'([!?.,:;-])\1{3,}',
            r'(\w+\s+)\1{3,}',
            r'((?:clinical trial|study|patient|treatment)\s+){3,}',
            r'(p[<=]\d+\.\d+\s*){3,}',
            r'(CI:\s*\d+\.\d+-\d+\.\d+\s*){3,}',
        ]

        # Medical exclusion patterns for artifact detection
        self.medical_exclusions = [
            r'dose-dose\s+(?:escalation|reduction)',
            r'first-line.*second-line.*third-line',
            r'pre-treatment.*post-treatment',
            r'primary.*secondary.*tertiary',
            r'grade\s+1.*grade\s+2.*grade\s+3',
            r'phase\s+I.*phase\s+II.*phase\s+III',
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

    def count_tokens(self, text: str) -> int:
        """
        Enhanced token counting that accounts for medical terminology density.
        Medical terms and statistical data typically require more tokens.
        """
        if not text:
            return 0
            
        # Base character-to-token ratio
        base_tokens = len(text) // 4
        
        # Add complexity bonus for medical content
        medical_density = 0
        for pattern_group in self.preservation_patterns.values():
            for pattern in pattern_group:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                medical_density += matches
        
        # Adjust token count based on medical complexity
        complexity_multiplier = 1 + (medical_density * 0.1)
        return int(base_tokens * complexity_multiplier)

    def detect_dynamic_preservation_terms(self, text: str) -> Dict[str, List[str]]:
        """
        Dynamically detect terms that should be preserved based on patterns.
        Returns categorized terms found in the text.
        """
        found_terms = {}
        
        for category, patterns in self.preservation_patterns.items():
            found_terms[category] = []
            for pattern in patterns:
                try:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    # Flatten nested matches and remove duplicates
                    if matches:
                        flat_matches = []
                        for match in matches:
                            if isinstance(match, tuple):
                                flat_matches.extend([m for m in match if m])
                            else:
                                flat_matches.append(match)
                        found_terms[category].extend(list(set(flat_matches)))
                except re.error:
                    continue
        
        return found_terms

    def should_translate_medical_term(self, term: str, context: str) -> bool:
        """
        Determine if a medical term should be translated based on context.
        Considers whether the term is a general medical concept vs. specific identifier.
        """
        term_lower = term.lower()
        context_lower = context.lower()
        
        # Never translate if it matches technical preservation patterns
        for pattern_group in ['drug_names', 'medical_abbreviations', 'statistical_measures']:
            if pattern_group in self.preservation_patterns:
                for pattern in self.preservation_patterns[pattern_group]:
                    if re.search(pattern, term, re.IGNORECASE):
                        return False
        
        # Consider translating if it's a general medical concept in translation context
        for category, patterns in self.translation_context_patterns.items():
            for pattern in patterns:
                if (re.search(pattern, term_lower) and 
                    any(re.search(ctx_pattern, context_lower) for ctx_pattern in patterns)):
                    return True
        
        return False

    def is_table_content(self, text: str) -> bool:
        """
        Enhanced table detection using multiple indicators.
        """
        if not text:
            return False
            
        # Check for strong table indicators
        strong_indicators = 0
        for pattern in self.table_indicators['strong_patterns']:
            if re.search(pattern, text, re.IGNORECASE):
                strong_indicators += 1
        
        if strong_indicators >= 2:
            return True
        
        # Check for specialized table types
        for table_type, patterns in self.table_indicators.items():
            if table_type == 'strong_patterns':
                continue
            matches = sum(1 for pattern in patterns if re.search(pattern, text, re.IGNORECASE))
            if matches >= 2:
                return True
        
        # Structural indicators
        line_count = len(text.split('\n'))
        pipe_density = text.count('|') / max(len(text), 1)
        numeric_density = len(re.findall(r'\d+(?:\.\d+)?', text)) / max(len(text.split()), 1)
        
        return (line_count >= 5 and pipe_density > 0.05) or numeric_density > 0.3

    def identify_semantic_boundaries(self, text: str) -> List[int]:
        """
        Identify positions where text can be safely split while preserving medical concepts.
        Returns list of character positions suitable for splitting.
        """
        boundaries = []
        
        # Medical concept boundaries (higher priority)
        medical_boundary_patterns = [
            r'\.\s+(?=(?:Study|Trial|Patient|Treatment|Drug|Therapy|Analysis)\s)',
            r'\.\s+(?=\d+\.\s)',  # Numbered sections
            r'\n\n•\s+',  # Bullet points
            r'\.\s+(?=The\s+(?:primary|secondary|main)\s)',  # Key sections
            r'\.\s+(?=In\s+(?:this|the)\s+(?:study|trial|analysis)\s)',
        ]
        
        # Standard boundaries (lower priority)
        standard_patterns = [
            r'\.\s+[A-Z]',  # Sentence boundaries
            r'\n\n',  # Paragraph breaks
            r'[;:]\s+',  # Semi-colon/colon breaks
        ]
        
        # Find medical boundaries first
        for pattern in medical_boundary_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                boundaries.append(match.start())
        
        # Add standard boundaries if not enough medical ones
        if len(boundaries) < 3:
            for pattern in standard_patterns:
                for match in re.finditer(pattern, text):
                    boundaries.append(match.start())
        
        return sorted(list(set(boundaries)))

    def adaptive_chunk_for_translation(self, text: str, max_tokens: int = None) -> List[Dict[str, Any]]:
        """
        Enhanced chunking that preserves medical concepts and regulatory language.
        """
        if max_tokens is None:
            max_tokens = self.translation_chunk_params['max_tokens']
            
        # Check for medical content and adjust token limit
        is_medical_heavy = any(
            len(re.findall(pattern, text, re.IGNORECASE)) > 2
            for pattern_group in self.preservation_patterns.values()
            for pattern in pattern_group[:2]  # Check first 2 patterns of each group
        )
        
        if is_medical_heavy:
            max_tokens += self.translation_chunk_params['medical_concept_boundary_bonus']
            
        if self.count_tokens(text) <= max_tokens:
            return [{'text': text, 'is_sub_chunk': False, 'chunk_index': 0, 'is_medical_heavy': is_medical_heavy}]
        
        print(f"      🔨 Medical-aware chunking needed (estimated {self.count_tokens(text)} tokens > {max_tokens})")
        
        # Use semantic boundaries for medical content
        boundaries = self.identify_semantic_boundaries(text)
        chunks = self._split_by_medical_boundaries(text, boundaries, max_tokens)
        
        # Add metadata to chunks
        chunk_data = []
        for i, chunk_text in enumerate(chunks):
            chunk_data.append({
                'text': chunk_text,
                'is_sub_chunk': len(chunks) > 1,
                'chunk_index': i,
                'total_sub_chunks': len(chunks),
                'is_medical_heavy': is_medical_heavy,
                'preservation_terms': self.detect_dynamic_preservation_terms(chunk_text)
            })
        
        print(f"      ✓ Created {len(chunks)} medical-aware translation sub-chunks")
        return chunk_data

    def _split_by_medical_boundaries(self, text: str, boundaries: List[int], max_tokens: int) -> List[str]:
        """
        Split text at medical boundaries while respecting token limits.
        """
        if not boundaries:
            return self._force_split_by_chars(text, max_tokens)
        
        chunks = []
        current_start = 0
        
        for boundary in boundaries:
            potential_chunk = text[current_start:boundary + 1].strip()
            
            if not potential_chunk:
                continue
                
            if self.count_tokens(potential_chunk) <= max_tokens:
                continue
            else:
                # Take chunk up to previous boundary or force split
                if current_start < boundary:
                    prev_chunk = text[current_start:boundary].strip()
                    if prev_chunk:
                        if self.count_tokens(prev_chunk) <= max_tokens:
                            chunks.append(prev_chunk)
                            current_start = boundary
                        else:
                            # Force split the oversized chunk
                            force_chunks = self._force_split_by_chars(prev_chunk, max_tokens)
                            chunks.extend(force_chunks)
                            current_start = boundary
        
        # Add remaining text
        remaining = text[current_start:].strip()
        if remaining:
            if self.count_tokens(remaining) <= max_tokens:
                chunks.append(remaining)
            else:
                force_chunks = self._force_split_by_chars(remaining, max_tokens)
                chunks.extend(force_chunks)
        
        return [chunk for chunk in chunks if chunk.strip()]

    def _force_split_by_chars(self, text: str, max_tokens: int) -> List[str]:
        """
        Force split text by character count while trying to preserve medical terms.
        """
        max_chars = max_tokens * 4
        chunks = []
        
        while text:
            if len(text) <= max_chars:
                chunks.append(text)
                break
            
            # Find a safe split point that doesn't break medical terms
            split_point = max_chars
            
            # Look for medical term boundaries
            for i in range(max_chars, max(max_chars // 2, 100), -1):
                if i >= len(text):
                    continue
                    
                char = text[i]
                if char.isspace():
                    # Check if we're not in the middle of a medical term
                    before_space = text[max(0, i-20):i]
                    after_space = text[i:min(len(text), i+20)]
                    
                    # Simple check for medical term patterns
                    if not (re.search(r'[A-Z]{2,}$', before_space) and 
                           re.search(r'^[A-Z0-9]', after_space.strip())):
                        split_point = i
                        break
            
            chunk = text[:split_point].strip()
            if chunk:
                chunks.append(chunk)
            
            text = text[split_point:].strip()
        
        return chunks

    def preserve_dynamic_terms(self, text: str) -> tuple:
        """
        Enhanced preservation using dynamic term detection and context analysis.
        """
        if not text or not isinstance(text, str):
            return text, {}
            
        preserved = {}
        modified_text = text
        preserved_count = 0
        
        # Get all terms to preserve from this text
        found_terms = self.detect_dynamic_preservation_terms(text)
        
        try:
            for category, terms in found_terms.items():
                for term in terms:
                    if not isinstance(term, str) or not term.strip():
                        continue
                    
                    # Check if this term should be preserved or translated
                    if category in ['cross_references', 'regulatory_bodies', 'statistical_measures']:
                        # Always preserve these categories
                        should_preserve = True
                    elif category in ['drug_names', 'medical_abbreviations']:
                        # Usually preserve, but check context
                        should_preserve = not self.should_translate_medical_term(term, text)
                    else:
                        # Context-dependent preservation
                        should_preserve = not self.should_translate_medical_term(term, text)
                    
                    if should_preserve:
                        placeholder = f"__PRESERVE_{category.upper()}_{preserved_count}__"
                        try:
                            # Use word boundaries for better matching
                            pattern = r'\b' + re.escape(term) + r'\b'
                            if re.search(pattern, modified_text, re.IGNORECASE):
                                modified_text = re.sub(pattern, placeholder, modified_text, flags=re.IGNORECASE)
                                preserved[placeholder] = term
                                preserved_count += 1
                        except re.error:
                            continue
                            
            return modified_text, preserved
        except (AttributeError, TypeError):
            return text, {}

    def restore_preserved_terms(self, text: str, preserved: dict) -> str:
        """
        Restore preserved terms after translation with additional cleanup.
        """
        if not preserved:
            return text
            
        for placeholder, term in preserved.items():
            text = text.replace(placeholder, term)
        
        # Clean up any spacing issues around restored terms
        text = re.sub(r'\s+([.,;:])', r'\1', text)  # Remove spaces before punctuation
        text = re.sub(r'([.,;:])\s+(\w)', r'\1 \2', text)  # Ensure space after punctuation
        
        return text

    def assess_medical_quality(self, original_text: str, translated_text: str) -> Dict[str, float]:
        """
        Enhanced quality assessment with medical-specific metrics.
        """
        if not translated_text.strip():
            return {
                'quantitative_preservation': 0.0,
                'regulatory_preservation': 0.0,
                'medical_coherence': 0.0,
                'cross_reference_preservation': 0.0
            }
        
        scores = {}
        
        # Quantitative data preservation
        orig_numbers = re.findall(r'\d+(?:\.\d+)?(?:\s*[%€$]|\s*mg|\s*CI)', original_text)
        trans_numbers = re.findall(r'\d+(?:\.\d+)?(?:\s*[%€$]|\s*mg|\s*CI)', translated_text)
        scores['quantitative_preservation'] = min(len(trans_numbers) / max(len(orig_numbers), 1), 1.0)
        
        # Regulatory language preservation
        orig_regulatory = self.detect_dynamic_preservation_terms(original_text).get('regulatory_bodies', [])
        trans_regulatory = self.detect_dynamic_preservation_terms(translated_text).get('regulatory_bodies', [])
        scores['regulatory_preservation'] = min(len(trans_regulatory) / max(len(orig_regulatory), 1), 1.0)
        
        # Cross-reference preservation
        orig_refs = self.detect_dynamic_preservation_terms(original_text).get('cross_references', [])
        trans_refs = self.detect_dynamic_preservation_terms(translated_text).get('cross_references', [])
        scores['cross_reference_preservation'] = min(len(trans_refs) / max(len(orig_refs), 1), 1.0)
        
        # Medical coherence assessment
        scores['medical_coherence'] = self.assess_medical_coherence(translated_text)
        
        return scores

    def assess_translation_quality(self, original_text: str, translated_text: str, language: str) -> Dict[str, float]:
        """
        Enhanced quality assessment incorporating medical-specific metrics.
        """
        if not translated_text.strip():
            return {'overall': 0.0, 'repetition': 0.0, 'coherence': 0.0, 'language': 0.0, 'medical': 0.0}
        
        # Basic quality metrics
        repetition_score = 1.0 - (1.0 if self.detect_repetition(translated_text) else 0.0)
        
        len_ratio = len(translated_text) / max(len(original_text), 1)
        length_score = 1.0 if 0.4 <= len_ratio <= 2.5 else 0.6 if 0.2 <= len_ratio <= 4.0 else 0.3
        
        try:
            detected_lang = detect(translated_text)
            language_score = 1.0 if detected_lang == 'en' else 0.2
        except:
            language_score = 0.5
        
        # Enhanced medical assessment
        medical_scores = self.assess_medical_quality(original_text, translated_text)
        
        # Table content handling
        table_penalty = 0.0
        if self.is_table_content(original_text):
            # More lenient scoring for table content
            if not self.is_table_content(translated_text):
                table_penalty = 0.2
        
        # Artifact detection
        artifact_count = self.count_artifacts(translated_text)
        word_count = len(translated_text.split())
        artifact_score = max(0.0, 1.0 - (artifact_count / max(word_count, 1)) * 5)
        
        # Weighted scoring with medical emphasis
        base_scores = {
            'repetition': repetition_score,
            'length': length_score,
            'language': language_score,
            'artifacts': artifact_score,
            'coherence': medical_scores['medical_coherence']
        }
        
        # Combine medical scores
        medical_score = (
            medical_scores['quantitative_preservation'] * 0.3 +
            medical_scores['regulatory_preservation'] * 0.25 +
            medical_scores['cross_reference_preservation'] * 0.25 +
            medical_scores['medical_coherence'] * 0.2
        ) - table_penalty
        
        base_scores['medical'] = max(0.0, medical_score)
        
        # Overall weighted score
        weights = {
            'repetition': 0.20,
            'length': 0.10,
            'language': 0.15,
            'medical': 0.40,  # Highest weight for medical content
            'artifacts': 0.10,
            'coherence': 0.05
        }
        
        overall_score = sum(weights[k] * base_scores[k] for k in weights.keys())
        base_scores['overall'] = overall_score
        
        return base_scores

    def assess_document_quality(self, translated_data: dict, original_data: dict, language: str) -> Dict[str, float]:
        """
        Enhanced document-level quality assessment with medical focus.
        """
        if 'chunks' not in translated_data or 'chunks' not in original_data:
            return {'overall': 0.0, 'chunk_count': 0}
        
        chunk_qualities = []
        total_original_text = ""
        total_translated_text = ""
        table_chunks = 0
        regular_chunks = 0
        
        for orig_chunk, trans_chunk in zip(original_data['chunks'], translated_data['chunks']):
            # Track content types
            is_table = False
            
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
                    is_table = self.is_table_content(orig_chunk['text'])
                    text_quality = self.assess_translation_quality(
                        orig_chunk['text'], trans_chunk['text'], language
                    )
                    chunk_qualities.append(text_quality)
                    total_original_text += " " + orig_chunk['text']
                    total_translated_text += " " + trans_chunk['text']
                    
                    if is_table:
                        table_chunks += 1
                    else:
                        regular_chunks += 1
        
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
        
        # Combine assessments with medical weighting
        final_scores = {}
        for metric in metrics:
            chunk_score = aggregate_scores.get(metric, 0.0)
            doc_score = doc_level_quality.get(metric, 0.0)
            
            # Weight chunk-level more heavily for medical content
            if metric == 'medical':
                final_scores[metric] = (chunk_score * 0.8) + (doc_score * 0.2)
            else:
                final_scores[metric] = (chunk_score * 0.7) + (doc_score * 0.3)
        
        final_scores.update({
            'chunk_count': len(chunk_qualities),
            'table_chunks': table_chunks,
            'regular_chunks': regular_chunks,
            'avg_chunk_quality': aggregate_scores.get('overall', 0.0),
            'doc_level_quality': doc_level_quality.get('overall', 0.0)
        })
        
        return final_scores

    def should_retranslate_with_higher_tier(self, quality_scores: Dict[str, float]) -> bool:
        """
        Enhanced retranslation decision with medical-specific thresholds.
        """
        overall_quality = quality_scores.get('overall', 0.0)
        medical_score = quality_scores.get('medical', 0.0)
        repetition_score = quality_scores.get('repetition', 0.0)
        
        # Primary threshold check
        if overall_quality < self.document_quality_thresholds['tier_1_to_2_threshold']:
            print(f"    📉 Overall quality {overall_quality:.3f} below threshold")
            return True
        
        # Medical-specific checks
        if medical_score < self.document_quality_thresholds['medical_preservation_threshold']:
            print(f"    🏥 Medical preservation {medical_score:.3f} below threshold")
            return True
            
        if repetition_score < self.document_quality_thresholds['repetition_penalty_threshold']:
            print(f"    🔄 Repetition issues detected")
            return True
        
        return False

    def compare_translation_quality(self, quality_1: Dict[str, float], quality_2: Dict[str, float]) -> int:
        """
        Enhanced quality comparison with medical priority.
        """
        # Primary: overall quality
        if abs(quality_1['overall'] - quality_2['overall']) > 0.05:
            return 1 if quality_1['overall'] > quality_2['overall'] else 2
        
        # Secondary: medical preservation (critical)
        med_diff = quality_1.get('medical', 0) - quality_2.get('medical', 0)
        if abs(med_diff) > 0.08:
            return 1 if med_diff > 0 else 2
        
        # Tertiary: repetition issues
        rep_diff = quality_1.get('repetition', 0) - quality_2.get('repetition', 0)
        if abs(rep_diff) > 0.1:
            return 1 if rep_diff > 0 else 2
        
        return 1

    def assess_medical_coherence(self, text: str) -> float:
        """
        Enhanced medical coherence assessment.
        """
        if not text:
            return 0.0
        
        penalties = 0
        
        # Check for medical nonsense patterns
        nonsense_patterns = [
            r'(\b(?:patient|treatment|study|drug)\b\s+){4,}',
            r'(\d+\.\d+\s+){4,}(?!CI|HR|OR)',  # Repeated numbers not in medical context
            r'(\b(?:mg|ml|kg)\b\s+){3,}',
        ]
        
        for pattern in nonsense_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                penalties += 1
        
        # Check for preserved medical structure
        medical_structure_bonus = 0
        structure_patterns = [
            r'\b(?:HR|OR|RR)\s*[=:]\s*\d+\.\d+',
            r'95%\s*CI[:\s]*\d+\.\d+[-–,;\s]*\d+\.\d+',
            r'p\s*[<>=≤≥]\s*\d+\.\d+',
        ]
        
        for pattern in structure_patterns:
            if re.search(pattern, text):
                medical_structure_bonus += 0.1
        
        base_score = max(0.0, 1.0 - (penalties * 0.25))
        return min(1.0, base_score + medical_structure_bonus)

    def count_artifacts(self, text: str) -> int:
        """
        Enhanced artifact counting with medical context awareness.
        """
        if not text or not isinstance(text, str):
            return 0
            
        # Check medical exclusions first
        try:
            for exclusion_pattern in self.medical_exclusions:
                if isinstance(exclusion_pattern, str) and re.search(exclusion_pattern, text, re.IGNORECASE):
                    return 0
        except (TypeError, re.error, AttributeError):
            pass
        
        artifact_count = 0
        try:
            for pattern in self.translation_artifact_patterns:
                if isinstance(pattern, str):
                    try:
                        matches = re.findall(pattern, text)
                        artifact_count += len(matches)
                    except (TypeError, re.error):
                        continue
        except (AttributeError, TypeError):
            pass
        
        return artifact_count

    def check_model_availability(self, model_name: str) -> bool:
        """Check if a model is available and can be loaded."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            return True
        except Exception as e:
            print(f"    ✗ Model {model_name} not available: {str(e)[:50]}")
            return False

    def get_available_models_for_tier(self, tier: int, language: str) -> List[str]:
        """Get list of available models for a tier and language."""
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
                if 'nllb_distilled' in self.model_tiers[2]:
                    model_name = self.model_tiers[2]['nllb_distilled']
                    if self.check_model_availability(model_name):
                        available_models.append(model_name)
                
                if 'nllb_large' in self.model_tiers[2]:
                    model_name = self.model_tiers[2]['nllb_large']
                    if self.check_model_availability(model_name):
                        available_models.append(model_name)
                
                if 'fallback' in self.model_tiers[2]:
                    fallback_model = self.model_tiers[2]['fallback']
                    if self.check_model_availability(fallback_model):
                        available_models.append(fallback_model)
                        
        elif tier == 3:
            if language in self.nllb_lang_mapping:
                if 'nllb_xl' in self.model_tiers[3]:
                    model_name = self.model_tiers[3]['nllb_xl']
                    if self.check_model_availability(model_name):
                        available_models.append(model_name)
                
                if 'm2m100_large' in self.model_tiers[3]:
                    model_name = self.model_tiers[3]['m2m100_large']
                    if self.check_model_availability(model_name):
                        available_models.append(model_name)
                
                if 'fallback' in self.model_tiers[3]:
                    fallback_model = self.model_tiers[3]['fallback']
                    if self.check_model_availability(fallback_model):
                        available_models.append(fallback_model)
        
        return available_models

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
        """Load NLLB model with medical-optimized parameters."""
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
        """Load the appropriate translator for the given language and tier."""
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

    def load_translator_for_language(self, language: str, target_tier: int = 1):
        """Load the appropriate translator for the given language."""
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

    def clean_translation_artifacts(self, text: str) -> str:
        """Clean translation artifacts while preserving medical terminology."""
        if not text or not isinstance(text, str):
            return text
            
        try:
            for exclusion_pattern in self.medical_exclusions:
                if isinstance(exclusion_pattern, str) and re.search(exclusion_pattern, text, re.IGNORECASE):
                    return text
        except (TypeError, re.error, AttributeError):
            pass
        
        cleaned_text = text
        try:
            for pattern in self.translation_artifact_patterns:
                if isinstance(pattern, str):
                    try:
                        cleaned_text = re.sub(pattern, r'\1', cleaned_text)
                    except (TypeError, re.error):
                        continue
        except (AttributeError, TypeError):
            pass
        
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

    def detect_repetition(self, text: str, max_repeat_ratio: float = 0.3) -> bool:
        """Detect if text has excessive repetition."""
        if not text or not isinstance(text, str) or len(text) < 20:
            return False
        
        try:
            words = text.lower().split()
            if len(words) < 10:
                return False
            
            for n in [2, 3, 4]:
                if len(words) < n:
                    continue
                    
                ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
                if not ngrams:
                    continue
                    
                ngram_counts = {}
                for ngram in ngrams:
                    ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
                
                if ngram_counts:
                    max_count = max(ngram_counts.values())
                    repeat_ratio = max_count / len(ngrams)
                    
                    if repeat_ratio > max_repeat_ratio:
                        return True
            
            return False
            
        except (TypeError, AttributeError):
            return False

    def get_generation_params(self, tier: int = 1) -> dict:
        """Get generation parameters optimized for each tier."""
        base_params = {
            'max_length': 400,
            'truncation': True,
            'no_repeat_ngram_size': 4,  # Increased for medical content
            'repetition_penalty': 1.25,  # Increased for medical accuracy
            'do_sample': False,
            'num_beams': 3,
        }
        
        if tier >= 2:
            base_params.update({
                'num_beams': 4,
                'length_penalty': 1.1,
                'repetition_penalty': 1.3,
                'no_repeat_ngram_size': 5,
            })
        
        return base_params

    def translate_single_chunk(self, text: str, translator, tier: int = 1) -> str:
        """
        Translate a single chunk with enhanced medical preservation.
        """
        if not text.strip() or not translator:
            return text

        try:
            # Enhanced chunking with medical awareness
            if self.should_rechunk_for_translation(text) or self.count_tokens(text) > self.translation_chunk_params['max_tokens']:
                print(f"      🔨 Medical-aware chunking needed")
                
                translation_chunks = self.adaptive_chunk_for_translation(text, self.translation_chunk_params['max_tokens'])
                
                translated_sub_chunks = []
                for chunk_data in translation_chunks:
                    sub_chunk_text = chunk_data['text']
                    
                    # Use enhanced preservation
                    protected_text, preserved_terms = self.preserve_dynamic_terms(sub_chunk_text)
                    
                    gen_params = self.get_generation_params(tier)
                    
                    # Special handling for table content
                    if self.is_table_content(sub_chunk_text):
                        gen_params['repetition_penalty'] = 1.1  # More lenient for tables
                        gen_params['length_penalty'] = 0.9
                    
                    if hasattr(translator, 'model'):
                        pipeline_params = {k: v for k, v in gen_params.items() 
                                         if k in ['max_length', 'truncation', 'no_repeat_ngram_size', 
                                                 'repetition_penalty', 'do_sample', 'num_beams']}
                        result = translator(protected_text, **pipeline_params)
                    else:
                        result = translator(protected_text, generation_params=gen_params)
                    
                    translated_sub_text = result[0]['translation_text']
                    final_sub_text = self.restore_preserved_terms(translated_sub_text, preserved_terms)
                    translated_sub_chunks.append(final_sub_text)
                
                final_text = self.merge_translated_chunks(translated_sub_chunks, text)
                print(f"      ✓ Merged {len(translated_sub_chunks)} medical-aware sub-chunks")
                
            else:
                # Standard single-chunk translation
                protected_text, preserved_terms = self.preserve_dynamic_terms(text)
                gen_params = self.get_generation_params(tier)
                
                if self.is_table_content(text):
                    gen_params['repetition_penalty'] = 1.1
                    gen_params['length_penalty'] = 0.9
                
                if hasattr(translator, 'model'):
                    pipeline_params = {k: v for k, v in gen_params.items() 
                                     if k in ['max_length', 'truncation', 'no_repeat_ngram_size', 
                                             'repetition_penalty', 'do_sample', 'num_beams']}
                    result = translator(protected_text, **pipeline_params)
                else:
                    result = translator(protected_text, generation_params=gen_params)
                
                translated_text = result[0]['translation_text']
                final_text = self.restore_preserved_terms(translated_text, preserved_terms)
            
            cleaned_text = self.clean_translation_artifacts(final_text)
            return cleaned_text

        except Exception as e:
            print(f"      Translation error: {str(e)[:50]}")
            return text

    def should_rechunk_for_translation(self, chunk: str, quality_threshold: float = None) -> bool:
        """
        Enhanced rechunking decision with medical context awareness.
        """
        if quality_threshold is None:
            quality_threshold = self.translation_chunk_params['quality_threshold']
            
        complexity_indicators = {
            'token_count': self.count_tokens(chunk),
            'medical_density': sum(len(terms) for terms in self.detect_dynamic_preservation_terms(chunk).values()),
            'table_content': 1.0 if self.is_table_content(chunk) else 0.0,
            'cross_references': len(self.detect_dynamic_preservation_terms(chunk).get('cross_references', [])),
            'statistical_measures': len(self.detect_dynamic_preservation_terms(chunk).get('statistical_measures', []))
        }
        
        complexity_score = self._calculate_complexity_score(complexity_indicators)
        return complexity_score > quality_threshold

    def _calculate_complexity_score(self, indicators: Dict[str, float]) -> float:
        """Calculate complexity score with medical weighting."""
        weights = {
            'token_count': 0.25,
            'medical_density': 0.30,
            'table_content': 0.20,
            'cross_references': 0.15,
            'statistical_measures': 0.10
        }
        
        normalized = {}
        normalized['token_count'] = min(1.0, indicators['token_count'] / 400)
        normalized['medical_density'] = min(1.0, indicators['medical_density'] / 10)
        normalized['table_content'] = indicators['table_content']
        normalized['cross_references'] = min(1.0, indicators['cross_references'] / 5)
        normalized['statistical_measures'] = min(1.0, indicators['statistical_measures'] / 5)
        
        score = sum(weights[key] * normalized[key] for key in weights.keys())
        return score

    def merge_translated_chunks(self, translated_sub_chunks: List[str], original_chunk_text: str) -> str:
        """
        Enhanced merging with medical context preservation.
        """
        if len(translated_sub_chunks) == 1:
            return translated_sub_chunks[0]
        
        merged = []
        
        for i, sub_chunk in enumerate(translated_sub_chunks):
            sub_chunk = sub_chunk.strip()
            if not sub_chunk:
                continue
                
            if merged:
                last_chunk = merged[-1]
                # Medical-aware spacing
                if (re.search(r'[.!?]\s*$', last_chunk) or 
                    re.search(r'^[A-Z]', sub_chunk) or
                    re.search(r'^\d+\.', sub_chunk)):  # Numbered items
                    if '\n\n' in original_chunk_text:
                        merged.append('\n\n')
                    else:
                        merged.append(' ')
                else:
                    merged.append(' ')
            
            merged.append(sub_chunk)
        
        result = ''.join(merged)
        result = re.sub(r'\n\n\n+', '\n\n', result)
        result = re.sub(r'  +', ' ', result)
        
        return result.strip()

    def process_batch_files(self, file_batch: List[Tuple[str, str]]) -> None:
        """
        Simple batch processing for files with similar languages.
        """
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

    def translate_document_with_tier(self, data: dict, language: str, tier: int) -> Tuple[dict, Dict[str, float], Dict[str, Any]]:
        """
        Enhanced document translation with medical optimization.
        """
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
        
        print(f"    Processing {total_chunks} chunks with enhanced medical awareness...")
        
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
            
            # Process text with table awareness
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
        
        # Enhanced quality assessment
        quality_scores = self.assess_document_quality(translated_data, data, language)
        
        print(f"    📊 Tier {tier} Quality Assessment:")
        print(f"      Overall: {quality_scores['overall']:.3f}")
        print(f"      Medical: {quality_scores.get('medical', 0):.3f}")
        print(f"      Repetition: {quality_scores.get('repetition', 0):.3f}")
        print(f"      Language: {quality_scores.get('language', 0):.3f}")
        
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
        """
        Enhanced JSON file processing with medical optimization.
        """
        file_name = os.path.basename(input_path)
        print(f"\n📄 Processing: {file_name}")
        
        self.processing_start_time = datetime.now()

        try:
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ✗ Error loading JSON: {e}")
            return

        # Enhanced text extraction for language detection
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

        # Enhanced metadata tracking
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
            "translation_strategy": "hierarchical_medical_optimized",
            "medical_preservation_used": "dynamic_pattern_based",
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

        # Check model availability
        tier_1_available = len(self.get_available_models_for_tier(1, document_language)) > 0
        tier_2_available = len(self.get_available_models_for_tier(2, document_language)) > 0
        
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

        # Enhanced translation with medical optimization
        if tier_1_available:
            print(f"  🎯 Starting with Tier 1 translation (medical-optimized)")
            tier_1_translation, tier_1_quality, tier_1_metadata = self.translate_document_with_tier(
                data, document_language, 1
            )
            translation_metadata["tier_attempts"].append(tier_1_metadata)
            
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
                
                if tier_2_available:
                    self.clear_translator()
                    tier_2_translation, tier_2_quality, tier_2_metadata = self.translate_document_with_tier(
                        data, document_language, 2
                    )
                    translation_metadata["tier_attempts"].append(tier_2_metadata)
                    
                    if self.compare_translation_quality(tier_2_quality, tier_1_quality) == 1:
                        print(f"  🏆 Tier 2 translation is better")
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
                        print(f"  🥈 Tier 1 translation is still better")
                        final_translation = tier_1_translation
                        final_quality = tier_1_quality
                        translation_metadata.update({
                            "translation_decision": "tier_1_better_than_tier_2",
                            "final_tier_used": 1,
                            "final_model_used": tier_1_metadata["model_name"],
                            "retranslation_occurred": True
                        })
                else:
                    print(f"  ⚠️  Tier 2 not available")
                    final_translation = tier_1_translation
                    final_quality = tier_1_quality
                    translation_metadata.update({
                        "translation_decision": "tier_1_only_tier_2_unavailable",
                        "final_tier_used": 1,
                        "final_model_used": tier_1_metadata["model_name"],
                        "retranslation_occurred": False
                    })
        else:
            print(f"  🎯 Tier 1 not available, using Tier 2")
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

    def translate_documents(self):
        """
        Enhanced main translation method with simple batch processing.
        """
        print("🚀 Starting enhanced medical document translation...")
        print("📋 Enhanced Strategy:")
        print("   • Dynamic medical term preservation using pattern detection")
        print("   • Medical-aware semantic chunking for complex content")
        print("   • Enhanced table content handling")
        print("   • Regulatory language and cross-reference preservation")
        print("   • Medical-specific quality assessment")
        print("   • Simple batch processing for efficiency")

        os.makedirs(self.output_dir, exist_ok=True)

        # Find and group files
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

        # Simple batching by similarity
        if self.batch_params['similar_language_batching'] and total_files > self.batch_params['max_batch_size']:
            # Group files by directory (assuming similar content)
            file_groups = {}
            for input_path, output_path in json_files:
                dir_key = os.path.dirname(input_path)
                if dir_key not in file_groups:
                    file_groups[dir_key] = []
                file_groups[dir_key].append((input_path, output_path))
            
            # Process groups in batches
            for group_dir, group_files in file_groups.items():
                if len(group_files) <= self.batch_params['max_batch_size']:
                    print(f"\n📦 Processing batch from {os.path.basename(group_dir)}")
                    self.process_batch_files(group_files)
                else:
                    # Split large groups
                    for i in range(0, len(group_files), self.batch_params['max_batch_size']):
                        batch = group_files[i:i + self.batch_params['max_batch_size']]
                        print(f"\n📦 Processing batch {i//self.batch_params['max_batch_size'] + 1} from {os.path.basename(group_dir)}")
                        self.process_batch_files(batch)
        else:
            # Process files individually
            for i, (input_path, output_path) in enumerate(json_files, 1):
                print(f"\n[{i}/{total_files}]", end=" ")
                try:
                    self.process_json_file(input_path, output_path)
                except Exception as e:
                    print(f"  ✗ Error processing file: {str(e)[:100]}")
                    try:
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        shutil.copy(input_path, output_path)
                        print(f"  📋 Copied original file instead")
                    except Exception as copy_error:
                        print(f"  ✗ Failed to copy original: {copy_error}")

        # Enhanced summary
        print(f"\n🎉 Enhanced Medical Translation Complete!")
        print(f"📊 Summary:")
        print(f"   • Total files processed: {total_files}")
        print(f"   • English chunks preserved: {self.english_chunks_preserved}")
        print(f"   • Chunks translated: {self.chunks_translated}")
        print(f"   • Dynamic medical term preservation applied")
        print(f"   • Medical-aware chunking used for complex content")
        print(f"   • Enhanced quality assessment with medical metrics")
        print(f"   • Table content and regulatory language specially handled")
