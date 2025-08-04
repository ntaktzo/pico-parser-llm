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
    Enhanced Translator class for medical documents with improved chunk management,
    robust quality control, and content preservation focus.
    """

    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.english_chunks_preserved = 0
        self.chunks_translated = 0
        self.total_translation_start_time = None
        self.processing_start_time = None

        # Critical medical terms that must be preserved (never translated)
        self.critical_terms = {
            'drug_names': [
                'sotorasib', 'lumykras', 'pembrolizumab', 'keytruda', 'atezolizumab',
                'nivolumab', 'durvalumab', 'ipilimumab', 'bevacizumab', 'cetuximab',
                'panitumumab', 'trastuzumab', 'pertuzumab', 'ramucirumab',
                'nintedanib', 'osimertinib', 'erlotinib', 'gefitinib', 'afatinib',
                'crizotinib', 'alectinib', 'brigatinib', 'lorlatinib',
                'pemetrexed', 'docetaxel', 'paclitaxel', 'carboplatin', 'cisplatin',
                'gemcitabine', 'vinorelbine', 'etoposide'
            ],
            'abbreviations': [
                'NSCLC', 'NDRP', 'SCLC', 'EGFR', 'ALK', 'ROS1', 'KRAS', 'BRAF',
                'PD-L1', 'TMB', 'MSI', 'MMR', 'HER2', 'MET', 'RET', 'NTRK',
                'PFS', 'OS', 'ORR', 'DCR', 'DOR', 'TTR', 'TTP', 'QoL',
                'ECOG', 'KPS', 'PS', 'CR', 'PR', 'SD', 'PD',
                'AE', 'SAE', 'TEAE', 'CTCAE', 'NCI', 'WHO', 'RECIST',
                'ITT', 'PP', 'mITT', 'FAS', 'SS'
            ],
            'regulatory': [
                'FDA', 'EMA', 'NICE', 'SMC', 'G-BA', 'HAS', 'AIFA', 'TGA', 'PMDA',
                'CADTH', 'IQWIG', 'HTA', 'QALY', 'ICER', 'ICUR'
            ]
        }

        # Model tiers with enhanced parameter sets
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
                'description': 'Fast Helsinki models with NLLB fallback - primary choice'
            },
            2: {
                'nllb_xl': 'facebook/nllb-200-3.3B',
                'nllb_large': 'facebook/nllb-200-1.3B',
                'fallback': 'facebook/nllb-200-distilled-1.3B',
                'description': 'High quality models - used when Tier 1 quality insufficient'
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

        # Enhanced quality thresholds with stricter content preservation requirements
        self.quality_thresholds = {
            'tier_1_acceptable': 0.70,  # Raised threshold for accepting Tier 1
            'minimum_acceptable': 0.60,  # Raised minimum threshold
            'excellent_quality': 0.85,   # High quality threshold
            'content_loss_critical': 0.40,  # Critical content loss threshold
        }

        # Enhanced chunking parameters with adaptive sizing
        self.chunk_params = {
            'base_max_chars': 1800,  # Reduced base size for better model handling
            'overlap_chars': 200,    # Increased overlap for context preservation
            'min_chunk_chars': 200,  # Reduced minimum to avoid forcing large chunks
            'sentence_boundary_window': 400,  # Window for finding sentence boundaries
            'paragraph_priority_window': 200,  # Priority window for paragraph breaks
        }

        # Model-specific parameters for optimal translation
        self.model_parameters = {
            'helsinki': {
                'max_input_length': 400,  # Conservative for Helsinki models
                'max_output_length': 600,  # Allow longer outputs to prevent truncation
                'generation_params': {
                    'max_length': 600,
                    'num_beams': 3,
                    'length_penalty': 1.1,
                    'do_sample': False,
                    'no_repeat_ngram_size': 2,
                    'repetition_penalty': 1.05,
                }
            },
            'nllb': {
                'max_input_length': 800,  # NLLB can handle longer inputs
                'max_output_length': 1000,  # Allow even longer outputs
                'generation_params': {
                    'max_length': 1000,
                    'num_beams': 4,
                    'length_penalty': 1.0,
                    'do_sample': False,
                    'no_repeat_ngram_size': 3,
                    'repetition_penalty': 1.1,
                }
            }
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
        self.current_model_type = None

    def get_critical_terms_from_text(self, text: str) -> List[str]:
        """Extract critical terms that should be preserved in translation."""
        if not text:
            return []

        found_terms = []
        text_lower = text.lower()

        # Combine all critical terms
        all_terms = []
        for category, terms in self.critical_terms.items():
            all_terms.extend(terms)

        # Sort by length (longest first) to avoid partial matches
        all_terms.sort(key=len, reverse=True)

        for term in all_terms:
            term_lower = term.lower()
            if term_lower in text_lower:
                # Find actual case-sensitive matches
                pattern = r'\b' + re.escape(term) + r'\b'
                matches = re.findall(pattern, text, re.IGNORECASE)
                found_terms.extend(matches)

        return list(set(found_terms))  # Remove duplicates

    def count_tokens_accurately(self, text: str) -> int:
        """Estimate token count for text with improved accuracy."""
        if not text:
            return 0
        # More accurate estimation considering medical text complexity
        words = len(text.split())
        chars = len(text)
        # Medical texts tend to have longer tokens due to technical terms
        estimated_tokens = min(words * 1.3, chars // 3.5)
        return int(estimated_tokens)

    def find_optimal_split_point(self, text: str, max_pos: int) -> int:
        """Find the optimal split point prioritizing sentence and paragraph boundaries."""
        if max_pos >= len(text):
            return len(text)
        
        sentence_window = min(self.chunk_params['sentence_boundary_window'], max_pos // 2)
        paragraph_window = min(self.chunk_params['paragraph_priority_window'], max_pos // 3)
        
        # First priority: paragraph breaks
        for i in range(max_pos, max(max_pos - paragraph_window, 0), -1):
            if i < len(text) and text[i:i+2] == '\n\n':
                return i + 2
            elif i < len(text) and text[i] == '\n' and (i == 0 or text[i-1] == '\n'):
                return i + 1
        
        # Second priority: sentence endings
        for i in range(max_pos, max(max_pos - sentence_window, 0), -1):
            if i >= len(text):
                continue
            char = text[i]
            if char in '.!?' and i + 1 < len(text):
                next_char = text[i + 1]
                # Better sentence boundary detection
                if next_char.isspace() and (i + 2 >= len(text) or text[i + 2].isupper() or text[i + 2].isspace()):
                    return i + 1
        
        # Third priority: clause boundaries
        clause_markers = [';', ':', ',']
        for marker in clause_markers:
            for i in range(max_pos, max(max_pos - sentence_window // 2, 0), -1):
                if i < len(text) and text[i] == marker and i + 1 < len(text) and text[i + 1].isspace():
                    return i + 1
        
        # Final fallback: word boundaries
        for i in range(max_pos, max_pos // 2, -1):
            if i < len(text) and text[i].isspace():
                return i
        
        return max_pos

    def split_text_into_chunks(self, text: str) -> List[Dict[str, Any]]:
        """Enhanced text chunking with structure preservation and overlap management."""
        if not text:
            return []

        max_chars = self.chunk_params['base_max_chars']
        overlap_chars = self.chunk_params['overlap_chars']
        min_chunk_chars = self.chunk_params['min_chunk_chars']

        if len(text) <= max_chars:
            return [{'text': text, 'start_pos': 0, 'end_pos': len(text), 'overlap_start': False, 'overlap_end': False}]

        chunks = []
        start_pos = 0
        chunk_index = 0

        while start_pos < len(text):
            # Calculate end position for this chunk
            remaining_text = text[start_pos:]
            if len(remaining_text) <= max_chars:
                # Last chunk
                chunk_text = remaining_text
                end_pos = len(text)
            else:
                # Find optimal split point
                optimal_split = self.find_optimal_split_point(remaining_text, max_chars)
                chunk_text = remaining_text[:optimal_split]
                end_pos = start_pos + optimal_split

            # Skip if chunk is too small (except for last chunk)
            if len(chunk_text.strip()) < min_chunk_chars and end_pos < len(text):
                start_pos = end_pos
                continue

            # Add overlap context for better continuity (except first chunk)
            overlap_start = chunk_index > 0
            overlap_context_start = ""
            if overlap_start and start_pos > 0:
                overlap_start_pos = max(0, start_pos - overlap_chars)
                overlap_context_start = text[overlap_start_pos:start_pos]
                if overlap_context_start.strip():
                    chunk_text = overlap_context_start + " " + chunk_text

            chunk_info = {
                'text': chunk_text.strip(),
                'start_pos': start_pos,
                'end_pos': end_pos,
                'overlap_start': overlap_start,
                'overlap_end': False,
                'chunk_index': chunk_index
            }

            chunks.append(chunk_info)
            start_pos = end_pos
            chunk_index += 1

        return [chunk for chunk in chunks if chunk['text'].strip()]

    def adaptive_chunk_for_translation(self, text: str) -> List[Dict[str, Any]]:
        """Enhanced adaptive chunking with model-aware sizing."""
        if not text:
            return []

        # Estimate tokens and determine if chunking is needed
        estimated_tokens = self.count_tokens_accurately(text)
        model_type = self.current_model_type or 'helsinki'
        max_input_tokens = self.model_parameters[model_type]['max_input_length']

        if estimated_tokens <= max_input_tokens and len(text) <= self.chunk_params['base_max_chars']:
            return [{'text': text, 'start_pos': 0, 'end_pos': len(text), 'overlap_start': False, 'overlap_end': False, 'chunk_index': 0}]

        print(f"      🔨 Enhanced chunking needed ({len(text)} chars, ~{estimated_tokens} tokens)")
        chunks = self.split_text_into_chunks(text)
        print(f"      ✓ Created {len(chunks)} structured chunks with overlap management")
        return chunks

    def reassemble_translated_chunks(self, chunk_results: List[Dict[str, Any]]) -> str:
        """Enhanced chunk reassembly with structure preservation and overlap handling."""
        if not chunk_results:
            return ""
        
        if len(chunk_results) == 1:
            return chunk_results[0]['translated_text']
        
        assembled_parts = []
        
        for i, chunk_result in enumerate(chunk_results):
            translated_text = chunk_result['translated_text']
            
            if i == 0:
                # First chunk - use as is
                assembled_parts.append(translated_text)
            else:
                # For subsequent chunks, handle overlap
                prev_chunk = chunk_results[i-1]
                
                # If this chunk had overlap context, try to remove redundancy
                if chunk_result.get('overlap_start', False):
                    # Simple redundancy removal - look for repeated phrases at boundaries
                    words_current = translated_text.split()
                    words_prev = prev_chunk['translated_text'].split()
                    
                    # Find potential overlap by comparing last words of previous with first words of current
                    max_overlap_check = min(10, len(words_prev), len(words_current))
                    overlap_found = 0
                    
                    for j in range(1, max_overlap_check + 1):
                        if words_prev[-j:] == words_current[:j]:
                            overlap_found = j
                    
                    if overlap_found > 0:
                        # Remove overlapping words from current chunk
                        cleaned_text = ' '.join(words_current[overlap_found:])
                        assembled_parts.append(cleaned_text)
                    else:
                        # No clear overlap found, add with space separation
                        assembled_parts.append(translated_text)
                else:
                    # No overlap context, add with proper separation
                    assembled_parts.append(translated_text)
        
        # Join with appropriate spacing, preserving paragraph structure
        final_text = ""
        for i, part in enumerate(assembled_parts):
            if i == 0:
                final_text = part
            else:
                # Determine appropriate separator
                prev_ends_sentence = final_text.rstrip().endswith(('.', '!', '?', ':'))
                current_starts_upper = part.lstrip() and part.lstrip()[0].isupper()
                
                if prev_ends_sentence and current_starts_upper:
                    # Likely sentence boundary
                    final_text += " " + part
                elif '\n' in part[:5] or final_text.endswith('\n'):
                    # Paragraph or line break context
                    final_text += "\n" + part.lstrip()
                else:
                    # Default: space separation
                    final_text += " " + part
        
        return final_text.strip()

    def validate_translation_completeness(self, original_text: str, translated_text: str) -> Dict[str, Any]:
        """Validate that translation preserves essential content without critical loss."""
        validation_results = {
            'is_complete': True,
            'issues_found': [],
            'content_loss_ratio': 0.0,
            'critical_elements_preserved': True
        }
        
        if not original_text or not translated_text:
            validation_results['is_complete'] = False
            validation_results['issues_found'].append('Empty translation result')
            return validation_results
        
        # Check for severe length discrepancy
        length_ratio = len(translated_text) / len(original_text)
        validation_results['content_loss_ratio'] = max(0, 1 - length_ratio)
        
        if length_ratio < self.quality_thresholds['content_loss_critical']:
            validation_results['is_complete'] = False
            validation_results['issues_found'].append(f'Severe content loss: {length_ratio:.2f} length ratio')
        
        # Check for preserved critical elements
        original_numbers = re.findall(r'\d+(?:\.\d+)?%?', original_text)
        translated_numbers = re.findall(r'\d+(?:\.\d+)?%?', translated_text)
        
        if original_numbers and len(translated_numbers) < len(original_numbers) * 0.7:
            validation_results['critical_elements_preserved'] = False
            validation_results['issues_found'].append('Numerical data loss detected')
        
        # Check for repetitive patterns (sign of model errors)
        words = translated_text.split()
        if len(words) > 10:
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            max_freq = max(word_freq.values())
            if max_freq > len(words) * 0.15:  # Single word appears >15% of the time
                validation_results['is_complete'] = False
                validation_results['issues_found'].append('Repetitive translation pattern detected')
        
        return validation_results

    def is_table_content(self, text: str) -> bool:
        """Check if text contains table content."""
        if not text:
            return False

        for pattern in self.table_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        line_count = len(text.split('\n'))
        pipe_density = text.count('|') / max(len(text), 1)
        numeric_density = len(re.findall(r'\d+(?:\.\d+)?', text)) / max(len(text.split()), 1)

        return (line_count >= 5 and pipe_density > 0.05) or numeric_density > 0.3

    def assess_translation_quality(self, original_text: str, translated_text: str, language: str) -> Dict[str, float]:
        """
        Enhanced quality assessment with stricter content preservation requirements.
        Components: English Quality (35%), Medical Preservation (30%), Length Appropriateness (25%), Content Completeness (10%)
        """
        if not translated_text.strip():
            return {
                'overall': 0.0,
                'english_quality': 0.0,
                'medical_preservation': 0.0,
                'length_appropriateness': 0.0,
                'content_completeness': 0.0
            }

        # 1. English Language Quality Assessment (35%)
        english_quality = self.assess_english_quality(translated_text)

        # 2. Medical Term Preservation Assessment (30%)
        medical_preservation = self.assess_medical_preservation(original_text, translated_text)

        # 3. Length and Information Preservation (25%)
        length_appropriateness = self.assess_length_appropriateness(original_text, translated_text)

        # 4. Content Completeness Assessment (10%)
        content_completeness = self.assess_content_completeness(original_text, translated_text)

        # Calculate weighted overall score
        overall_score = (
            english_quality * 0.35 +
            medical_preservation * 0.30 +
            length_appropriateness * 0.25 +
            content_completeness * 0.10
        )

        return {
            'overall': overall_score,
            'english_quality': english_quality,
            'medical_preservation': medical_preservation,
            'length_appropriateness': length_appropriateness,
            'content_completeness': content_completeness
        }

    def assess_english_quality(self, text: str) -> float:
        """Assess English language quality of translated text."""
        if not text.strip():
            return 0.0

        score = 1.0

        # Language detection confidence
        try:
            detected_lang = detect(text)
            if detected_lang != 'en':
                score *= 0.3
        except:
            score *= 0.5

        # Check for basic English patterns
        common_english_words = r'\b(?:the|and|of|to|a|in|is|it|that|for|on|with|as|by|from|up|about|into|over|after)\b'
        english_word_count = len(re.findall(common_english_words, text.lower()))
        total_words = len(text.split())

        if total_words > 5:
            english_ratio = english_word_count / total_words
            if english_ratio < 0.1:
                score *= 0.5
            elif english_ratio < 0.2:
                score *= 0.7

        # Check for sentence completeness (no abrupt endings)
        sentences = re.split(r'[.!?]+', text)
        complete_sentences = [s for s in sentences if len(s.strip()) > 5]
        if len(sentences) > 0:
            completion_ratio = len(complete_sentences) / len(sentences)
            score *= completion_ratio

        # Check for excessive repetition
        if total_words > 10:
            unique_words = set(text.lower().split())
            diversity_ratio = len(unique_words) / total_words
            if diversity_ratio < 0.3:
                score *= 0.6
            elif diversity_ratio < 0.5:
                score *= 0.8

        return min(1.0, score)

    def assess_medical_preservation(self, original_text: str, translated_text: str) -> float:
        """Assess preservation of critical medical terms."""
        if not original_text or not translated_text:
            return 0.0

        # Get critical terms from original text
        original_critical_terms = self.get_critical_terms_from_text(original_text)

        if not original_critical_terms:
            return 1.0  # No critical terms to preserve

        # Check how many critical terms are preserved
        translated_lower = translated_text.lower()
        preserved_count = 0

        for term in original_critical_terms:
            if term.lower() in translated_lower:
                preserved_count += 1

        preservation_ratio = preserved_count / len(original_critical_terms)

        # Also check for general medical term consistency
        # Count medical-looking terms (abbreviations, technical terms)
        original_medical_pattern = r'\b[A-Z]{2,}(?:-[A-Z]+)*\b|\b\w*(?:ine|mab|nib|tin|ase)\b'
        original_medical_terms = len(re.findall(original_medical_pattern, original_text))
        translated_medical_terms = len(re.findall(original_medical_pattern, translated_text))

        if original_medical_terms > 0:
            medical_density_ratio = min(1.0, translated_medical_terms / original_medical_terms)
            # Combine critical term preservation with general medical density
            combined_score = (preservation_ratio * 0.7) + (medical_density_ratio * 0.3)
        else:
            combined_score = preservation_ratio

        return combined_score

    def assess_length_appropriateness(self, original_text: str, translated_text: str) -> float:
        """Enhanced length assessment with stricter content loss penalties."""
        if not original_text or not translated_text:
            return 0.0

        original_len = len(original_text)
        translated_len = len(translated_text)

        if original_len == 0:
            return 0.0

        length_ratio = translated_len / original_len

        # More stringent length requirements
        if 0.75 <= length_ratio <= 1.4:
            score = 1.0
        elif 0.6 <= length_ratio < 0.75 or 1.4 < length_ratio <= 1.8:
            score = 0.8
        elif 0.45 <= length_ratio < 0.6 or 1.8 < length_ratio <= 2.2:
            score = 0.5
        elif 0.3 <= length_ratio < 0.45:
            score = 0.2  # Severe penalty for major content loss
        else:
            score = 0.1  # Critical content loss

        # Check for information preservation (numbers, percentages)
        original_numbers = re.findall(r'\d+(?:\.\d+)?%?', original_text)
        translated_numbers = re.findall(r'\d+(?:\.\d+)?%?', translated_text)

        if original_numbers:
            number_preservation = min(1.0, len(translated_numbers) / len(original_numbers))
            score = (score * 0.7) + (number_preservation * 0.3)

        return score

    def assess_content_completeness(self, original_text: str, translated_text: str) -> float:
        """Assess overall content completeness and coherence."""
        if not original_text or not translated_text:
            return 0.0
        
        validation_results = self.validate_translation_completeness(original_text, translated_text)
        
        if not validation_results['is_complete']:
            return 0.3  # Major penalty for incomplete translations
        
        # Check content diversity preservation
        original_words = set(original_text.lower().split())
        translated_words = set(translated_text.lower().split())
        
        if len(original_words) > 0:
            # We can't expect word-for-word mapping, but diversity should be preserved
            orig_diversity = len(original_words) / len(original_text.split())
            trans_diversity = len(translated_words) / len(translated_text.split())
            
            diversity_preservation = min(1.0, trans_diversity / orig_diversity) if orig_diversity > 0 else 1.0
        else:
            diversity_preservation = 1.0
        
        # Content structure preservation (sentences, paragraphs)
        orig_sentences = len(re.split(r'[.!?]+', original_text))
        trans_sentences = len(re.split(r'[.!?]+', translated_text))
        
        sentence_preservation = min(1.0, trans_sentences / orig_sentences) if orig_sentences > 0 else 1.0
        
        # Combine metrics
        completeness_score = (
            0.4 * (1.0 - validation_results['content_loss_ratio']) +
            0.3 * diversity_preservation +
            0.3 * sentence_preservation
        )
        
        return max(0.0, min(1.0, completeness_score))

    def assess_document_quality(self, translated_data: dict, original_data: dict, language: str) -> Dict[str, float]:
        """Document-level quality assessment using enhanced approach."""
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
        metrics = ['overall', 'english_quality', 'medical_preservation', 'length_appropriateness', 'content_completeness']
        final_scores = {}

        for metric in metrics:
            scores = [quality[metric] for quality in chunk_qualities if metric in quality]
            final_scores[metric] = sum(scores) / len(scores) if scores else 0.0

        final_scores['chunk_count'] = len(chunk_qualities)

        return final_scores

    def should_escalate_to_tier_2(self, quality_scores: Dict[str, float]) -> bool:
        """Determine if Tier 1 quality is insufficient and escalation needed."""
        overall_quality = quality_scores.get('overall', 0.0)

        # Primary decision: overall quality threshold
        if overall_quality < self.quality_thresholds['tier_1_acceptable']:
            print(f"    📉 Overall quality {overall_quality:.3f} below threshold {self.quality_thresholds['tier_1_acceptable']}")
            return True

        # Check for critical content loss
        content_completeness = quality_scores.get('content_completeness', 0.0)
        if content_completeness < 0.5:
            print(f"    ⚠️ Critical content loss detected {content_completeness:.3f} - escalating")
            return True

        # Secondary checks for specific quality issues
        english_quality = quality_scores.get('english_quality', 0.0)
        medical_preservation = quality_scores.get('medical_preservation', 0.0)
        length_appropriateness = quality_scores.get('length_appropriateness', 0.0)

        # Escalate if English quality is very poor
        if english_quality < 0.5:
            print(f"    🔄 Poor English quality {english_quality:.3f} - escalating")
            return True

        # Escalate if medical preservation is very poor
        if medical_preservation < 0.4:
            print(f"    🏥 Poor medical preservation {medical_preservation:.3f} - escalating")
            return True

        # Escalate if length indicates severe content loss
        if length_appropriateness < 0.4:
            print(f"    📏 Poor length preservation {length_appropriateness:.3f} - escalating")
            return True

        return False

    def compare_translation_quality(self, quality_1: Dict[str, float], quality_2: Dict[str, float]) -> int:
        """Compare two quality scores to determine which is better."""
        # Primary comparison: overall quality
        diff = quality_1['overall'] - quality_2['overall']

        if abs(diff) > 0.05:  # Significant difference
            return 1 if diff > 0 else 2

        # Secondary comparison: content completeness (critical for medical documents)
        completeness_diff = quality_1.get('content_completeness', 0) - quality_2.get('content_completeness', 0)
        if abs(completeness_diff) > 0.1:
            return 1 if completeness_diff > 0 else 2

        # Tertiary comparison: English quality
        english_diff = quality_1.get('english_quality', 0) - quality_2.get('english_quality', 0)
        if abs(english_diff) > 0.08:
            return 1 if english_diff > 0 else 2

        # Final comparison: medical preservation
        medical_diff = quality_1.get('medical_preservation', 0) - quality_2.get('medical_preservation', 0)
        if abs(medical_diff) > 0.08:
            return 1 if medical_diff > 0 else 2

        return 1  # Default to first if similar

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

            self.current_model_type = 'helsinki'
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
                    self.current_model_type = 'helsinki'
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
                    max_input_length = self.model_parameters['nllb']['max_input_length']

                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_input_length)
                    if self.device.startswith("cuda"):
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    gen_kwargs = self.model_parameters['nllb']['generation_params'].copy()
                    gen_kwargs['forced_bos_token_id'] = tokenizer.convert_tokens_to_ids(tgt_lang)

                    with torch.no_grad():
                        translated_tokens = model.generate(**inputs, **gen_kwargs)

                    translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                    return [{'translation_text': translation}]

                except Exception as e:
                    print(f"      NLLB translation error: {str(e)[:50]}")
                    return [{'translation_text': text}]

            self.current_model_type = 'nllb'
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

    def load_translator_for_language(self, language: str, target_tier: int = 1):
        """Load translator for given language - defaults to Tier 1."""
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
            fallback_tier = 2 if target_tier == 1 else 1
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

    def translate_single_chunk(self, text: str, translator) -> str:
        """Enhanced single chunk translation with completeness validation."""
        if not text.strip() or not translator:
            return text

        try:
            # Adaptive chunking based on current model capabilities
            chunk_info_list = self.adaptive_chunk_for_translation(text)

            if len(chunk_info_list) == 1:
                # Single chunk translation
                model_type = self.current_model_type or 'helsinki'
                gen_params = self.model_parameters[model_type]['generation_params'].copy()

                if hasattr(translator, 'model'):
                    result = translator(text, **gen_params)
                else:
                    result = translator(text, generation_params=gen_params)

                translated_text = result[0]['translation_text']
                
                # Validate translation completeness
                validation = self.validate_translation_completeness(text, translated_text)
                if not validation['is_complete']:
                    print(f"      ⚠️ Translation completeness issues: {', '.join(validation['issues_found'])}")
                
            else:
                # Multi-chunk translation with enhanced reassembly
                chunk_results = []
                model_type = self.current_model_type or 'helsinki'
                
                for i, chunk_info in enumerate(chunk_info_list):
                    chunk_text = chunk_info['text']
                    
                    # Adjust parameters for chunk translation
                    gen_params = self.model_parameters[model_type]['generation_params'].copy()
                    # Slightly reduce max_length for chunks to prevent cutoff
                    gen_params['max_length'] = int(gen_params['max_length'] * 0.8)
                    
                    if hasattr(translator, 'model'):
                        result = translator(chunk_text, **gen_params)
                    else:
                        result = translator(chunk_text, generation_params=gen_params)

                    chunk_result = {
                        'translated_text': result[0]['translation_text'],
                        'original_info': chunk_info
                    }
                    chunk_results.append(chunk_result)

                # Enhanced reassembly
                translated_text = self.reassemble_translated_chunks(chunk_results)
                print(f"      ✓ Reassembled {len(chunk_results)} chunks with enhanced structure preservation")
                
                # Validate overall translation completeness
                validation = self.validate_translation_completeness(text, translated_text)
                if not validation['is_complete']:
                    print(f"      ⚠️ Multi-chunk translation issues: {', '.join(validation['issues_found'])}")

            return translated_text.strip()

        except Exception as e:
            print(f"      Translation error: {str(e)[:50]}")
            return text

    def translate_document_with_tier(self, data: dict, language: str, tier: int) -> Tuple[dict, Dict[str, float], Dict[str, Any]]:
        """Translate document with specific tier using enhanced processing."""
        print(f"  📝 Translating document with enhanced Tier {tier}")

        tier_start_time = datetime.now()

        translator = self.load_translator_for_language(language, tier)
        actual_tier = self.current_tier
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
                'tier': actual_tier,
                'model_loaded': True,
                'processing_time_seconds': (datetime.now() - tier_start_time).total_seconds(),
                'model_name': self.current_model_name,
                'chunks_found': False
            }

        total_chunks = len(translated_data['chunks'])
        translated_count = 0
        english_count = 0
        table_count = 0
        completeness_issues = 0

        print(f"    Processing {total_chunks} chunks with enhanced translation...")

        for i, chunk in enumerate(translated_data['chunks']):
            if i % 20 == 0 or i == total_chunks - 1:
                print(f"      Chunk {i+1}/{total_chunks}")

            # Process heading
            if 'heading' in chunk and chunk['heading']:
                if self.is_english_chunk(chunk['heading']):
                    english_count += 1
                else:
                    original_heading = chunk['heading']
                    chunk['heading'] = self.translate_single_chunk(
                        chunk['heading'], translator
                    )
                    # Quick completeness check
                    if len(chunk['heading']) < len(original_heading) * 0.3:
                        completeness_issues += 1
                    translated_count += 1

            # Process text
            if 'text' in chunk and chunk['text']:
                if self.is_english_chunk(chunk['text']):
                    english_count += 1
                else:
                    if self.is_table_content(chunk['text']):
                        table_count += 1

                    original_text = chunk['text']
                    chunk['text'] = self.translate_single_chunk(
                        chunk['text'], translator
                    )
                    # Quick completeness check
                    if len(chunk['text']) < len(original_text) * 0.3:
                        completeness_issues += 1
                    translated_count += 1

        processing_time = (datetime.now() - tier_start_time).total_seconds()

        print(f"    ✓ Enhanced Tier {actual_tier} translation complete:")
        print(f"      English chunks: {english_count}")
        print(f"      Translated chunks: {translated_count}")
        print(f"      Table chunks: {table_count}")
        if completeness_issues > 0:
            print(f"      ⚠️ Potential completeness issues: {completeness_issues}")

        # Enhanced quality assessment
        quality_scores = self.assess_document_quality(translated_data, data, language)

        print(f"    📊 Enhanced Tier {actual_tier} Quality Assessment:")
        print(f"      Overall: {quality_scores['overall']:.3f}")
        print(f"      English Quality: {quality_scores.get('english_quality', 0):.3f}")
        print(f"      Medical Preservation: {quality_scores.get('medical_preservation', 0):.3f}")
        print(f"      Length Appropriateness: {quality_scores.get('length_appropriateness', 0):.3f}")
        print(f"      Content Completeness: {quality_scores.get('content_completeness', 0):.3f}")

        tier_metadata = {
            'tier': actual_tier,
            'model_loaded': True,
            'model_name': self.current_model_name,
            'model_type': self.current_model_type,
            'processing_time_seconds': processing_time,
            'chunks_found': True,
            'total_chunks': total_chunks,
            'chunks_translated': translated_count,
            'chunks_english': english_count,
            'table_chunks_processed': table_count,
            'completeness_issues_detected': completeness_issues,
            'quality_scores': quality_scores
        }

        return translated_data, quality_scores, tier_metadata

    def clear_translator(self):
        """Clear current translator and free memory."""
        self.current_translator = None
        self.current_language = None
        self.current_tier = None
        self.current_model_name = None
        self.current_model_type = None

        gc.collect()
        if self.use_cuda and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except:
                pass

    def process_json_file(self, input_path: str, output_path: str):
        """Process a single JSON file with enhanced approach."""
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

        # Enhanced translation metadata
        translation_metadata = {
            "processing_timestamp": self.processing_start_time.isoformat(),
            "source_file": file_name,
            "detected_language": document_language,
            "was_translation_needed": False,
            "tier_attempts": [],
            "final_tier_used": None,
            "final_model_used": None,
            "final_model_type": None,
            "escalation_occurred": False,
            "quality_comparison": {},
            "final_quality_scores": {},
            "chunks_translated": 0,
            "chunks_preserved_english": 0,
            "table_chunks_processed": 0,
            "completeness_issues_detected": 0,
            "total_processing_time_seconds": 0,
            "translation_strategy": "enhanced_tier_1_first_with_content_preservation",
            "quality_assessment": "enhanced_weighted_approach_with_completeness",
            "chunking_strategy": "adaptive_structure_preserving_with_overlap",
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

        # Check model availability - Enhanced Tier 1 first approach
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

        # Enhanced processing: Start with Tier 1
        final_translation = None
        final_quality = None

        if tier_1_available:
            print(f"  🎯 Starting with enhanced Tier 1 translation")
            tier_1_translation, tier_1_quality, tier_1_metadata = self.translate_document_with_tier(
                data, document_language, 1
            )
            translation_metadata["tier_attempts"].append(tier_1_metadata)

            if not self.should_escalate_to_tier_2(tier_1_quality):
                print(f"  ✅ Enhanced Tier 1 quality sufficient (overall: {tier_1_quality['overall']:.3f})")
                final_translation = tier_1_translation
                final_quality = tier_1_quality
                translation_metadata.update({
                    "translation_decision": "enhanced_tier_1_sufficient",
                    "final_tier_used": 1,
                    "final_model_used": tier_1_metadata["model_name"],
                    "final_model_type": tier_1_metadata["model_type"],
                    "escalation_occurred": False,
                    "completeness_issues_detected": tier_1_metadata.get("completeness_issues_detected", 0)
                })
            else:
                print(f"  📈 Enhanced Tier 1 quality insufficient, escalating to Tier 2")

                if tier_2_available:
                    self.clear_translator()
                    tier_2_translation, tier_2_quality, tier_2_metadata = self.translate_document_with_tier(
                        data, document_language, 2
                    )
                    translation_metadata["tier_attempts"].append(tier_2_metadata)

                    if self.compare_translation_quality(tier_2_quality, tier_1_quality) == 1:
                        print(f"  🏆 Enhanced Tier 2 translation is better")
                        final_translation = tier_2_translation
                        final_quality = tier_2_quality
                        translation_metadata.update({
                            "translation_decision": "enhanced_tier_2_better_than_tier_1",
                            "final_tier_used": 2,
                            "final_model_used": tier_2_metadata["model_name"],
                            "final_model_type": tier_2_metadata["model_type"],
                            "escalation_occurred": True,
                            "completeness_issues_detected": tier_2_metadata.get("completeness_issues_detected", 0),
                            "quality_comparison": {
                                "tier_1_overall": tier_1_quality['overall'],
                                "tier_2_overall": tier_2_quality['overall'],
                                "improvement": tier_2_quality['overall'] - tier_1_quality['overall']
                            }
                        })
                    else:
                        print(f"  🥈 Enhanced Tier 1 translation is still better, keeping it")
                        final_translation = tier_1_translation
                        final_quality = tier_1_quality
                        translation_metadata.update({
                            "translation_decision": "enhanced_tier_1_better_than_tier_2",
                            "final_tier_used": 1,
                            "final_model_used": tier_1_metadata["model_name"],
                            "final_model_type": tier_1_metadata["model_type"],
                            "escalation_occurred": True,
                            "completeness_issues_detected": tier_1_metadata.get("completeness_issues_detected", 0)
                        })
                else:
                    print(f"  ⚠️  Tier 2 not available, accepting enhanced Tier 1 result")
                    final_translation = tier_1_translation
                    final_quality = tier_1_quality
                    translation_metadata.update({
                        "translation_decision": "enhanced_tier_1_only_tier_2_unavailable",
                        "final_tier_used": 1,
                        "final_model_used": tier_1_metadata["model_name"],
                        "final_model_type": tier_1_metadata["model_type"],
                        "escalation_occurred": False,
                        "completeness_issues_detected": tier_1_metadata.get("completeness_issues_detected", 0)
                    })
        else:
            print(f"  🎯 Tier 1 not available, using enhanced Tier 2")
            final_translation, final_quality, tier_2_metadata = self.translate_document_with_tier(
                data, document_language, 2
            )
            translation_metadata["tier_attempts"].append(tier_2_metadata)
            translation_metadata.update({
                "translation_decision": "enhanced_tier_2_only_tier_1_unavailable",
                "final_tier_used": 2,
                "final_model_used": tier_2_metadata["model_name"],
                "final_model_type": tier_2_metadata["model_type"],
                "escalation_occurred": False,
                "completeness_issues_detected": tier_2_metadata.get("completeness_issues_detected", 0)
            })

        # Final statistics with enhanced metrics
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

        print(f"  ✓ Enhanced result: {english_count} English, {translated_count} translated, {table_count} table chunks")
        print(f"  📊 Enhanced quality: {final_quality['overall']:.3f}")
        print(f"  🎯 Content completeness: {final_quality.get('content_completeness', 0):.3f}")
        print(f"  ⏱️  Processing time: {total_processing_time:.2f}s")

        self.english_chunks_preserved += english_count
        self.chunks_translated += translated_count
        self.clear_translator()

    def process_batch_files(self, file_batch: List[Tuple[str, str]]) -> None:
        """Process a batch of files with enhanced handling."""
        print(f"  📦 Processing enhanced batch of {len(file_batch)} files")

        for input_path, output_path in file_batch:
            try:
                self.process_json_file(input_path, output_path)
            except Exception as e:
                print(f"  ✗ Enhanced batch processing error for {os.path.basename(input_path)}: {str(e)[:100]}")
                try:
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    shutil.copy(input_path, output_path)
                    print(f"  📋 Copied original file instead")
                except Exception as copy_error:
                    print(f"  ✗ Failed to copy original: {copy_error}")

    def translate_documents(self):
        """Main translation method with enhanced processing."""
        print("🚀 Starting enhanced medical document translation...")
        print("📋 Enhanced Strategy:")
        print("   • Enhanced chunk management with structure preservation")
        print("   • Intelligent boundary detection and overlap handling")
        print("   • Robust quality control with content completeness validation")
        print("   • Model-specific parameter optimization")
        print("   • Tier 1 models first with smart escalation")
        print("   • Comprehensive quality scoring (English 35%, Medical 30%, Length 25%, Completeness 10%)")

        # Start total runtime tracking
        self.total_translation_start_time = datetime.now()
        print(f"🕐 Enhanced translation started at: {self.total_translation_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

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
        print(f"📁 Found {total_files} JSON files for enhanced processing")

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

        # Process groups with enhanced handling
        for group_dir, group_files in file_groups.items():
            print(f"\n📦 Processing enhanced batch from {group_dir}")
            self.process_batch_files(group_files)

        # Calculate total runtime
        total_translation_end_time = datetime.now()
        total_runtime_seconds = (total_translation_end_time - self.total_translation_start_time).total_seconds()
        total_runtime_minutes = total_runtime_seconds / 60
        total_runtime_hours = total_runtime_minutes / 60

        # Enhanced summary with runtime information
        print(f"\n🎉 Enhanced Medical Translation Complete!")
        print(f"📊 Enhanced Summary:")
        print(f"   • Total files processed: {total_files}")
        print(f"   • English chunks preserved: {self.english_chunks_preserved}")
        print(f"   • Chunks translated: {self.chunks_translated}")
        print(f"   • Enhanced translation with content preservation")
        print(f"   • Structure-aware chunking with overlap management")
        print(f"   • Robust quality control with completeness validation")
        print(f"   • Tier 1 first strategy with intelligent escalation")
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