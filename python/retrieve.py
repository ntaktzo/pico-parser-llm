import os
import json
import re
import tiktoken
from typing import List, Dict, Any, Optional, Union
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, BaseMessage
import pandas as pd
from IPython.display import display, HTML



class TextSimilarityUtils:
    """
    Utility class for text similarity and comparator extraction functions.
    """
    @staticmethod
    def jaccard_similarity(text1, text2):
        """Calculate Jaccard similarity between two text strings."""
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        
        intersection = len(set1.intersection(set2))
        union = len(set1) + len(set2) - intersection
        
        return intersection / union if union > 0 else 0

    @staticmethod
    def is_subset(text1, text2):
        """Check if text1 is effectively contained within text2."""
        # Clean and tokenize
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        # If most of text1's tokens are in text2, consider it a subset
        overlap_ratio = len(tokens1.intersection(tokens2)) / len(tokens1) if tokens1 else 0
        return overlap_ratio > 0.9  # 90% of tokens are contained

    @staticmethod
    def extract_potential_comparators(text):
        """
        Extract potential drug names/comparators from text using pattern matching.
        Completely rewritten to avoid any look-behind patterns.
        """
        words = text.split()
        capitalized_words = []
        
        # Find capitalized words that might be drug names
        for i, word in enumerate(words):
            # Check if this is a potential sentence start or after a space
            if (i > 0 and words[i-1][-1] in '.!?') or i == 0:
                # Clean the word of punctuation
                clean_word = word.strip('.,;:()[]{}')
                if clean_word and clean_word[0].isupper() and len(clean_word) > 1 and clean_word.lower() not in ['the', 'a', 'an', 'in', 'on', 'at', 'for', 'with', 'by', 'to', 'of']:
                    capitalized_words.append(clean_word)
        
        # Find words followed by dosages (simple pattern)
        dosage_pattern = r'\b\w+\s+\d+\s*(?:mg|mcg|g|ml)\b'
        dosages = re.findall(dosage_pattern, text)
        
        # Find drug name suffixes
        suffix_pattern = r'\b\w+(?:mab|nib|zumab|tinib|ciclib|parib|vastatin)\b'
        suffix_matches = re.findall(suffix_pattern, text.lower())
        
        # Combine all matches
        all_matches = capitalized_words + dosages + suffix_matches
        
        # Filter out common words that aren't likely drug names
        common_words = {'the', 'and', 'for', 'with', 'that', 'this', 'not', 'are', 'from', 'was', 'were'}
        filtered_matches = [m for m in all_matches if m.lower() not in common_words]
        
        return set(filtered_matches)



class DocumentDeduplicator:
    """
    Class to handle deduplication of retrieved documents and context optimization.
    """
    def __init__(self, similarity_threshold=0.7):
        self.similarity_threshold = similarity_threshold
        self.similarity_utils = TextSimilarityUtils()
    
    def deduplicate_documents(self, docs, preserve_country_diversity=True):
        """
        Deduplicate similar documents while preserving diversity.
        """
        unique_docs = []
        seen_texts = set()
        removed_docs = []  # Track removed documents
        
        for doc in docs:
            # Simple deduplication for identical content
            text = doc.page_content.strip()
            if text in seen_texts:
                removed_info = {
                    "doc": doc,
                    "reason": "exact duplicate",
                    "similar_to": None
                }
                removed_docs.append(removed_info)
                continue
                
            # Check for near-duplicates or subset relationships
            is_duplicate = False
            similar_to = None
            
            for kept_doc in unique_docs:
                # Skip comparison if preserving country diversity and documents are from different countries
                if preserve_country_diversity and doc.metadata.get("country") != kept_doc.metadata.get("country"):
                    continue
                    
                similarity = self.similarity_utils.jaccard_similarity(text, kept_doc.page_content)
                is_subset_relation = (self.similarity_utils.is_subset(text, kept_doc.page_content) or 
                                     self.similarity_utils.is_subset(kept_doc.page_content, text))
                
                if similarity > self.similarity_threshold or is_subset_relation:
                    is_duplicate = True
                    similar_to = kept_doc
                    break
            
            if is_duplicate:
                removed_info = {
                    "doc": doc,
                    "reason": f"similarity: {similarity:.2f}" if 'similarity' in locals() else "subset relation",
                    "similar_to": similar_to
                }
                removed_docs.append(removed_info)
            else:
                unique_docs.append(doc)
                seen_texts.add(text)
                
        return unique_docs, removed_docs
    
    def prioritize_by_comparator_coverage(self, docs, final_k=10):
        """
        Score and prioritize documents to maximize comparator coverage.
        """
        # Extract all potential comparators from documents
        all_comparators = set()
        doc_comparators = []
        
        for doc in docs:
            comparators = self.similarity_utils.extract_potential_comparators(doc.page_content)
            all_comparators.update(comparators)
            doc_comparators.append((doc, comparators))
        
        # Prioritize documents with unique comparators
        selected_docs = []
        covered_comparators = set()
        skipped_docs = []
        
        # Sort by number of unique comparators (most unique first)
        while doc_comparators and len(selected_docs) < final_k:
            # Find document with most uncovered comparators
            best_idx = -1
            best_unique_count = -1
            
            for idx, (_, comparators) in enumerate(doc_comparators):
                unique_count = len(comparators - covered_comparators)
                if unique_count > best_unique_count:
                    best_unique_count = unique_count
                    best_idx = idx
            
            if best_idx >= 0:
                doc, comparators = doc_comparators.pop(best_idx)
                selected_docs.append(doc)
                covered_comparators.update(comparators)
            else:
                # If no more unique comparators, just take the next document
                doc, comparators = doc_comparators.pop(0)
                selected_docs.append(doc)
        
        # Remaining docs weren't selected
        for doc, comparators in doc_comparators:
            skipped_docs.append((doc, comparators))
            
        return selected_docs, skipped_docs, covered_comparators



class ChunkRetriever:
    """
    Retriever with improved deduplication and adaptive context handling.
    """
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.chroma_collection = self.vectorstore._collection
        self.deduplicator = DocumentDeduplicator()
        self.similarity_utils = TextSimilarityUtils()
        self.context_manager = ContextManager()
        self.debug_mode = False

    def primary_filter_by_country(
        self, 
        country: str,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        result = self.chroma_collection.get(
            where={"country": country},
            limit=limit
        )
        return [
            {"text": txt, "metadata": meta}
            for txt, meta in zip(result["documents"], result["metadatas"])
        ]


# Modified portions of the ChunkRetriever class in retrieve.py

class ChunkRetriever:
    """
    Retriever with improved deduplication and adaptive context handling.
    Enhanced to support source type filtering.
    """
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.chroma_collection = self.vectorstore._collection
        self.deduplicator = DocumentDeduplicator()
        self.similarity_utils = TextSimilarityUtils()
        self.context_manager = ContextManager()
        self.debug_mode = False

    def primary_filter_by_country(
        self, 
        country: str,
        source_filter: Optional[Dict[str, Any]] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Filter by country with optional additional source type filter.
        
        Args:
            country: Country code to filter by
            source_filter: Optional additional filter by source type
            limit: Maximum number of results to return
        """
        # Combine country filter with source filter if provided
        where_filter = {"country": country}
        if source_filter:
            where_filter.update(source_filter)
            
        result = self.chroma_collection.get(
            where=where_filter,
            limit=limit
        )
        return [
            {"text": txt, "metadata": meta}
            for txt, meta in zip(result["documents"], result["metadatas"])
        ]

    def vector_similarity_search(
        self,
        query: str,
        filter_meta: Optional[Dict[str, Any]] = None,
        heading_keywords: Optional[List[str]] = None,
        drug_keywords: Optional[List[str]] = None,
        initial_k: int = 30,
        final_k: int = 10,
        heading_boost: float = 3.0,
        drug_boost: float = 8.0,
        source_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents by vector similarity with heading and drug boosts.
        Enhanced to support source type filtering.
        """
        # For tracking retrieval process in debug mode
        retrieval_info = {
            "initial_docs": [],
            "removed_docs": [],
            "scored_docs": [],
            "skipped_docs": [],
            "final_docs": [],
            "all_comparators": set()
        }
        
        # Combine filter_meta with source_filter if provided
        combined_filter = filter_meta or {}
        if source_filter:
            combined_filter.update(source_filter)
        
        # Get initial chunks via vector similarity
        docs = self.vectorstore.similarity_search(
            query=query,
            k=initial_k,
            filter=combined_filter,
        )
        
        # Store initial docs for debug
        if self.debug_mode:
            for i, doc in enumerate(docs):
                retrieval_info["initial_docs"].append({
                    "position": i+1,
                    "doc": doc,
                    "heading": doc.metadata.get("heading", "No heading"),
                    "text_preview": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content,
                    "length": len(doc.page_content)
                })
        
        # Deduplicate similar chunks
        unique_docs, removed_docs = self.deduplicator.deduplicate_documents(docs)
        
        # Store removed docs for debug
        if self.debug_mode:
            for removed in removed_docs:
                doc = removed["doc"]
                similar_to = removed["similar_to"]
                retrieval_info["removed_docs"].append({
                    "doc": doc,
                    "heading": doc.metadata.get("heading", "No heading"),
                    "text_preview": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content,
                    "reason": removed["reason"],
                    "similar_to": similar_to.page_content[:150] + "..." if similar_to else None
                })
        
        # Score chunks by heading and drug keyword relevance
        keyword_set = set(kw.lower() for kw in (heading_keywords or []))
        drug_set = set(dr.lower() for dr in (drug_keywords or []))
        
        scored_docs = []
        for i, doc in enumerate(unique_docs):
            # Base score: higher for earlier docs
            base_score = (len(unique_docs) - i)
            
            # Heading boost
            heading_lower = doc.metadata.get("heading", "").lower()
            heading_boost_applied = 0
            if any(k in heading_lower for k in keyword_set):
                heading_boost_applied = heading_boost
                base_score += heading_boost_applied
                
            # Drug name boost
            text_lower = doc.page_content.lower()
            drug_boost_applied = 0
            if any(drug in text_lower for drug in drug_set):
                drug_boost_applied = drug_boost
                base_score += drug_boost_applied
                
            scored_docs.append((doc, base_score, heading_boost_applied, drug_boost_applied))
            
            # Store scored docs for debug
            if self.debug_mode:
                comparators = self.similarity_utils.extract_potential_comparators(doc.page_content)
                retrieval_info["scored_docs"].append({
                    "position": i+1,
                    "doc": doc,
                    "heading": doc.metadata.get("heading", "No heading"),
                    "text_preview": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content,
                    "base_score": base_score - heading_boost_applied - drug_boost_applied,
                    "heading_boost": heading_boost_applied,
                    "drug_boost": drug_boost_applied,
                    "total_score": base_score,
                    "comparators": list(comparators)
                })
        
        # Sort by score (highest first)
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Take top K by score, then ensure maximal comparator coverage 
        # by prioritizing documents with unique comparators
        initial_top_docs = [doc for doc, _, _, _ in scored_docs[:final_k*2]]  # Get more than needed initially
        
        # Prioritize by comparator coverage
        selected_docs, skipped_docs, covered_comparators = self.deduplicator.prioritize_by_comparator_coverage(
            initial_top_docs, 
            final_k=final_k
        )
        
        # Store final selection and skipped docs for debug
        if self.debug_mode:
            retrieval_info["all_comparators"] = covered_comparators
            
            for i, doc in enumerate(selected_docs):
                comparators = self.similarity_utils.extract_potential_comparators(doc.page_content)
                retrieval_info["final_docs"].append({
                    "position": i+1,
                    "doc": doc,
                    "heading": doc.metadata.get("heading", "No heading"),
                    "text_preview": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content,
                    "comparators": list(comparators)
                })
                
            for i, (doc, comparators) in enumerate(skipped_docs):
                retrieval_info["skipped_docs"].append({
                    "doc": doc,
                    "heading": doc.metadata.get("heading", "No heading"),
                    "text_preview": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content,
                    "comparators": list(comparators),
                    "reason": "Insufficient unique comparators or over limit"
                })
        
        # Return the formatted results
        formatted_docs = [
            {"text": doc.page_content, "metadata": doc.metadata}
            for doc in selected_docs
        ]
        
        return formatted_docs if not self.debug_mode else (formatted_docs, retrieval_info)

        
    # Alias for backward compatibility with existing code
    vector_similarity_search_with_deduplication = vector_similarity_search

    def retrieve_pico_chunks(
        self,
        query: str,
        countries: List[str],
        heading_keywords: Optional[List[str]] = None,
        drug_keywords: Optional[List[str]] = None,
        initial_k: int = 30,
        final_k: int = 10,
        debug: bool = False,
        source_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve chunks by country using vector similarity search.
        Enhanced to support source type filtering.
        
        Args:
            query: The search query
            countries: List of country codes to retrieve for
            heading_keywords: Keywords to boost if found in headings
            drug_keywords: Drug keywords to boost in content
            initial_k: Initial number of chunks to retrieve
            final_k: Final number of chunks after filtering
            debug: Whether to return debug information
            source_filter: Optional filter by source type (e.g., {"source_type": "hta_submission"})
            
        Returns:
            Dictionary mapping countries to retrieved chunks, with optional debug info
        """
        self.debug_mode = debug
        results_by_country = {}
        debug_info = {}

        for country in countries:
            # Create combined filter with country and source type
            filter_meta = {"country": country}
            if source_filter:
                filter_meta.update(source_filter)
            
            # Use vector similarity search with debugging if enabled
            if debug:
                final_hits, country_debug_info = self.vector_similarity_search(
                    query=query,
                    filter_meta=filter_meta,
                    heading_keywords=heading_keywords,
                    drug_keywords=drug_keywords,
                    initial_k=initial_k,
                    final_k=final_k
                )
                debug_info[country] = country_debug_info
            else:
                final_hits = self.vector_similarity_search(
                    query=query,
                    filter_meta=filter_meta,
                    heading_keywords=heading_keywords,
                    drug_keywords=drug_keywords,
                    initial_k=initial_k,
                    final_k=final_k
                )
                
            results_by_country[country] = final_hits

        return results_by_country if not debug else (results_by_country, debug_info)
    
    def test_retrieval(
        self,
        query: str,
        countries: List[str],
        heading_keywords: Optional[List[str]] = None,
        drug_keywords: Optional[List[str]] = None,
        initial_k: int = 20,
        final_k: int = 10,
        show_tables: bool = False,
        source_filter: Optional[Dict[str, Any]] = None
    ):
        """
        Test the retrieval pipeline and show diagnostic information.
        Enhanced to support source type filtering.
        
        Args:
            query: The search query
            countries: List of country codes to test
            heading_keywords: Keywords to boost in headings
            drug_keywords: List of drug names to boost
            initial_k: Initial number of chunks to retrieve
            final_k: Final number of chunks after filtering
            show_tables: Whether to display HTML tables (for notebooks)
            source_filter: Optional filter by source type (e.g., {"source_type": "clinical_guideline"})
            
        Returns:
            Dictionary with retrieval results and diagnostics
        """
        source_type = source_filter.get("source_type", "any") if source_filter else "any"
        print(f"📋 Testing retrieval pipeline for query: '{query}'")
        print(f"🔍 Initial retrieval: {initial_k} chunks, Final target: {final_k} chunks")
        print(f"🌍 Processing countries: {', '.join(countries)}")
        print(f"📑 Source type filter: {source_type}")
        print(f"💡 Heading keywords: {heading_keywords}")
        print(f"💊 Drug keywords: {drug_keywords}")
        
        # Retrieve chunks with debug info and source filter
        results, debug_info = self.retrieve_pico_chunks(
            query=query,
            countries=countries,
            heading_keywords=heading_keywords,
            drug_keywords=drug_keywords,
            initial_k=initial_k,
            final_k=final_k,
            debug=True,
            source_filter=source_filter
        )
        
        # Print diagnostics for each country
        for country in countries:
            country_debug = debug_info[country]
            country_results = results[country]
            
            print(f"\n{'='*80}\n📍 COUNTRY: {country}\n{'='*80}")
            
            # Summary
            print(f"\n📊 SUMMARY:")
            print(f"  Initial chunks retrieved: {len(country_debug['initial_docs'])}")
            print(f"  Chunks removed as duplicates: {len(country_debug['removed_docs'])}")
            print(f"  Final chunks selected: {len(country_debug['final_docs'])}")
            
            # Initial retrieval
            print(f"\n🔎 INITIAL RETRIEVAL (showing top 5):")
            for doc in country_debug['initial_docs'][:5]:
                print(f"  * {doc['heading']}")
                print(f"    {doc['text_preview'][:100]}...")
            
            # Removed duplicates
            if country_debug['removed_docs']:
                print(f"\n🗑️ REMOVED DUPLICATES (showing first 3):")
                for doc in country_debug['removed_docs'][:3]:
                    print(f"  * {doc['heading']}")
                    print(f"    Reason: {doc['reason']}")
                    if doc['similar_to']:
                        print(f"    Similar to: {doc['similar_to'][:70]}...")
            
            # Final selection
            print(f"\n✅ FINAL SELECTION:")
            for doc in country_debug['final_docs']:
                print(f"  * {doc['heading']}")
                print(f"    {doc['text_preview'][:100]}...")
            
            # Top scored chunks (now shown last)
            print(f"\n⭐ TOP SCORED CHUNKS (showing top 5):")
            for doc in country_debug['scored_docs'][:5]:
                print(f"  * {doc['heading']} (Score: {doc['total_score']:.1f})")
                if doc['heading_boost'] > 0:
                    print(f"    Heading boost: +{doc['heading_boost']}")
                if doc['drug_boost'] > 0:
                    print(f"    Drug boost: +{doc['drug_boost']}")
            
            # Display tables if requested
            if show_tables:
                try:
                    print("\nInitial Chunks:")
                    self._display_table(country_debug, "initial_docs")
                    
                    print("\nRemoved Duplicates:")
                    self._display_table(country_debug, "removed_docs")
                    
                    print("\nFinal Selection:")
                    self._display_table(country_debug, "final_docs")
                except Exception as e:
                    print(f"Could not display tables: {e}")
        
        return {"results": results, "debug_info": debug_info}
    
    def _display_table(self, debug_info, section):
        """Helper method to display a table of results in Jupyter notebooks."""
        try:
            if section not in debug_info or not debug_info[section]:
                print(f"No data available for {section}")
                return
                
            data = []
            for item in debug_info[section]:
                row = {}
                
                if "position" in item:
                    row["Pos"] = item["position"]
                
                if "heading" in item:
                    row["Heading"] = item["heading"][:50] + "..." if len(item["heading"]) > 50 else item["heading"]
                
                if "text_preview" in item:
                    row["Preview"] = item["text_preview"][:70] + "..." if len(item["text_preview"]) > 70 else item["text_preview"]
                
                if "reason" in item:
                    row["Reason"] = item["reason"]
                    
                if "total_score" in item:
                    row["Score"] = f"{item['total_score']:.1f}"
                
                data.append(row)
            
            df = pd.DataFrame(data)
            display(HTML(df.to_html(index=False)))
        except Exception as e:
            print(f"Error displaying table: {e}")


class ContextManager:
    """
    Class to handle adaptive context management for LLM prompts.
    """
    def __init__(self, max_tokens=12000):
        self.max_tokens = max_tokens
        self.similarity_utils = TextSimilarityUtils()
        
        # Initialize tokenizer
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-4")
        except:
            self.encoding = tiktoken.encoding_for_model("cl100k_base")  # Fallback
    
    def count_tokens(self, text):
        """Count tokens in text using the current encoding."""
        return len(self.encoding.encode(text))
    
    def process_chunks(self, chunks):
        """Process chunks to estimate tokens and extract comparators."""
        processed = []
        for chunk in chunks:
            text = chunk.get("text", "").strip()
            if not text:
                continue
            
            token_count = self.count_tokens(text)
            comparators = self.similarity_utils.extract_potential_comparators(text)
            
            processed.append({
                "text": text,
                "tokens": token_count,
                "comparators": comparators,
                "metadata": chunk.get("metadata", {})
            })
        
        return processed
    
    def build_optimal_context(self, processed_chunks):
        """
        Build optimal context block maximizing comparator coverage
        while respecting token limits.
        """
        context_parts = []
        current_tokens = 0
        covered_comparators = set()
        
        # Get all potential comparators
        all_comparators = set()
        for chunk in processed_chunks:
            all_comparators.update(chunk["comparators"])
        
        # Sort by unique comparator coverage
        def sort_key(chunk):
            unique_count = len(chunk["comparators"] - covered_comparators)
            return unique_count
        
        # First pass: include chunks with unique comparators
        remaining_chunks = list(processed_chunks)
        while remaining_chunks and current_tokens < self.max_tokens:
            # Resort each time as covered_comparators changes
            remaining_chunks.sort(key=sort_key, reverse=True)
            chunk = remaining_chunks.pop(0)
            
            # Skip if adding would exceed token limit (unless it has unique critical info)
            if current_tokens + chunk["tokens"] > self.max_tokens:
                new_comparators = chunk["comparators"] - covered_comparators
                # Only include if it has unique comparators and we're not too far over limit
                if not new_comparators or current_tokens + chunk["tokens"] > self.max_tokens * 1.1:
                    continue
            
            context_parts.append(chunk["text"])
            current_tokens += chunk["tokens"]
            covered_comparators.update(chunk["comparators"])
            
            # If we've covered all comparators, we can stop
            if covered_comparators >= all_comparators:
                break
        
        return "\n\n".join(context_parts)


class PICOExtractor:
    """
    PICOExtractor with improved chunk deduplication and adaptive context handling.
    """
    def __init__(
        self,
        chunk_retriever,
        system_prompt: str,
        user_prompt_template: str,
        model_name: str = "gpt-4o-mini"
    ):
        self.chunk_retriever = chunk_retriever
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.model_name = model_name
        self.llm = ChatOpenAI(model_name=self.model_name, temperature=0)
        self.context_manager = ContextManager()

    def debug_retrieve_chunks(
        self,
        countries: List[str],
        query: str,
        initial_k: int = 10,
        final_k: int = 5,
        heading_keywords: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        For debugging, retrieve relevant chunks for each country and print them.
        """
        retrieval_results = {}

        for country in countries:
            # Retrieve chunks for this country
            results_dict = self.chunk_retriever.retrieve_pico_chunks(
                query=query,
                countries=[country],
                heading_keywords=heading_keywords,
                initial_k=initial_k,
                final_k=final_k
            )

            country_chunks = results_dict.get(country, [])
            retrieval_results[country] = country_chunks

            # Print each chunk for debugging
            print(f"===== Retrieved Chunks for country: {country} =====")
            if not country_chunks:
                print("No chunks retrieved.")
            else:
                for idx, chunk_info in enumerate(country_chunks, start=1):
                    print(f"Chunk {idx}:")
                    print(f"  {chunk_info}")
            print("========================================\n")

        return retrieval_results

    def extract_picos(
        self,
        countries: List[str],
        query: str,
        initial_k: int = 10,
        final_k: int = 5,
        heading_keywords: Optional[List[str]] = None,
        model_override: Optional[Union[str, ChatOpenAI]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract PICOs using improved context management and LLM.
        """
        results = []

        # Optionally override the model
        llm_to_use = self.llm
        if model_override:
            if isinstance(model_override, str):
                llm_to_use = ChatOpenAI(model_name=model_override, temperature=0)
            elif isinstance(model_override, ChatOpenAI):
                llm_to_use = model_override

        # Ensure the output directory exists
        os.makedirs("results", exist_ok=True)

        for country in countries:
            # Retrieve chunks for the country
            results_dict = self.chunk_retriever.retrieve_pico_chunks(
                query=query,
                countries=[country],
                heading_keywords=heading_keywords,
                initial_k=initial_k,
                final_k=final_k
            )
            
            country_chunks = results_dict.get(country, [])
            if not country_chunks:
                continue

            # Process chunks with context manager
            processed_chunks = self.context_manager.process_chunks(country_chunks)
            context_block = self.context_manager.build_optimal_context(processed_chunks)

            # Prepare system and user messages
            system_msg = SystemMessage(content=self.system_prompt)
            user_msg_text = self.user_prompt_template.format(
                context_block=context_block,
                country=country
            )
            user_msg = HumanMessage(content=user_msg_text)

            # LLM call
            try:
                llm_response: BaseMessage = llm_to_use([system_msg, user_msg])
            except Exception as exc:
                print(f"LLM call failed for {country}: {exc}")
                continue

            answer_text = getattr(llm_response, 'content', str(llm_response))

            # Parse JSON response
            try:
                parsed_json = json.loads(answer_text)
            except json.JSONDecodeError:
                # Retry once with explicit instruction to fix JSON
                fix_msg = HumanMessage(content="Please correct and return valid JSON in the specified format only.")
                try:
                    fix_response = llm_to_use([system_msg, user_msg, fix_msg])
                    fix_text = getattr(fix_response, 'content', str(fix_response))
                    parsed_json = json.loads(fix_text)
                except Exception as parse_err:
                    print(f"Failed to parse JSON for {country}: {parse_err}")
                    continue

            # Save results
            if isinstance(parsed_json, dict):
                parsed_json["Country"] = country  # Ensure correct country
                results.append(parsed_json)
                outpath = os.path.join("results", f"picos_{country}.json")
                with open(outpath, "w", encoding="utf-8") as f:
                    json.dump(parsed_json, f, indent=2, ensure_ascii=False)
            else:
                # Handle non-dict response
                wrapped_json = {
                    "Country": country,
                    "PICOs": parsed_json if isinstance(parsed_json, list) else []
                }
                results.append(wrapped_json)
                outpath = os.path.join("results", f"picos_{country}.json")
                with open(outpath, "w", encoding="utf-8") as f:
                    json.dump(wrapped_json, f, indent=2, ensure_ascii=False)

        return results