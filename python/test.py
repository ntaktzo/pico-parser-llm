# Add this to retrieve.py
import pandas as pd
from IPython.display import display, HTML
import pprint

class RAGDiagnostics:
    """
    A diagnostic tool to test and visualize the retrieval and deduplication processes
    before sending content to the LLM.
    """
    def __init__(self, chunk_retriever, verbose=True):
        """
        Initialize the diagnostic tool with a ChunkRetriever.
        
        Args:
            chunk_retriever: An instance of ChunkRetriever
            verbose: Whether to print detailed output during processing
        """
        self.chunk_retriever = chunk_retriever
        self.verbose = verbose
        self.text_utils = TextSimilarityUtils()
        self.deduplicator = DocumentDeduplicator()
        self.context_manager = ContextManager()
        
    def analyze_retrieval_process(
        self,
        query: str,
        countries: List[str],
        heading_keywords: Optional[List[str]] = None,
        drug_keywords: Optional[List[str]] = None,
        initial_k: int = 30,
        final_k: int = 10
    ):
        """
        Run and analyze the retrieval process for the specified countries.
        
        Args:
            query: Query for vector search
            countries: List of country codes
            heading_keywords: Keywords to boost in headings
            drug_keywords: Drug keywords to boost
            initial_k: Initial number of chunks to retrieve
            final_k: Final number of chunks after filtering
            
        Returns:
            Dictionary containing analysis results for each country
        """
        results = {}
        
        if self.verbose:
            print(f"📋 Running RAG diagnostics with query: '{query}'")
            print(f"🔍 Initial retrieval: {initial_k} chunks, Final target: {final_k} chunks")
            print(f"🌍 Processing countries: {', '.join(countries)}")
            print(f"💡 Heading keywords: {heading_keywords}")
            print(f"💊 Drug keywords: {drug_keywords}")
            
        for country in countries:
            if self.verbose:
                print(f"\n{'='*80}\n📍 ANALYZING COUNTRY: {country}\n{'='*80}")
            
            # Step 1: Initial vector retrieval
            if self.verbose:
                print(f"\n🔎 Step 1: Initial vector retrieval (top {initial_k})")
            
            # Get initial chunks
            docs = self.chunk_retriever.vectorstore.similarity_search(
                query=query,
                k=initial_k,
                filter={"country": country},
            )
            
            if self.verbose:
                print(f"  Retrieved {len(docs)} initial chunks")
                
            # Create record of initial chunks
            initial_chunks = []
            for i, doc in enumerate(docs):
                chunk_summary = {
                    "position": i+1,
                    "text_preview": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content,
                    "heading": doc.metadata.get("heading", "No heading"),
                    "length": len(doc.page_content),
                    "doc_id": doc.metadata.get("doc_id", "Unknown"),
                    "full_text": doc.page_content,
                    "metadata": doc.metadata
                }
                initial_chunks.append(chunk_summary)
            
            # Step 2: Deduplication
            if self.verbose:
                print(f"\n🧹 Step 2: Deduplication of similar chunks")
            
            unique_docs = self.deduplicator.deduplicate_documents(docs)
            
            # Identify which documents were removed by deduplication
            removed_docs = []
            for i, doc in enumerate(docs):
                if doc not in unique_docs:
                    removed_info = {
                        "original_position": i+1,
                        "text_preview": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content,
                        "heading": doc.metadata.get("heading", "No heading"),
                        "length": len(doc.page_content),
                        "full_text": doc.page_content,
                        "similar_to": []  # We'll fill this in next
                    }
                    
                    # Find which document it was similar to
                    for kept_doc in unique_docs:
                        similarity = self.text_utils.jaccard_similarity(doc.page_content, kept_doc.page_content)
                        is_subset = (self.text_utils.is_subset(doc.page_content, kept_doc.page_content) or 
                                    self.text_utils.is_subset(kept_doc.page_content, doc.page_content))
                        
                        if similarity > self.deduplicator.similarity_threshold or is_subset:
                            kept_preview = kept_doc.page_content[:100] + "..."
                            similarity_info = {
                                "preview": kept_preview,
                                "jaccard_similarity": similarity,
                                "is_subset": is_subset
                            }
                            removed_info["similar_to"].append(similarity_info)
                    
                    removed_docs.append(removed_info)
            
            if self.verbose:
                print(f"  {len(removed_docs)} chunks removed as duplicates")
                print(f"  {len(unique_docs)} unique chunks remain")
            
            # Step 3: Score and prioritize chunks
            if self.verbose:
                print(f"\n⭐ Step 3: Scoring and prioritizing chunks")
            
            # Extract comparators and score
            keyword_set = set(kw.lower() for kw in (heading_keywords or []))
            drug_set = set(dr.lower() for dr in (drug_keywords or []))
            
            scored_chunks = []
            
            for i, doc in enumerate(unique_docs):
                # Base score: higher for earlier docs
                base_score = (len(unique_docs) - i)
                
                # Heading boost
                heading_lower = doc.metadata.get("heading", "").lower()
                heading_boost = 0
                if any(k in heading_lower for k in keyword_set):
                    heading_boost = 3.0
                    base_score += heading_boost
                
                # Drug name boost
                text_lower = doc.page_content.lower()
                drug_boost = 0
                if any(drug in text_lower for drug in drug_set):
                    drug_boost = 8.0
                    base_score += drug_boost
                
                # Extract comparators
                comparators = self.text_utils.extract_potential_comparators(doc.page_content)
                
                chunk_info = {
                    "position": i+1,
                    "text_preview": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content,
                    "heading": doc.metadata.get("heading", "No heading"),
                    "base_score": base_score,
                    "heading_boost": heading_boost,
                    "drug_boost": drug_boost,
                    "total_score": base_score,
                    "comparators": list(comparators),
                    "full_text": doc.page_content,
                    "doc": doc
                }
                scored_chunks.append(chunk_info)
            
            # Sort by score
            scored_chunks.sort(key=lambda x: x["total_score"], reverse=True)
            
            if self.verbose:
                print(f"  Scored {len(scored_chunks)} chunks based on relevance")
            
            # Step 4: Maximize comparator coverage
            if self.verbose:
                print(f"\n🧩 Step 4: Maximizing comparator coverage")
            
            # Get top-scored docs
            initial_top_docs = [chunk["doc"] for chunk in scored_chunks[:final_k*2]]
            
            # Prioritize by comparator coverage
            final_docs = self.deduplicator.prioritize_by_comparator_coverage(
                initial_top_docs, 
                final_k=final_k
            )
            
            # Create record of final docs
            final_chunks = []
            all_comparators = set()
            
            for i, doc in enumerate(final_docs):
                comparators = self.text_utils.extract_potential_comparators(doc.page_content)
                all_comparators.update(comparators)
                
                chunk_info = {
                    "position": i+1,
                    "text_preview": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content,
                    "heading": doc.metadata.get("heading", "No heading"),
                    "comparators": list(comparators),
                    "full_text": doc.page_content,
                    "metadata": doc.metadata
                }
                final_chunks.append(chunk_info)
            
            if self.verbose:
                print(f"  Final selection: {len(final_chunks)} chunks")
                print(f"  Total unique comparators: {len(all_comparators)}")
                print(f"  Potential comparators found: {', '.join(sorted(all_comparators))}")
            
            # Step 5: Context building and token management
            if self.verbose:
                print(f"\n📝 Step 5: Context building and token counting")
            
            # Process chunks for token counting
            processed_chunks = self.context_manager.process_chunks([
                {"text": chunk["full_text"], "metadata": chunk["metadata"]} 
                for chunk in final_chunks
            ])
            
            # Build optimal context
            context_block = self.context_manager.build_optimal_context(processed_chunks)
            total_tokens = self.context_manager.count_tokens(context_block)
            
            if self.verbose:
                print(f"  Context built with {total_tokens} tokens")
                print(f"  Context contains {context_block.count(' ') + 1} chunks")
            
            # Store all results for this country
            results[country] = {
                "initial_chunks": initial_chunks,
                "removed_duplicates": removed_docs,
                "scored_chunks": scored_chunks,
                "final_chunks": final_chunks,
                "all_comparators": sorted(list(all_comparators)),
                "total_tokens": total_tokens,
                "context_block": context_block
            }
        
        return results
    
    def print_chunk_details(self, chunk, include_full_text=False):
        """Print details of a single chunk in a readable format."""
        print(f"📄 {chunk['heading']}")
        print(f"   Preview: {chunk['text_preview']}")
        
        if "comparators" in chunk:
            print(f"   Comparators: {', '.join(chunk['comparators'])}")
        
        if "total_score" in chunk:
            print(f"   Score: {chunk['total_score']} (base: {chunk['base_score']}, heading_boost: {chunk['heading_boost']}, drug_boost: {chunk['drug_boost']})")
        
        if include_full_text:
            print("\n--- FULL TEXT ---")
            print(chunk['full_text'])
            print("----------------\n")
        
        print()
    
    def display_results_table(self, results, country, result_type="initial_chunks"):
        """Display results as an HTML table for better visualization in Jupyter notebooks."""
        if result_type not in results[country]:
            print(f"No data for {result_type} in {country}")
            return
            
        chunks = results[country][result_type]
        
        if not chunks:
            print(f"No {result_type} found for {country}")
            return
            
        # Create dataframe for display
        df_data = []
        for chunk in chunks:
            row = {
                "Position": chunk.get("position", ""),
                "Heading": chunk.get("heading", "")[:50] + "..." if len(chunk.get("heading", "")) > 50 else chunk.get("heading", ""),
                "Preview": chunk.get("text_preview", "")[:100] + "..." if len(chunk.get("text_preview", "")) > 100 else chunk.get("text_preview", ""),
                "Length": chunk.get("length", len(chunk.get("full_text", "")))
            }
            
            if "total_score" in chunk:
                row["Score"] = f"{chunk['total_score']:.1f}"
                
            if "comparators" in chunk:
                row["Comparators"] = ", ".join(chunk["comparators"][:3]) + ("..." if len(chunk["comparators"]) > 3 else "")
                
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        display(HTML(df.to_html(index=False)))

    def create_diagnostic_report(self, results, country):
        """Create a detailed diagnostic report for the specified country."""
        if country not in results:
            print(f"No results for country: {country}")
            return
            
        country_results = results[country]
        
        print(f"\n{'='*100}")
        print(f"📊 DIAGNOSTIC REPORT FOR {country}")
        print(f"{'='*100}\n")
        
        # 1. Overview
        print(f"📌 OVERVIEW")
        print(f"  Initial chunks retrieved: {len(country_results['initial_chunks'])}")
        print(f"  Chunks removed as duplicates: {len(country_results['removed_duplicates'])}")
        print(f"  Final chunks selected: {len(country_results['final_chunks'])}")
        print(f"  Total tokens in context: {country_results['total_tokens']}")
        print(f"  Potential comparators found: {', '.join(country_results['all_comparators'])}")
        print()
        
        # 2. Initial Retrieval
        print(f"📌 INITIAL RETRIEVAL")
        print(f"  Top 5 initially retrieved chunks:")
        for chunk in country_results['initial_chunks'][:5]:
            print(f"  {chunk['position']}. {chunk['heading']} - {chunk['text_preview'][:70]}...")
        print()
        
        # 3. Removed Duplicates
        print(f"📌 REMOVED DUPLICATES")
        if not country_results['removed_duplicates']:
            print("  No duplicates removed.")
        else:
            print(f"  First 3 removed duplicates:")
            for i, doc in enumerate(country_results['removed_duplicates'][:3]):
                print(f"  {i+1}. {doc['heading']} - {doc['text_preview'][:70]}...")
                if doc['similar_to']:
                    print(f"     Similar to: {doc['similar_to'][0]['preview'][:70]}...")
                    print(f"     Similarity: {doc['similar_to'][0]['jaccard_similarity']:.2f}")
        print()
        
        # 4. Final Selection
        print(f"📌 FINAL SELECTION")
        print(f"  Selected chunks with comparators:")
        for chunk in country_results['final_chunks']:
            print(f"  - {chunk['heading']}")
            print(f"    Comparators: {', '.join(chunk['comparators'])}")
            print(f"    Preview: {chunk['text_preview'][:70]}...")
        print()
        
        # 5. Context Sample
        print(f"📌 CONTEXT SAMPLE")
        context_preview = country_results['context_block'][:300] + "..."
        print(f"  {context_preview}")
        print()
        
        return f"Diagnostic report complete for {country}. Full details available in the results dictionary."