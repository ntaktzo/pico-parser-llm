import os
from typing import List, Dict, Any, Optional
from langchain.vectorstores import Chroma

class ChunkRetriever:
    """
    Demonstration class for Step 1 of your RAG pipeline:
      1) Primary metadata filtering (e.g., by country)
      2) Secondary metadata filtering (e.g., by heading)
      3) Semantic similarity search
      4) Ensuring coverage across multiple countries/documents
    """

    def __init__(self, vectorstore: Chroma):
        """
        :param vectorstore: A loaded Chroma vectorstore object, 
                            as created by your Vectoriser class.
        """
        self.vectorstore = vectorstore
        # Access to underlying collection, if needed for low-level queries.
        # (Chroma internally exposes the collection as `_collection`.)
        self.chroma_collection = self.vectorstore._collection

    def primary_filter_by_country(
        self, 
        country: str,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Retrieves chunks from the database *strictly* filtered by 'country' metadata.
        This is an optional step if you want to gather all content from a given country
        before narrower filtering.
        
        :param country: e.g. "DE", "FR", etc.
        :param limit: Safety cap on number of chunks.
        :return: List of chunk dicts, each with 'text' and 'metadata' keys.
        """
        # `_collection.get(...)` can be used to filter by metadata:
        result = self.chroma_collection.get(
            where={"country": country},
            limit=limit
        )
        
        # Transform the result into a list of dicts: {"text":..., "metadata":...}
        filtered_chunks = []
        for doc_text, meta in zip(result["documents"], result["metadatas"]):
            filtered_chunks.append({
                "text": doc_text,
                "metadata": meta
            })
        return filtered_chunks

    def secondary_filter_by_heading(
        self,
        chunks: List[Dict[str, Any]],
        heading_keywords: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Narrows a set of already-fetched chunks by examining the 'heading' metadata.
        
        :param chunks: List of chunk dicts (from primary_filter_by_country).
        :param heading_keywords: e.g. ["Population", "Comparator", "Outcomes"].
                                 If None or empty, no heading-based filter is applied.
        :return: Filtered list of chunk dicts.
        """
        if not heading_keywords:
            return chunks  # No additional heading filter

        filtered = []
        for chunk in chunks:
            heading = chunk["metadata"].get("heading", "").lower()
            # If any keyword is found in the heading, keep this chunk
            if any(keyword.lower() in heading for keyword in heading_keywords):
                filtered.append(chunk)
        return filtered

    def vector_similarity_search(
        self,
        query: str,
        where_filter: Optional[Dict[str, Any]] = None,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Performs semantic similarity search within the vectorstore.
        You can combine it with a metadata filter (where_filter) to restrict results.
        """
        docs = self.vectorstore.similarity_search(
            query,  # <-- Pass as positional argument, not named
            k=k,
            filter=where_filter  # Use `filter=` instead of `where=`
        )

        results = [{"text": d.page_content, "metadata": d.metadata} for d in docs]
        return results


    def retrieve_pico_chunks(
        self,
        query: str,
        countries: List[str],
        heading_keywords: Optional[List[str]] = None,
        k: int = 5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        High-level method to systematically ensure coverage across multiple countries.
        
        Workflow for each country:
          1) Primary filter by 'country'
          2) Optional heading-based filter
          3) Vector similarity search (restricted to that country)
          
        This helps ensure we collect relevant chunks from each country,
        while also preserving which chunks came from where.
        
        :param query: A semantic query for the final vector similarity step, 
                      e.g. "comparator therapy" or "pico population".
        :param countries: e.g. ["DE", "FR", "PL", ...]
        :param heading_keywords: e.g. ["Comparator", "Population"]. If None, skip heading filter.
        :param k: Number of top matching chunks per country to return.
        :return: Dictionary keyed by country, each with a list of chunk dicts.
        """
        results_by_country = {}

        for country in countries:
            # 1) Primary: gather all from country
            country_chunks = self.primary_filter_by_country(country)
            
            # 2) Secondary: refine by heading if desired
            if heading_keywords:
                country_chunks = self.secondary_filter_by_heading(country_chunks, heading_keywords)
            
            # 3) Vector similarity search (restricted to the same country)
            #    We rely on metadata to ensure we only retrieve from that country.
            #    If you prefer to *only* search among your local 'country_chunks', you'd need
            #    a separate approach or ephemeral sub-vectorstore. This approach uses the full DB 
            #    but filters by metadata, so only that country’s chunks are considered.
            final_hits = self.vector_similarity_search(
                query=query,
                where_filter={"country": country}, 
                k=k
            )
            
            # Combine or store the final result for the country
            results_by_country[country] = final_hits

        return results_by_country
