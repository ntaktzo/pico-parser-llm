import json
from typing import List, Dict, Any, Optional, Union
from chromadb import Collection
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, BaseMessage


class ChunkRetriever:
    """
    Updated class for RAG retrieval pipeline with corrected metadata filtering.
    """

    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore
        self.chroma_collection: Collection = self.vectorstore._collection

    def primary_filter_by_country(
        self, 
        country: str,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        result = self.chroma_collection.get(
            where={"country": country},
            limit=limit
        )
        
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
        Narrows chunks by 'heading' metadata.
        """
        if not heading_keywords:
            return chunks

        filtered = []
        for chunk in chunks:
            heading = chunk["metadata"].get("heading", "").lower()
            if any(keyword.lower() in heading for keyword in heading_keywords):
                filtered.append(chunk)
        return filtered

    def vector_similarity_search(
        self,
        query: str,
        filter_meta: Optional[Dict[str, Any]] = None,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Semantic similarity search within vectorstore.
        """
        docs = self.vectorstore.similarity_search(
            query, 
            k=k,
            filter=filter_meta
        )

        return [{"text": d.page_content, "metadata": d.metadata} for d in docs]

    def retrieve_pico_chunks(
        self,
        query: str,
        countries: List[str],
        heading_keywords: Optional[List[str]] = None,
        k: int = 5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Ensures coverage across multiple countries.
        """
        results_by_country = {}

        for country in countries:
            country_chunks = self.primary_filter_by_country(country)
            
            if heading_keywords:
                country_chunks = self.secondary_filter_by_heading(country_chunks, heading_keywords)

            # Use metadata filter to restrict similarity search to the given country
            final_hits = self.vector_similarity_search(
                query=query,
                filter_meta={"country": country}, 
                k=k
            )

            results_by_country[country] = final_hits

        return results_by_country



class PICOExtractor:
    """
    PICOExtractor is a class that integrates with a ChunkRetriever and a LangChain-compatible LLM 
    (such as OpenAI's ChatGPT) to extract PICO (Population, Intervention, Comparator, Outcome) elements 
    from chunked Health Technology Assessment (HTA) documents.

    This class:
    - Accepts a system prompt and model name during initialization.
    - For each country, retrieves the top-k relevant document chunks using the ChunkRetriever.
    - Deduplicates and concatenates chunk text to form a prompt context.
    - Queries the LLM with the context to extract structured PICO information.
    - Saves each country's extracted PICOs to a file in the 'results' folder.
    - Returns a list of all extracted PICO dictionaries, one per country.
    """

    def __init__(
        self,
        chunk_retriever,  # Instance of ChunkRetriever
        system_prompt: str,  # The system prompt used to guide the LLM's behavior
        model_name: str = "gpt-4o-mini"  # Default model name
    ):
        self.chunk_retriever = chunk_retriever
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.llm = ChatOpenAI(model_name=self.model_name, temperature=0)

    def debug_retrieve_chunks(
        self,
        countries: List[str],
        query: str,
        k: int = 5,
        heading_keywords: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        For debugging, retrieve relevant chunks for each country and print them.

        :param countries: List of country codes to retrieve chunks for.
        :param query: Query string used for semantic search on the chunk retriever.
        :param k: Number of top chunks to retrieve per country.
        :param heading_keywords: Optional list of heading keywords to filter chunks.
        :return: A dictionary keyed by country, mapping to a list of retrieved chunk dicts.
        """
        retrieval_results = {}

        for country in countries:
            # Retrieve chunks for this country
            results_dict = self.chunk_retriever.retrieve_pico_chunks(
                query=query,
                countries=[country],
                heading_keywords=heading_keywords,
                k=k
            )

            # The retriever's return is a dict keyed by country
            country_chunks = results_dict.get(country, [])

            # Store in our return dictionary
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
        k: int = 5,
        heading_keywords: Optional[List[str]] = None,
        model_override: Optional[Union[str, ChatOpenAI]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extracts PICO elements for a list of countries using retrieved document chunks and an LLM.

        :param countries: List of country codes.
        :param query: Query string used to retrieve relevant chunks.
        :param k: Number of top chunks to retrieve per country.
        :param heading_keywords: Optional list of heading keywords to further filter chunks.
        :param model_override: Optional model override as a string or ChatOpenAI instance.
        :return: A list of dictionaries, each containing the country code and extracted PICOs.
        """
        results = []

        # Use model override if provided
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
                k=k
            )
            country_chunks = results_dict.get(country, [])
            if not country_chunks:
                continue

            # Deduplicate chunk texts
            seen_texts = set()
            context_parts = []
            for chunk_info in country_chunks:
                text = chunk_info.get("text", "").strip()
                if text and text not in seen_texts:
                    seen_texts.add(text)
                    context_parts.append(text)

            context_block = "\n\n".join(context_parts)

            # Construct system and user messages
            system_msg = SystemMessage(content=self.system_prompt)
            user_prompt = (
                f"Context:\n{context_block}\n\n"
                "Instructions:\n"
                "Identify all distinct sets of Population, Intervention, Comparator, and Outcomes (PICOs) in the above text.\n"
                "List multiple PICOs if needed. Output valid JSON ONLY in the form:\n"
                "{\n"
                f"  \"Country\": \"{country}\",\n"
                "  \"PICOs\": [\n"
                "    {\"Population\":\"...\", \"Intervention\":\"...\", \"Comparator\":\"...\", \"Outcomes\":\"...\"}\n"
                "    <!-- more if multiple PICOs -->\n"
                "  ]\n"
                "}\n"
                "Nothing else. Only JSON."
            )
            user_msg = HumanMessage(content=user_prompt)

            # LLM call
            try:
                llm_response: BaseMessage = llm_to_use([system_msg, user_msg])
            except Exception as exc:
                print(f"LLM call failed for {country}: {exc}")
                continue

            answer_text = getattr(llm_response, 'content', str(llm_response))

            # Attempt to parse JSON
            try:
                parsed_json = json.loads(answer_text)
            except json.JSONDecodeError:
                fix_msg = HumanMessage(content="Please correct and return valid JSON in the specified format only.")
                try:
                    fix_response = llm_to_use([system_msg, user_msg, fix_msg])
                    fix_text = getattr(fix_response, 'content', str(fix_response))
                    parsed_json = json.loads(fix_text)
                except Exception as parse_err:
                    print(f"Failed to parse JSON for {country}: {parse_err}")
                    continue

            # Save and append result
            if isinstance(parsed_json, dict):
                parsed_json["Country"] = country
                results.append(parsed_json)
                outpath = os.path.join("results", f"picos_{country}.json")
                with open(outpath, "w") as f:
                    json.dump(parsed_json, f, indent=2)
            else:
                wrapped_json = {
                    "Country": country,
                    "PICOs": parsed_json if isinstance(parsed_json, list) else []
                }
                results.append(wrapped_json)
                outpath = os.path.join("results", f"picos_{country}.json")
                with open(outpath, "w") as f:
                    json.dump(wrapped_json, f, indent=2)

        return results
