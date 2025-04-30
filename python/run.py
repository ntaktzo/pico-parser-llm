import sys
import os
import json
from typing import List, Dict, Any, Optional, Union

# Add the project directory to sys.path to ensure function imports
project_dir = os.getcwd()  # Get the current working directory (project root)
if project_dir not in sys.path:
    sys.path.append(project_dir)

# Import necessary components
from python.utils import FolderTree, HeadingPrinter
from python.process import PDFProcessor, Translator
from python.vectorise import Chunker, Vectoriser
from python.retrieve import ChunkRetriever, PICOExtractor
from python.open_ai import validate_api_key

# LLM related imports
from openai import OpenAI


class RagHTASubmission:
    """
    A class to manage the entire RAG (Retrieval-Augmented Generation) pipeline:
    1. PDF processing
    2. Translation
    3. Chunking
    4. Vectorization
    5. Retrieval
    6. PICO extraction

    Enhanced to support different source types (HTA submissions vs clinical guidelines).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        pdf_path: str = "data/PDF",
        clean_path: str = "data/text_cleaned",
        translated_path: str = "data/text_translated",
        chunked_path: str = "data/text_chunked",
        vectorstore_path: str = "data/vectorstore",
        chunk_size: int = 600,
        chunk_overlap: int = 200,
        chunk_strategy: str = "semantic",
        vectorstore_type: str = "biobert"
    ):
        """
        Initialize the RAG system with customizable parameters.

        Args:
            model: OpenAI model to use
            pdf_path: Path to the directory containing PDFs
            clean_path: Path to store cleaned text
            translated_path: Path to store translated text
            chunked_path: Path to store chunked text
            vectorstore_path: Path to store vector embeddings
            chunk_size: Size of chunks for splitting documents
            chunk_overlap: Overlap between chunks
            chunk_strategy: Chunking strategy ("semantic" or "recursive")
            vectorstore_type: Type of vectorstore to use ("openai", "biobert", or "both")
        """
        # Store parameters
        self.model = model
        self.path_pdf = pdf_path
        self.path_clean = clean_path
        self.path_translated = translated_path
        self.path_chunked = chunked_path
        self.path_vectorstore = vectorstore_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_strategy = chunk_strategy
        self.vectorstore_type = vectorstore_type

        # Initialize OpenAI client
        self.openai = OpenAI()

        # Initialize components to None (will be created as needed)
        self.translator = None
        self.chunker = None
        self.vectoriser_openai = None
        self.vectoriser_biobert = None
        self.vectorstore_openai = None
        self.vectorstore_biobert = None
        self.retriever = None
        self.pico_extractor_hta = None
        self.pico_extractor_clinical = None

        # Define system prompts for different source types
        self.system_prompt_hta = (
            "You are a medical expert assisting with evidence synthesis for Health Technology Assessment documents. "
            "Carefully review the provided context about Health Technology Assessment submissions. "
            "You must find all relevant Population, Intervention, Comparator, and Outcomes (PICO) information. "
            "IMPORTANT: For this specific task, do NOT consider Sotorasib as the 'Intervention' in the PICO framework. "
            "Instead, identify each treatment option mentioned in the documents as a separate PICO entry. "
            "Fill the 'Intervention' field with a placeholder like 'New medicine under assessment'. "
            "Treatments such as Sotorasib, docetaxel, or any other treatment options should be listed in the 'Comparator' field. "
            "Use step-by-step reasoning internally (chain-of-thought) to identify explicit or implicit comparators, "
            "including control groups, placebo, 'versus' statements, standard of care, or alternative therapies. "
            "For HTA submissions, focus specifically on the comparators that are officially considered in the assessment process. "
            "However, do NOT reveal your chain-of-thought in the final answer. "
            "Your final output must be valid JSON only, strictly following the requested format. "
            "No additional commentary or explanation outside the JSON."
        )
        
        self.system_prompt_clinical = (
            "You are a medical expert assisting with evidence synthesis for clinical guidelines. "
            "Carefully review the provided context about clinical practice guidelines. "
            "You must find all relevant Population, Intervention, Comparator, and Outcomes (PICO) information. "
            "IMPORTANT: For this specific task, extract each distinct treatment recommendation from the clinical guidelines. "
            "Each recommendation should be a separate PICO entry. "
            "For clinical guidelines, focus on identifying the current standard of care treatments and their indications. "
            "Capture all treatment options mentioned for specific patient populations, including first-line, second-line, and subsequent therapies. "
            "Pay special attention to biomarkers, staging information, and patient characteristics that influence treatment decisions. "
            "Use step-by-step reasoning internally (chain-of-thought) to identify these recommendations. "
            "However, do NOT reveal your chain-of-thought in the final answer. "
            "Your final output must be valid JSON only, strictly following the requested format. "
            "No additional commentary or explanation outside the JSON."
        )

        # Define user prompt templates for different source types
        self.user_prompt_template_hta = (
            "Context:\n{context_block}\n\n"
            "Instructions:\n"
            "Identify all distinct sets of Population, Comparator, and Outcomes in the above HTA submission text.\n"
            "For each comparator, create a separate PICO entry. Do NOT include Sotorasib as the 'Intervention'.\n"
            "Instead, leave the 'Intervention' field as 'New medicine under assessment' and include all treatment options "
            "(including Sotorasib) as separate PICOs with different comparators.\n"
            "Output valid JSON ONLY in the following form:\n"
            "{{\n"
            "  \"Country\": \"{country}\",\n"
            "  \"PICOs\": [\n"
            "    {{\"Population\":\"...\", \"Intervention\":\"New medicine under assessment\", \"Comparator\":\"...\", \"Outcomes\":\"...\"}},\n"
            "    <!-- more if multiple PICOs -->\n"
            "  ]\n"
            "}}\n"
            "Nothing else. **Only JSON**, no extra text."
        )
        
        self.user_prompt_template_clinical = (
            "Context:\n{context_block}\n\n"
            "Instructions:\n"
            "Analyze this clinical guideline text and extract all treatment recommendations in PICO format.\n"
            "For each distinct treatment recommendation, create a separate PICO entry.\n"
            "Be specific about patient populations, including biomarkers, staging, and prior therapy status.\n"
            "For each recommendation, clearly identify:\n"
            "1. The specific patient population (Population)\n"
            "2. The recommended treatment (Intervention)\n"
            "3. Alternative treatments or comparators mentioned (Comparator)\n"
            "4. Expected outcomes or goals of therapy (Outcomes)\n"
            "Output valid JSON ONLY in the following form:\n"
            "{{\n"
            "  \"Country\": \"{country}\",\n"
            "  \"PICOs\": [\n"
            "    {{\"Population\":\"...\", \"Intervention\":\"...\", \"Comparator\":\"...\", \"Outcomes\":\"...\"}},\n"
            "    <!-- more if multiple PICOs -->\n"
            "  ]\n"
            "}}\n"
            "Nothing else. **Only JSON**, no extra text."
        )

        # Default queries for different source types
        self.default_query_hta = """
        In the context of this HTA submission, identify all medicines mentioned, including:
        1. The medicine under evaluation (submission medicine)
        2. ALL possible comparator treatments (alternative interventions, control arms, standard-of-care options, placebos, or therapies described as being 'compared to' or 'versus' the main medicine)
        3. For each medicine, identify the specific intended population with details about disease severity, prior therapy requirements, biomarkers, and any relevant inclusion/exclusion criteria.
        """
        
        self.default_query_clinical = """
        In this clinical guideline, identify all treatment recommendations, including:
        1. First-line, second-line, and subsequent treatment options
        2. Specific patient populations for each treatment (by biomarker status, disease stage, prior therapy)
        3. Treatment algorithms or decision trees for different patient populations
        4. Comparative information between different treatment options, including efficacy and safety profiles
        """

    def show_folder_structure(self, root_path: str = ".", show_hidden: bool = False, max_depth: Optional[int] = None):
        """Show the folder structure of the project."""
        tree = FolderTree(root_path=root_path, show_hidden=show_hidden, max_depth=max_depth)
        tree.generate()

    def print_all_headings(self):
        """Print all detected headings in the translated documents."""
        printer = HeadingPrinter()
        printer.print_all_headings()

    def validate_api_key(self):
        """Validate the OpenAI API key."""
        message = validate_api_key()  # Read in, and validate the OpenAI API key.
        print(message)  # Print the validation message.
        return message

    def process_pdfs(self):
        """Process PDFs to extract cleaned text."""
        # Ensure directories exist
        os.makedirs(self.path_clean, exist_ok=True)
        
        # Process PDFs
        PDFProcessor.process_pdfs(self.path_pdf, self.path_clean)
        print(f"Processed PDFs from {self.path_pdf} to {self.path_clean}")

    def translate_documents(self):
        """Translate cleaned text to English."""
        # Ensure directories exist
        os.makedirs(self.path_translated, exist_ok=True)
        
        # Initialize translator if not already done
        if self.translator is None:
            self.translator = Translator(self.path_clean, self.path_translated)
        
        # Translate documents
        self.translator.translate_documents()
        print(f"Translated documents from {self.path_clean} to {self.path_translated}")

    def chunk_documents(self):
        """Chunk translated documents for vectorization."""
        # Ensure directories exist
        os.makedirs(self.path_chunked, exist_ok=True)
        
        # Initialize chunker
        self.chunker = Chunker(
            json_folder_path=self.path_translated,
            output_dir=self.path_chunked,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            chunk_strat=self.chunk_strategy,
            maintain_folder_structure=True
        )
        
        # Run chunking pipeline
        self.chunker.run_pipeline()
        print(f"Chunked documents from {self.path_translated} to {self.path_chunked}")

    def vectorize_documents(self, embeddings_type: Optional[str] = None):
        """
        Vectorize chunked documents using specified embedding type.
        
        Args:
            embeddings_type: Type of embeddings to use ("openai", "biobert", or "both")
                           If None, uses the value specified in self.vectorstore_type
        """
        # Use class-level vectorstore_type if embeddings_type is not specified
        if embeddings_type is None:
            embeddings_type = self.vectorstore_type
            
        # Ensure directories exist
        os.makedirs(self.path_vectorstore, exist_ok=True)
        
        if embeddings_type.lower() in ["openai", "both"]:
            # Create OpenAI vectorstore
            self.vectoriser_openai = Vectoriser(
                chunked_folder_path=self.path_chunked,
                embedding_choice="openai",
                db_parent_dir=self.path_vectorstore
            )
            self.vectorstore_openai = self.vectoriser_openai.run_pipeline()
            print("Created OpenAI vectorstore")
            
        if embeddings_type.lower() in ["biobert", "both"]:
            # Create BioBERT vectorstore
            self.vectoriser_biobert = Vectoriser(
                chunked_folder_path=self.path_chunked,
                embedding_choice="biobert",
                db_parent_dir=self.path_vectorstore
            )
            self.vectorstore_biobert = self.vectoriser_biobert.run_pipeline()
            print("Created BioBERT vectorstore")
            
        # Visualize vectorstore if both are available
        if embeddings_type.lower() == "both" and self.vectoriser_openai and self.vectoriser_biobert:
            print("Visualizing vectorstore comparison")
            self.vectoriser_openai.visualize_vectorstore(self.vectorstore_biobert)

    def initialize_retriever(self, vectorstore_type: Optional[str] = None):
        """
        Initialize the retriever with the specified vectorstore.
        
        Args:
            vectorstore_type: Type of vectorstore to use ("openai" or "biobert")
                            If None, uses the value specified in self.vectorstore_type
        """
        # Use class-level vectorstore_type if vectorstore_type is not specified
        if vectorstore_type is None:
            vectorstore_type = self.vectorstore_type
            
        if vectorstore_type.lower() == "openai" and self.vectorstore_openai:
            self.retriever = ChunkRetriever(vectorstore=self.vectorstore_openai)
        elif vectorstore_type.lower() == "biobert" and self.vectorstore_biobert:
            self.retriever = ChunkRetriever(vectorstore=self.vectorstore_biobert)
        else:
            if vectorstore_type.lower() == "openai":
                print("OpenAI vectorstore not available. Please run vectorize_documents first.")
            else:
                print("BioBERT vectorstore not available. Please run vectorize_documents first.")
            return
        
        print(f"Initialized retriever with {vectorstore_type} vectorstore")

    def initialize_pico_extractors(self):
        """Initialize separate PICO extractors for HTA and clinical guideline sources."""
        if self.retriever is None:
            print("Retriever not initialized. Please run initialize_retriever first.")
            return
            
        # HTA Submissions extractor
        self.pico_extractor_hta = PICOExtractor(
            chunk_retriever=self.retriever,
            system_prompt=self.system_prompt_hta,
            user_prompt_template=self.user_prompt_template_hta,
            model_name=self.model
        )
        
        # Clinical Guidelines extractor
        self.pico_extractor_clinical = PICOExtractor(
            chunk_retriever=self.retriever,
            system_prompt=self.system_prompt_clinical,
            user_prompt_template=self.user_prompt_template_clinical,
            model_name=self.model
        )
        
        print(f"Initialized PICO extractors for both source types with model {self.model}")

    def extract_picos_by_source_type(
        self,
        countries: List[str],
        source_type: str = "hta_submission",
        query: Optional[str] = None,
        initial_k: int = 30,
        final_k: int = 15,
        heading_keywords: Optional[List[str]] = None
    ):
        """
        Extract PICOs from the specified countries and source type.
        
        Args:
            countries: List of country codes to extract PICOs from
            source_type: Source type to extract from ("hta_submission" or "clinical_guideline")
            query: Query to use for retrieval (defaults to source-specific default query)
            initial_k: Initial number of documents to retrieve
            final_k: Final number of documents to use after filtering
            heading_keywords: Keywords to look for in document headings
        
        Returns:
            List of extracted PICOs
        """
        # Initialize extractors if not already done
        if self.pico_extractor_hta is None or self.pico_extractor_clinical is None:
            self.initialize_pico_extractors()
            
        # Set source-specific defaults
        if source_type == "hta_submission":
            extractor = self.pico_extractor_hta
            default_query = self.default_query_hta
            default_headings = ["comparator", "alternative", "therapy"]
            output_prefix = "hta"
        elif source_type == "clinical_guideline":
            extractor = self.pico_extractor_clinical
            default_query = self.default_query_clinical
            default_headings = ["recommendation", "treatment", "therapy", "algorithm"]
            output_prefix = "clinical"
        else:
            raise ValueError(f"Unsupported source_type: {source_type}")
            
        # Use defaults if not specified
        if query is None:
            query = default_query
            
        if heading_keywords is None:
            heading_keywords = default_headings
            
        # Create a filter for the source type
        source_filter = {"source_type": source_type}
        
        # Extract PICOs with source type filter
        extracted_picos = []
        
        # Ensure the output directory exists
        os.makedirs("results", exist_ok=True)
        
        for country in countries:
            # Get document chunks for this country and source type
            results_dict = extractor.chunk_retriever.retrieve_pico_chunks(
                query=query,
                countries=[country],
                heading_keywords=heading_keywords,
                initial_k=initial_k,
                final_k=final_k,
                source_filter=source_filter  # Add source type filtering
            )
            
            country_chunks = results_dict.get(country, [])
            if not country_chunks:
                print(f"No {source_type} chunks found for country {country}")
                continue
            
            # Process chunks with context manager
            processed_chunks = extractor.context_manager.process_chunks(country_chunks)
            context_block = extractor.context_manager.build_optimal_context(processed_chunks)
            
            # Prepare system and user messages
            system_msg = extractor.system_message
            user_msg_text = extractor.user_prompt_template.format(
                context_block=context_block,
                country=country
            )
            user_msg = extractor.user_message(user_msg_text)
            
            # LLM call
            try:
                llm_response = extractor.llm([system_msg, user_msg])
            except Exception as exc:
                print(f"LLM call failed for {country}: {exc}")
                continue
            
            answer_text = getattr(llm_response, 'content', str(llm_response))
            
            # Parse JSON response
            try:
                parsed_json = json.loads(answer_text)
            except json.JSONDecodeError:
                # Retry once with explicit instruction to fix JSON
                fix_msg = extractor.user_message("Please correct and return valid JSON in the specified format only.")
                try:
                    fix_response = extractor.llm([system_msg, user_msg, fix_msg])
                    fix_text = getattr(fix_response, 'content', str(fix_response))
                    parsed_json = json.loads(fix_text)
                except Exception as parse_err:
                    print(f"Failed to parse JSON for {country}: {parse_err}")
                    continue
            
            # Save results
            if isinstance(parsed_json, dict):
                parsed_json["Country"] = country  # Ensure correct country
                parsed_json["SourceType"] = source_type  # Add source type
                extracted_picos.append(parsed_json)
                outpath = os.path.join("results", f"{output_prefix}_picos_{country}.json")
                with open(outpath, "w", encoding="utf-8") as f:
                    json.dump(parsed_json, f, indent=2, ensure_ascii=False)
            else:
                # Handle non-dict response
                wrapped_json = {
                    "Country": country,
                    "SourceType": source_type,
                    "PICOs": parsed_json if isinstance(parsed_json, list) else []
                }
                extracted_picos.append(wrapped_json)
                outpath = os.path.join("results", f"{output_prefix}_picos_{country}.json")
                with open(outpath, "w", encoding="utf-8") as f:
                    json.dump(wrapped_json, f, indent=2, ensure_ascii=False)
        
        print(f"Extracted PICOs for {source_type} from countries: {', '.join(countries)}")
        return extracted_picos

    def extract_picos_hta(
        self,
        countries: List[str],
        query: Optional[str] = None,
        initial_k: int = 30,
        final_k: int = 15,
        heading_keywords: Optional[List[str]] = None
    ):
        """Extract PICOs specifically from HTA submissions."""
        return self.extract_picos_by_source_type(
            countries=countries,
            source_type="hta_submission",
            query=query,
            initial_k=initial_k,
            final_k=final_k,
            heading_keywords=heading_keywords
        )

    def extract_picos_clinical(
        self,
        countries: List[str],
        query: Optional[str] = None,
        initial_k: int = 30,
        final_k: int = 15,
        heading_keywords: Optional[List[str]] = None
    ):
        """Extract PICOs specifically from clinical guidelines."""
        return self.extract_picos_by_source_type(
            countries=countries,
            source_type="clinical_guideline",
            query=query,
            initial_k=initial_k,
            final_k=final_k,
            heading_keywords=heading_keywords
        )

    # Original method kept for backward compatibility
    def extract_picos(
        self,
        countries: List[str],
        query: Optional[str] = None,
        initial_k: int = 30,
        final_k: int = 15,
        heading_keywords: Optional[List[str]] = None
    ):
        """
        Extract PICOs from the specified countries (both source types).
        
        Args:
            countries: List of country codes to extract PICOs from
            query: Query to use for retrieval (defaults to self.default_query_hta)
            initial_k: Initial number of documents to retrieve
            final_k: Final number of documents to use after filtering
            heading_keywords: Keywords to look for in document headings
        
        Returns:
            List of extracted PICOs
        """
        # For backward compatibility, use the HTA extractor
        if self.pico_extractor_hta is None:
            self.initialize_pico_extractors()
            
        if query is None:
            query = self.default_query_hta
            
        if heading_keywords is None:
            heading_keywords = ["comparator", "alternative", "therapy"]
            
        extracted_picos = self.pico_extractor_hta.extract_picos(
            countries=countries,
            query=query,
            initial_k=initial_k,
            final_k=final_k,
            heading_keywords=heading_keywords
        )
        
        print(f"Extracted PICOs for countries: {', '.join(countries)}")
        return extracted_picos

    def test_retrieval(
        self,
        query: str,
        countries: List[str],
        source_type: Optional[str] = None,
        heading_keywords: Optional[List[str]] = None,
        drug_keywords: Optional[List[str]] = None,
        initial_k: int = 20,
        final_k: int = 10
    ):
        """
        Test the retrieval process.
        
        Args:
            query: Query for retrieval
            countries: List of country codes to retrieve from
            source_type: Optional source type filter (hta_submission or clinical_guideline)
            heading_keywords: Keywords to look for in document headings
            drug_keywords: Keywords for drugs to prioritize
            initial_k: Initial number of documents to retrieve
            final_k: Final number of documents to use after filtering
        
        Returns:
            Test results
        """
        if self.retriever is None:
            print("Retriever not initialized. Please run initialize_retriever first.")
            return None
            
        if heading_keywords is None:
            heading_keywords = ["comparator", "alternative", "treatment"]
            
        if drug_keywords is None:
            drug_keywords = ["docetaxel", "nintedanib"]
            
        # Create source_filter if source_type is provided
        source_filter = {"source_type": source_type} if source_type else None
            
        test_results = self.retriever.test_retrieval(
            query=query,
            countries=countries,
            heading_keywords=heading_keywords,
            drug_keywords=drug_keywords,
            initial_k=initial_k,
            final_k=final_k,
            source_filter=source_filter  # Add source filter
        )
        
        return test_results

    def run_full_pipeline_for_source(
        self,
        source_type: str,
        countries: List[str] = ["EN", "DE", "FR", "PO"],
        skip_processing: bool = False,
        skip_translation: bool = True,
        vectorstore_type: Optional[str] = None
    ):
        """
        Run the full RAG pipeline for a specific source type.
        
        Args:
            source_type: Type of source to process ("hta_submission" or "clinical_guideline")
            countries: List of country codes to extract PICOs from
            skip_processing: Skip PDF processing if True
            skip_translation: Skip translation if True
            vectorstore_type: Type of vectorstore to use ("openai", "biobert", or "both")
                            If None, uses the value specified in self.vectorstore_type
        
        Returns:
            Extracted PICOs
        """
        # Use class-level vectorstore_type if vectorstore_type is not specified
        if vectorstore_type is None:
            vectorstore_type = self.vectorstore_type
            
        # Validate API key
        self.validate_api_key()
        
        # Process PDFs
        if not skip_processing:
            self.process_pdfs()
        
        # Translate documents
        if not skip_translation:
            self.translate_documents()
        
        # Chunk documents
        self.chunk_documents()
        
        # Vectorize documents
        self.vectorize_documents(embeddings_type=vectorstore_type)
        
        # Initialize retriever
        self.initialize_retriever(vectorstore_type=vectorstore_type if vectorstore_type != "both" else "biobert")
        
        # Initialize PICO extractors
        self.initialize_pico_extractors()
        
        # Extract PICOs based on source type
        if source_type == "hta_submission":
            extracted_picos = self.extract_picos_hta(countries=countries)
        elif source_type == "clinical_guideline":
            extracted_picos = self.extract_picos_clinical(countries=countries)
        else:
            raise ValueError(f"Unsupported source_type: {source_type}")
        
        return extracted_picos