import sys
import os
import json
from typing import List, Dict, Any, Optional, Union

# Add the project directory to sys.path to ensure function imports
project_dir = os.getcwd()  # Get the current working directory (project root)
if project_dir not in sys.path:
    sys.path.append(project_dir)

# LLM related imports
from openai import OpenAI


class RAGHTASubmission:
    """
    A class to manage the entire RAG (Retrieval-Augmented Generation) pipeline:
    1. PDF processing
    2. Translation
    3. Chunking
    4. Vectorization
    5. Retrieval
    6. PICO extraction
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
        chunk_strategy: str = "semantic"
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
        self.pico_extractor = None

        # Define system prompt for PICO extraction
        self.system_prompt = (
            "You are a medical expert assisting with evidence synthesis. "
            "Carefully review the provided context about Health Technology Assessment documents. "
            "You must find all relevant Population, Intervention, Comparator, and Outcomes (PICO) information. "
            "IMPORTANT: For this specific task, do NOT consider Sotorasib as the 'Intervention' in the PICO framework. "
            "Instead, identify each treatment option mentioned in the documents as a separate PICO entry. "
            "Fill the 'Intervention' field with a placeholder like 'New medicine under assessment'. "
            "Treatments such as Sotorasib, docetaxel, or any other treatment options should be listed in the 'Comparator' field. "
            "Use step-by-step reasoning internally (chain-of-thought) to identify explicit or implicit comparators, "
            "including control groups, placebo, 'versus' statements, standard of care, or alternative therapies. "
            "However, do NOT reveal your chain-of-thought in the final answer. "
            "Your final output must be valid JSON only, strictly following the requested format. "
            "No additional commentary or explanation outside the JSON."
        )

        # Define user prompt template for PICO extraction
        self.user_prompt_template = (
            "Context:\n{context_block}\n\n"
            "Instructions:\n"
            "Identify all distinct sets of Population, Comparator, and Outcomes in the above text.\n"
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

        # Default query for PICO extraction
        self.default_query = """
        In the context of this HTA submission, identify all medicines mentioned, including:
        1. The medicine under evaluation (submission medicine)
        2. ALL possible comparator treatments (alternative interventions, control arms, standard-of-care options, placebos, or therapies described as being 'compared to' or 'versus' the main medicine)
        3. For each medicine, identify the specific intended population with details about disease severity, prior therapy requirements, biomarkers, and any relevant inclusion/exclusion criteria.
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
            chunk_strat=self.chunk_strategy
        )
        
        # Run chunking pipeline
        self.chunker.run_pipeline()
        print(f"Chunked documents from {self.path_translated} to {self.path_chunked}")

    def vectorize_documents(self, embeddings_type: str = "both"):
        """
        Vectorize chunked documents using specified embedding type.
        
        Args:
            embeddings_type: Type of embeddings to use ("openai", "biobert", or "both")
        """
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

    def initialize_retriever(self, vectorstore_type: str = "biobert"):
        """
        Initialize the retriever with the specified vectorstore.
        
        Args:
            vectorstore_type: Type of vectorstore to use ("openai" or "biobert")
        """
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

    def initialize_pico_extractor(self):
        """Initialize the PICO extractor with the current retriever."""
        if self.retriever is None:
            print("Retriever not initialized. Please run initialize_retriever first.")
            return
            
        self.pico_extractor = PICOExtractor(
            chunk_retriever=self.retriever,
            system_prompt=self.system_prompt,
            user_prompt_template=self.user_prompt_template,
            model_name=self.model
        )
        print(f"Initialized PICO extractor with model {self.model}")

    def extract_picos(
        self,
        countries: List[str],
        query: Optional[str] = None,
        initial_k: int = 30,
        final_k: int = 15,
        heading_keywords: Optional[List[str]] = None
    ):
        """
        Extract PICOs from the specified countries.
        
        Args:
            countries: List of country codes to extract PICOs from
            query: Query to use for retrieval (defaults to self.default_query)
            initial_k: Initial number of documents to retrieve
            final_k: Final number of documents to use after filtering
            heading_keywords: Keywords to look for in document headings
        
        Returns:
            List of extracted PICOs
        """
        if self.pico_extractor is None:
            print("PICO extractor not initialized. Please run initialize_pico_extractor first.")
            return []
            
        if query is None:
            query = self.default_query
            
        if heading_keywords is None:
            heading_keywords = ["comparator", "alternative", "therapy"]
            
        extracted_picos = self.pico_extractor.extract_picos(
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
            heading_keywords = ["comparator", "alternative"]
            
        if drug_keywords is None:
            drug_keywords = ["docetaxel", "nintedanib"]
            
        test_results = self.retriever.test_retrieval(
            query=query,
            countries=countries,
            heading_keywords=heading_keywords,
            drug_keywords=drug_keywords,
            initial_k=initial_k,
            final_k=final_k
        )
        
        return test_results

    def run_full_pipeline(
        self,
        countries: List[str] = ["EN", "DE", "FR", "PO"],
        skip_processing: bool = False,
        skip_translation: bool = True,
        vectorstore_type: str = "biobert"
    ):
        """
        Run the full RAG pipeline.
        
        Args:
            countries: List of country codes to extract PICOs from
            skip_processing: Skip PDF processing if True
            skip_translation: Skip translation if True
            vectorstore_type: Type of vectorstore to use ("openai", "biobert", or "both")
        
        Returns:
            Extracted PICOs
        """
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
        
        # Initialize PICO extractor
        self.initialize_pico_extractor()
        
        # Extract PICOs
        extracted_picos = self.extract_picos(countries=countries)
        
        return extracted_picos