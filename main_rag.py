# Local imports
from python.utils import FolderTree
from python.process import PDFProcessor, Translator, PostCleaner
from python.vectorise import Chunker, Vectoriser
from python.run import RagHTASubmission
from python.open_ai import validate_api_key

# Define paths
PDF_PATH = "data/PDF"
CLEAN_PATH = "data/text_cleaned"
TRANSLATED_PATH = "data/text_translated"
POST_CLEANED_PATH = "data/post_cleaned"  # New directory for cleaned translations
CHUNKED_PATH = "data/text_chunked"
VECTORSTORE_PATH = "data/vectorstore"
VECTORSTORE_TYPE = "biobert"  # Choose between "openai", "biobert", or "both"
MODEL = "gpt-4o-mini"
COUNTRIES = ["DE", "DK", "EN", "FR", "IT", "PO", "SE", "NL"]

# Validate OpenAI API key
validate_api_key()

# Show folder structure
tree = FolderTree(root_path=".")
tree.generate()

# Step 1: Process PDFs
PDFProcessor.process_pdfs(
    input_dir=PDF_PATH,
    output_dir=CLEAN_PATH
)

# Step 2: Translate documents
translator = Translator(
    input_dir=CLEAN_PATH,
    output_dir=TRANSLATED_PATH
)
translator.translate_documents()

# Step 3: Clean translated documents 
cleaner = PostCleaner(
    input_dir=TRANSLATED_PATH,
    output_dir=POST_CLEANED_PATH,  # Store cleaned files in new directory
    maintain_folder_structure=True  # Preserve folder structure
)
cleaner.clean_all_documents()

# Step 4: Chunk documents (use cleaned translations)
chunker = Chunker(
    json_folder_path=POST_CLEANED_PATH,  # Use cleaned documents as input
    output_dir=CHUNKED_PATH,
    chunk_size=600,
    chunk_overlap=200,
    chunk_strat="semantic",
    maintain_folder_structure=True  # Set to True to preserve folder structure
)
chunker.run_pipeline()

# Step 5: Vectorize documents
vectoriser = Vectoriser(
    chunked_folder_path=CHUNKED_PATH,
    embedding_choice=VECTORSTORE_TYPE,
    db_parent_dir=VECTORSTORE_PATH
)
vectorstore = vectoriser.run_pipeline()

# Step 6: Initialize RAG system for retrieval and LLM querying
rag = RagHTASubmission(
    model=MODEL,
    vectorstore_type=VECTORSTORE_TYPE
)

# Initialize the retriever with already created vectorstore
if VECTORSTORE_TYPE.lower() == "openai":
    rag.vectorstore_openai = vectorstore
elif VECTORSTORE_TYPE.lower() == "biobert":
    rag.vectorstore_biobert = vectorstore

# Initialize the retriever
rag.initialize_retriever(vectorstore_type=VECTORSTORE_TYPE)

# Initialize the PICO extractor (add this line)
rag.initialize_pico_extractor()

# Now extract PICOs
extracted_picos = rag.extract_picos(countries=COUNTRIES)

# Print extracted PICOs
for pico in extracted_picos:
    print(f"Country: {pico['Country']}")
    print(f"Number of PICOs: {len(pico.get('PICOs', []))}")
    print("---")