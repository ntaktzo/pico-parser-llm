# Local imports
from python.utils import FolderTree, HeadingPrinter
from python.process import PDFProcessor, Translator
from python.vectorise import Chunker, Vectoriser
from python.retrieve import ChunkRetriever, PICOExtractor
from python.open_ai import validate_api_key
from python.run import RagHTASubmission

# Initialize RAG system with custom parameters
rag = RagHTASubmission(
    model="gpt-4o-mini",
    vectorstore_type="biobert"  # Choose between "openai", "biobert", or "both"
)

# Show folder structure
rag.show_folder_structure()

# Run full pipeline
extracted_picos = rag.run_full_pipeline(
    countries=["EN", "DE", "PO", "FR"],  # Limit to fewer countries for testing
    skip_processing=True,
    skip_translation=True,
    vectorstore_type="biobert"  # Optionally override the default vectorstore type
)

# Print extracted PICOs
for pico in extracted_picos:
    print(f"Country: {pico['Country']}")
    print(f"Number of PICOs: {len(pico.get('PICOs', []))}")
    print("---")
