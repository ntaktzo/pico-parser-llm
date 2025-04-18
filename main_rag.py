# Local imports
from python.utils import FolderTree, HeadingPrinter
from python.process import PDFProcessor, Translator
from python.vectorise import Chunker, Vectoriser
from python.retrieve import ChunkRetriever, PICOExtractor
from python.open_ai import validate_api_key
from python.run import RAGHTASubmission


# Initialize RAG system with default parameters
rag = RagHTASubmission()

# Show folder structure
rag.show_folder_structure()

# Run full pipeline
extracted_picos = rag.run_full_pipeline(
    countries=["EN", "DE"],  # Limit to fewer countries for testing
    skip_processing=False,
    skip_translation=True
)

# Print extracted PICOs
for pico in extracted_picos:
    print(f"Country: {pico['Country']}")
    print(f"Number of PICOs: {len(pico.get('PICOs', []))}")
    print("---")