##### 0. PRELIMINARIES
# Standard library imports
import sys
import os
import json

# Add the project directory to sys.path to ensure function imports
project_dir = os.getcwd() # Get the current working directory (project root)
if project_dir not in sys.path: 
    sys.path.append(project_dir)

# LLM related imports
from openai import OpenAI

# Local imports
from python.utils import FolderTree
from python.utils import HeadingPrinter
from python.process import PDFProcessor
from python.process import Translator
from python.vectorise import Chunker
from python.vectorise import Vectoriser
from python.retrieve import ChunkRetriever
from python.open_ai import validate_api_key
from python.retrieve import PICOExtractor

##### 0. UTILS
# Show folder structure:
tree = FolderTree(root_path=".", show_hidden=False, max_depth=None)
tree.generate()

# Print all dettectedheadings in the translated documents
#printer = HeadingPrinter()
#printer.print_all_headings()



##### 1. DEFINE PARAMETERS
# For openai API
openai = OpenAI()
model = "gpt-4o-mini"  # Replace with your model
embedding = "openai"  # Replace with your embedding

#open ai key
message = validate_api_key() # Read in, and validate the OpenAI API key.
print(message) # Print the validation message.



##### 2. PDF PROCESSING
# prerquisites
path_pdf = "data/PDF"  # Ensure this folder exists
path_clean = "data/text_cleaned"
path_translated = "data/text_translated"
path_chunked = "data/text_chunked"
path_vectorstore = "data/vectorstore"

# Call the base function that processes all PDFs in the folder
PDFProcessor.process_pdfs(path_pdf, path_clean)

# Step 2. Translate text to English
translator = Translator(path_clean, path_translated)
translator.translate_documents()



### 3. CHUNKING & VECTORISATION
# Initialise and run the chunker
chunker = Chunker(
    json_folder_path = path_translated,
    output_dir = path_chunked,
    chunk_size=1000,
    chunk_overlap=200
)
chunker.run_pipeline()

# Initialise and run the vectorizer
vectoriser_openai = Vectoriser(
    chunked_folder_path = path_chunked,
    embedding_choice = embedding,
    db_parent_dir = path_vectorstore
)
vectorstore = vectoriser_openai.run_pipeline()

# Then pass the actual vectorstore object to visualize:
vectoriser_openai.visualize_vectorstore(vectorstore)

# Test the retrieval
#TestRetrieval(vectorstore).similarity_search("lung cancer", k=5)



### 4. RETRIEVER 

# Define the system prompt
SYSTEM_PROMPT = (
    "You are a medical expert assisting with evidence synthesis. "
    "When given a context of Health Technology Assessment documents, carefully read and analyze them step-by-step. "
    "Explicitly identify and clearly distinguish the following elements: Population, Intervention, Comparator, and Outcomes (PICO). "
    "If multiple PICOs are identified, list each separately. "
    "Respond only in valid JSON format, strictly following the provided structure."
)

# Define your query
retriever = ChunkRetriever(vectorstore=vectorstore)


def test_retriever():
    # your Chroma store presumably has loaded docs that mention docetaxel, etc.
    my_retriever = ChunkRetriever(vectorstore=vectorstore)

    query = "What is the comparator for this therapy?"  # a typical user query
    countries = ["DE"]

    heading_keywords = ["Comparator", "Comparison", "Appropriate comparator therapy"]
    drug_keywords = [
        "docetaxel", 
        "pemetrexed", 
        "nivolumab", 
        "pembrolizumab", 
        "atezolizumab", 
        "ramucirumab", 
        "nintedanib"
    ]

    results = my_retriever.retrieve_pico_chunks(
        query=query,
        countries=countries,
        heading_keywords=heading_keywords,
        drug_keywords=drug_keywords,
        initial_k=30,  # can increase further if needed
        final_k=10
    )

    for country, docs in results.items():
        print(f"\n=== Results for country: {country} ===")
        for i, d in enumerate(docs, start=1):
            print(f"Document #{i}:")
            print(f"  Heading: {d['metadata'].get('heading', 'N/A')}")
            snippet = d["text"].replace("\n", " ")
            print(f"  Text (snippet): {snippet}...")
            print(f"  Metadata: {d['metadata']}")
            print("-----")
            
test_retriever()



# Instantiate your PICOExtractor
pico_extractor = PICOExtractor(
    chunk_retriever=retriever,
    system_prompt=SYSTEM_PROMPT
)

# Define your debug input
countries_to_debug = ["EN", "DE", "FR", "PO"]  # Replace with actual country codes relevant to your data
query = "What is the population and intervention for this HTA?"  # Replace with your real retrieval query
k = 8  # Number of chunks to retrieve
heading_keywords = None  # Optional, or set to None

# Run the debug retrieval
retrieved_chunks = pico_extractor.debug_retrieve_chunks(
    countries=countries_to_debug,
    query=query,
    k=k,
    heading_keywords=heading_keywords
)


