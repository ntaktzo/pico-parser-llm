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
#translator.translate_documents()



### 3. CHUNKING & VECTORISATION
# Initialise and run the chunker
chunker = Chunker(
    json_folder_path = path_translated,
    output_dir = path_chunked,
    chunk_size=600,
    chunk_overlap=200,
    chunk_strat="semantic"  # Use "semantic" for semantic chunking
)
chunker.run_pipeline()

# Initialise and run the vectorizer
# Create OPENAI vectorstore
vectoriser_openai = Vectoriser(
    chunked_folder_path = path_chunked,
    embedding_choice = "openai",
    db_parent_dir = path_vectorstore
)

# Create BioBERT vectorstore
vectoriser_biobert = Vectoriser(
    chunked_folder_path = path_chunked,
    embedding_choice = "biobert",
    db_parent_dir = path_vectorstore
)

# create vectorstores
vectorstore_openai = vectoriser_openai.run_pipeline()
vectorstore_biobert = vectoriser_biobert.run_pipeline()

# Then pass the actual vectorstore object to visualize:
vectoriser_openai.visualize_vectorstore(vectorstore_biobert)

# Test the retrieval
#TestRetrieval(vectorstore).similarity_search("lung cancer", k=5)



### 4. RETRIEVER 
# Define the retriever
retriever = ChunkRetriever(vectorstore=vectorstore_biobert)

# Define the system prompt
SYSTEM_PROMPT = (
    "You are a medical expert assisting with evidence synthesis. "
    "Carefully review the provided context about Health Technology Assessment documents. "
    "You must find all relevant Population, Intervention, Comparator, and Outcomes (PICO) information. "
    "Use step-by-step reasoning internally (chain-of-thought) to identify implicit or explicit comparators,"
    "including control groups, placebo, 'versus' statements, standard of care, or alternative therapies. "
    "However, do NOT reveal your chain-of-thought in the final answer. "
    "Your final output must be valid JSON only, strictly following the requested format. "
    "No additional commentary or explanation outside the JSON."
    )

# Define the user prompt template
USER_PROMPT_TEMPLATE = (
    "Context:\n{context_block}\n\n"
    "Instructions:\n"
    "Identify all distinct sets of Population, Intervention, Comparator, and Outcomes (PICOs) in the above text.\n"
    "List multiple PICOs if needed. Output valid JSON ONLY in the following form:\n"
    "{{\n"
    "  \"Country\": \"{country}\",\n"
    "  \"PICOs\": [\n"
    "    {{\"Population\":\"...\", \"Intervention\":\"...\", \"Comparator\":\"...\", \"Outcomes\":\"...\"}}\n"
    "    <!-- more if multiple PICOs -->\n"
    "  ]\n"
    "}}\n"
    "Nothing else. **Only JSON**, no extra text."
)

# Define query
query = "In the context of this HTA (Health Technology Assessment) submission, identify all comparator treatments (including any alternative interventions, control arms, standard-of-care options, placebos, or therapies described as being ‘compared to’ or ‘versus’ the main intervention). For each comparator, also provide details about the intended population—such as disease severity, prior lines of therapy, and any relevant demographic or clinical inclusion/exclusion criteria."

# Instantiate your PICOExtractor
pico_extractor = PICOExtractor(
    chunk_retriever=retriever,
    system_prompt=SYSTEM_PROMPT,
    user_prompt_template=USER_PROMPT_TEMPLATE,
    model_name="gpt-4o-mini"
)

# Run the extraction
extracted_picos = pico_extractor.extract_picos(
    countries=["EN", "DE", "FR", "PO"],  
    query=query,
    initial_k=30,
    final_k=10,
    heading_keywords="comparator"
)

"""
# Define your debug input
countries_to_debug = ["EN", "DE", "FR", "PO"]  # Replace with actual country codes relevant to your data
query = "In the context of this HTA (Health Technology Assessment) submission, identify all comparator treatments (including any alternative interventions, control arms, standard-of-care options, placebos, or therapies described as being ‘compared to’ or ‘versus’ the main intervention). For each comparator, also provide details about the intended population—such as disease severity, prior lines of therapy, and any relevant demographic or clinical inclusion/exclusion criteria."
initial_k = 30  # Number of chunks to retrieve
final_k = 10  # Number of chunks to return after scoring
heading_keywords = "comparator"   # Optional, or set to None

# Run the debug retrieval
retrieved_chunks = pico_extractor.debug_retrieve_chunks(
    countries=countries_to_debug,
    query=query,
    initial_k=initial_k,
    final_k=final_k,
    heading_keywords=heading_keywords
)

pico_extractor.extract_picos(
    countries=countries_to_debug,
    query=query,
    initial_k=initial_k,
    final_k=final_k,
    heading_keywords=heading_keywords
)   
"""

