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

# Print all detected headings in the translated documents
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

### 4. RETRIEVER 
# Define the retriever
retriever = ChunkRetriever(vectorstore=vectorstore_biobert)

# Define the system prompt
SYSTEM_PROMPT = (
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

# Define the user prompt template
USER_PROMPT_TEMPLATE = (
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

# Define an enhanced query focused specifically on comparators and populations
query = """
In the context of this HTA submission, identify all medicines mentioned, including:
1. The medicine under evaluation (submission medicine)
2. ALL possible comparator treatments (alternative interventions, control arms, standard-of-care options, placebos, or therapies described as being 'compared to' or 'versus' the main medicine)
3. For each medicine, identify the specific intended population with details about disease severity, prior therapy requirements, biomarkers, and any relevant inclusion/exclusion criteria.
"""

# Instantiate your PICOExtractor with the original prompts
pico_extractor = PICOExtractor(
    chunk_retriever=retriever,
    system_prompt=SYSTEM_PROMPT,
    user_prompt_template=USER_PROMPT_TEMPLATE,
    model_name="gpt-4o-mini"
)


# Then run the extraction with the enhanced method
extracted_picos_cot = pico_extractor.extract_picos(
    countries=["EN", "DE", "FR", "PO"],  
    query=query,
    initial_k=30,
    final_k=15,  # Increased to get more context
    heading_keywords=["comparator", "alternative", "therapy"]  # Extended keywords
)



# Test retrieval process
test_results = retriever.test_retrieval(
    query="In the context of this HTA submission, identify all comparator treatments.",
    countries=["EN", "DE"],  # Test with fewer countries for quicker results
    heading_keywords=["comparator", "alternative"],
    drug_keywords=["docetaxel", "nintedanib"],
    initial_k=20,
    final_k=10
)