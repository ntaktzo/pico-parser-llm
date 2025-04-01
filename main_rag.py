##### 0. PRELIMINARIES
# Standard library imports
import sys
import os

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


from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

##### 0. UTILS
# Show folder structure:
tree = FolderTree(root_path=".", show_hidden=False, max_depth=None)
tree.generate()

# Print all dettectedheadings in the translated documents
printer = HeadingPrinter()
printer.print_all_headings()



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
    chunk_size=600,
    chunk_overlap=100
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



### 4. RETRIEVE STEP
# Define your query
retriever = ChunkRetriever(vectorstore=vectorstore)

# Retrieve comparator-related chunks explicitly:
results = retriever.retrieve_pico_chunks(
    query="Comparator therapies",
    countries=["DE", "FR", "EN", "PO"],
    heading_keywords=None,
    k=5
)

debug_dump = retriever.chroma_collection.get(limit=1, include=["metadatas"])
print(debug_dump["metadatas"])



### 5. QUERY LLM
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate




