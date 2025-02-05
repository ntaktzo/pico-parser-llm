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
from preprocess.process_pdf import process_pdfs

from llm.open_ai import count_tokens
from llm.open_ai import validate_api_key
from llm.open_ai import create_batched_prompts
from llm.open_ai import query_gpt_api_batched

from preprocess.vectorization import chuncker
from preprocess.vectorization import create_vectorstore
from preprocess.vectorization import visualize_vectorstore


# Visualization


##### 1. DEFINE PARAMETERS
# For pubmed extraction
message = validate_api_key() # Read in, and validate the OpenAI API key.
print(message) # Print the validation message.

# For openai API
openai = OpenAI()
model = "gpt-4o-mini"  # Replace with your model


##### 2. PDF PROCESSING
#process_pdfs("data/PDF", "data/text")


### 3. CHUNKING
chunks = chuncker(base_path = "data/text/*/*", chunck_size = 3000, chunck_overlap = 300)


### 4. VECTORSTORE
# Create the vectorstore    
vectorstore = create_vectorstore(chunks)

# Visualize the vectorstore
visualize_vectorstore(vectorstore._collection)



### 5. QUERY LLM
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- Custom System Prompt ---
system_prompt_1 = (
    "You are a highly experienced clinical information extraction expert. You have been provided with chunks "
    "of clinical guidelines on lung cancer from various European countries in different languages. Your task is "
    "to extract all medicines and treatments that can be considered relevant comparators for an HTA study, as well as "
    "the corresponding populations for which these treatments may be used. \n\n"
    "Please provide your answer in a structured format, listing each extracted medicine or treatment and the associated "
    "population details (e.g., age group, disease stage, other eligibility criteria) if available. "
    "If no specific population details are mentioned for a treatment, note that explicitly. \n\n"
    "Your output should not contain any additional commentary or explanation—just the structured list. "
    "Focus on precision and ensure that you capture all relevant details."
)


# --- Wrap the system prompt in a LangChain PromptTemplate ---
# (Note: Depending on your version, you may need to adjust how prompts are passed into the chain.)
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=system_prompt_1 + "\n\nContext:\n{context}\n\nQuestion:\n{question}"
)

# --- Function to run the extraction using RAG ---
def run_extraction(query, vectorstore, prompt_template):
    # Initialize the language model with your chosen model (e.g., "gpt-4o-mini")
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    # Create a retriever from the vector store
    retriever = vectorstore.as_retriever()

    # Set up the RetrievalQA chain using the "stuff" chain type, which concatenates all retrieved docs
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={'prompt': prompt_template}
    )
    
    # Run the query and return the result
    result = qa_chain.run(query)
    return result

# --- Main code block ---
# For demonstration, assume that you have already created 'chunks'
# If you haven't, load or generate your document chunks here.
# Example: chunks = load_your_chunks_function()

# If vectorstore has already been created and persisted, you can load it:
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory="vector_db", embedding_function=embeddings)

# Define your query
query = (
    "Extract all medicines/treatments that can be considered relevant comparators for an HTA study "
    "and specify the corresponding patient population for which these treatments are intended."
)

# Run the extraction
extraction_result = run_extraction(query, vectorstore, prompt_template)

# Print the results
print("Extraction result:")
print(extraction_result)








