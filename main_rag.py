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
from python.process_pdf import PDFProcessor
from python.process_pdf import Translator
from python.vectorisation import Chunker
from python.vectorisation import Vectoriser


from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Show folder structure:
tree = FolderTree(root_path=".", show_hidden=False, max_depth=None)
tree.generate()



##### 1. DEFINE PARAMETERS
# For openai API
openai = OpenAI()
model = "gpt-4o-mini"  # Replace with your model



##### 2. PDF PROCESSING
# prerquisites
input_pdf_dir = "data/PDF"  # Ensure this folder exists
output_text_dir = "data/text"
output_translated_dir = "data/text_translated"

# Call the base function that processes all PDFs in the folder
PDFProcessor.process_pdfs(input_pdf_dir, output_text_dir)

# Step 2. Translate text to English
translator = Translator(output_text_dir, output_translated_dir)
translator.translate_documents()



### 3. CHUNKING & VECTORISATION
# Run the chunker
chunker = Chunker(
    json_folder_path="./data/text/nsclc_kras_g12c",
    output_dir="./data/text_chunked",
    chunk_size=600,
    chunk_overlap=100
)
chunker.run_pipeline()

# Run the vectoriser
vectoriser = Vectoriser(
    chunked_folder_path="./data/text_chunked",
    db_name="my_vector_db"
)
vectoriser.run_pipeline()



# Optional: visualize vectorstore
vectorisation.visualize_vectorstore(vectorstore)


# load
vectorstore = Chroma(persist_directory=db_name, embedding_function=embeddings)

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



# --- Function to run the extraction using RAG ---
def run_extraction_per_country(query, vector_store, prompt_template, countries, k):
    from langchain.chat_models import ChatOpenAI
    from langchain.chains import LLMChain
    from langchain.chains.combine_documents.stuff import StuffDocumentsChain
    
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    # Dictionary to store each country’s result
    results_per_country = {}
    
    for country in countries:
        # Each country gets its own retriever
        retriever = vector_store.as_retriever(
            search_kwargs={
                "k": k,
                "filter": {"country": country}
            }
        )
        
        # Retrieve the top K docs for just this country
        retrieved_for_country = retriever.get_relevant_documents(query)
        
        print(f"--- Retrieved {len(retrieved_for_country)} docs for {country} ---")
        for i, doc in enumerate(retrieved_for_country):
            print(f"Doc {i+1} (country={country}): {doc.page_content}...")
        
        # Build a StuffDocumentsChain and run it for the current country
        stuff_chain = StuffDocumentsChain(
            llm_chain=LLMChain(llm=llm, prompt=prompt_template),
            document_variable_name="context"
        )
        
        country_output = stuff_chain.run(
            input_documents=retrieved_for_country,
            question=query
        )
        
        # Store this country’s result
        results_per_country[country] = country_output

    return results_per_country

# --- Updated System Prompt ---
system_prompt_1 = (
    """
    You are a highly experienced clinical information extraction expert. You have been provided with chunks of clinical guidelines on non-small cell lung cancer with KRAS G12C mutuation from various European countries in different languages.Your task is to extract all medicines, treatments, and therapies that can be considered relevant comparators for an HTA study, along with the corresponding populations for which these treatments may be used, for each country.

    Your answer must be a single table in valid Markdown with three columns: “Country”, “Comparator”, and “Population details.” For country, write the country that is defined in the metadata of the document. For each comparator, create a separate row. If no population details are specified, write ‘No specific details provided.’ Use only the information in the provided context. Do not add extra commentary, explanations, or any text outside of the table. Be as complete as possible.
    """
)

# --- Wrap the system prompt in a LangChain PromptTemplate ---
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=system_prompt_1 + "\n\nContext:\n{context}\n\nQuestion:\n{question}"
)

# Define your query
query = (
    """
    Extract all relevant comparators for an HTA study on non-small cell lung cancer with KRAS G12C mutuation for each European country, along with exact patient population details. Present your findings as instructed, in a single Markdown table with columns for “Country”, “Comparator”, and “Population”
    """
)

# Run the extraction for each country
extraction_results = run_extraction_per_country(
    query=query,
    vector_store=vector_store,
    prompt_template=prompt_template,
    countries=["NL", "EN", "SE", "DE"],
    k=10
)

# Now you have a separate output for each country.
# Decide how you want to handle these separate outputs.
# For example, you can write each one to a separate file:
for country, result in extraction_results.items():
    filename = f"results/output_{country}.md"
    with open(filename, "w") as f:
        f.write(result)

