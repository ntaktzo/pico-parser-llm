import glob
import os
import numpy as np
import shutil

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from sklearn.manifold import TSNE
import plotly.graph_objects as go



def load_documents(base_path):
    text_loader_kwargs = {'encoding': 'utf-8'}
    documents = []

    # Iterate over disease folders
    disease_folders = glob.glob(base_path)

    for disease_folder in disease_folders:
        disease_name = os.path.basename(os.path.dirname(disease_folder))  # Extract disease name
        country_name = os.path.basename(disease_folder)  # Extract country name
        
        # Load documents from country folders
        loader = DirectoryLoader(disease_folder, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
        folder_docs = loader.load()
        
        for doc in folder_docs:
            doc.metadata["disease"] = disease_name
            doc.metadata["country"] = country_name
            documents.append(doc)

    print(f"Loaded {len(documents)} documents before chunking.")

    return documents



def chuncker(base_path, chunck_size, chunck_overlap):
    # Load the documents
    documents = load_documents(base_path)

    # Use improved text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunck_size, 
        chunk_overlap=chunck_overlap, 
        separators=["\n", ".", "-", ","]
    )

    # Apply chunking
    chunks = text_splitter.split_documents(documents)

    print(f"Total chunks after chunking: {len(chunks)}")
    
    # Find and print metadata
    doc_types = set(chunk.metadata['disease'] for chunk in chunks)
    countries = set(chunk.metadata['country'] for chunk in chunks)

    print(f"Document types: {', '.join(doc_types)}")
    print(f"Countries: {', '.join(countries)}")
    
    return chunks



def create_vectorstore(chunks, db_name="vector_db"):

    # Ensure we have write permissions
    if os.path.exists(db_name):
        shutil.rmtree(db_name)  # Force delete previous DB

    os.makedirs(db_name, exist_ok=True)

    # Embeddings
    embeddings = OpenAIEmbeddings()

    # Create vectorstore
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
    print(f"Vectorstore created with {vectorstore._collection.count()} documents")

    return vectorstore




def visualize_vectorstore(collection):
    result = collection.get(include=['embeddings', 'documents', 'metadatas'])
    vectors = np.array(result['embeddings'])
    documents = result['documents']
    doc_types = [metadata['country'] for metadata in result['metadatas']]
    colors = [['blue', 'green', 'red', 'orange'][['SE', 'NL', 'DE', 'EN'].index(t)] for t in doc_types]

    # Reduce the dimensionality of the vectors to 2D using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    reduced_vectors = tsne.fit_transform(vectors)

    # Create the 2D scatter plot
    fig = go.Figure(data=[go.Scatter(
        x=reduced_vectors[:, 0],
        y=reduced_vectors[:, 1],
        mode='markers',
        marker=dict(size=5, color=colors, opacity=0.8),
        text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
        hoverinfo='text'
    )])

    fig.update_layout(
        title='2D Chroma Vector Store Visualization',
        scene=dict(xaxis_title='x', yaxis_title='y'),
        width=800,
        height=600,
        margin=dict(r=20, b=10, l=10, t=40)
    )

    fig.show()
