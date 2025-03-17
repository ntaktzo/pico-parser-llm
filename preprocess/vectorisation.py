import glob
import os
import shutil
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from sklearn.manifold import TSNE
import plotly.graph_objects as go

class Vectorisation:
    def __init__(self, base_path, chunk_size=600, chunk_overlap=150, db_name='vector_db'):
        self.base_path = base_path  # ✅ Corrected parameter
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.db_name = db_name

    def load_documents(self):
        """Loads documents from the specified base path."""
        text_loader_kwargs = {'encoding': 'utf-8'}
        documents = []
        disease_folders = glob.glob(self.base_path)  # ✅ Now correctly referencing base_path

        for disease_folder in disease_folders:
            disease_name = os.path.basename(os.path.dirname(disease_folder))
            country_name = os.path.basename(disease_folder)

            loader = DirectoryLoader(disease_folder, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
            folder_docs = loader.load()

            for doc in folder_docs:
                doc.metadata['disease'] = disease_name
                doc.metadata['country'] = country_name
                documents.append(doc)

        print(f"Loaded {len(documents)} documents before chunking.")
        return documents

    def create_chunks(self):
        """Splits documents into smaller chunks using RecursiveCharacterTextSplitter."""
        documents = self.load_documents()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n", ".", "-", ","]
        )
        chunks = text_splitter.split_documents(documents)

        print(f"Total chunks after chunking: {len(chunks)}")
        return chunks

    def create_vectorstore(self, chunks):
        """Creates a vector store from document chunks and saves it to disk."""
        if os.path.exists(self.db_name):
            shutil.rmtree(self.db_name)
        os.makedirs(self.db_name, exist_ok=True)

        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=self.db_name
        )

        print(f"Vectorstore created with {vectorstore._collection.count()} documents.")
        return vectorstore

    def visualize_vectorstore(self, vectorstore):
        """Creates a t-SNE visualization of the vectorstore embeddings."""
        result = vectorstore.get(include=['embeddings', 'documents', 'metadatas'])
        vectors = np.array(result['embeddings'])
        documents = result['documents']
        doc_types = [metadata['country'] for metadata in result['metadatas']]

        color_map = {'SE': 'blue', 'NL': 'green', 'DE': 'red', 'EN': 'orange', 'EU': 'yellow'}
        colors = [color_map.get(t, 'grey') for t in doc_types]

        tsne = TSNE(n_components=2, random_state=42)
        reduced_vectors = tsne.fit_transform(vectors)

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
            width=800,
            height=600,
            margin=dict(r=20, b=10, l=10, t=40)
        )

        fig.show()
