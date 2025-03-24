import os
import json
import glob
import shutil
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings

import os
import json
import glob
import shutil
import numpy as np
from typing import List
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.manifold import TSNE
import plotly.graph_objs as go

class Chunker:
    def __init__(
        self,
        json_folder_path,
        output_dir="./data/text_chunked",
        chunk_size=600,
        chunk_overlap=100
    ):
        self.json_folder_path = json_folder_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def load_json_documents(self) -> List[Document]:
        """
        Loads JSON files recursively and converts each heading-based entry
        into a preliminary LangChain Document.
        """
        documents = []
        json_files = glob.glob(os.path.join(self.json_folder_path, "**/*.json"), recursive=True)
        print(f"Found {len(json_files)} JSON files.")

        for jf in json_files:
            print(f"Processing file: {jf}")
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "doc_id" not in data or "chunks" not in data:
                print(f"Skipping file {jf}: missing 'doc_id' or 'chunks'.")
                continue

            doc_id = data["doc_id"]
            chunks = data["chunks"]

            if not isinstance(chunks, list) or len(chunks) == 0:
                print(f"Skipping file {jf}: 'chunks' is empty or not a list.")
                continue

            for c in chunks:
                if "text" not in c:
                    print(f"Skipping a chunk in {jf}: missing 'text'.")
                    continue
                heading_metadata = {
                    "doc_id": doc_id,
                    "heading": c.get("heading", ""),
                    "start_page": c.get("start_page"),
                    "end_page": c.get("end_page")
                }
                documents.append(
                    Document(
                        page_content=c["text"],
                        metadata=heading_metadata
                    )
                )

        print(f"Loaded {len(documents)} heading-level documents.")
        return documents

    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        """
        Further splits each heading-based Document if it exceeds chunk_size.
        Preserves heading metadata.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )

        all_splits = []

        for doc in docs:
            splits = text_splitter.split_text(doc.page_content)
            for i, split_text in enumerate(splits):
                sub_doc = Document(
                    page_content=split_text,
                    metadata=dict(doc.metadata)
                )
                sub_doc.metadata["split_index"] = i
                all_splits.append(sub_doc)

        print(f"Total chunks after further splitting: {len(all_splits)}")
        return all_splits

    def save_chunks_to_files(self, chunks: List[Document]):
        """
        Saves the final chunks to individual files in the output directory.
        Each file is named <doc_id>_chunked.json and contains all chunks with that doc_id.
        """
        from collections import defaultdict

        grouped = defaultdict(list)
        for doc in chunks:
            doc_id = doc.metadata.get("doc_id", "unknown_doc")
            grouped[doc_id].append({
                "text": doc.page_content,
                "metadata": doc.metadata
            })

        for doc_id, doc_chunks in grouped.items():
            out_path = os.path.join(self.output_dir, f"{doc_id}_chunked.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"doc_id": doc_id, "chunks": doc_chunks}, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(doc_chunks)} chunks to {out_path}")

    def debug_print_chunks(self, docs: List[Document], num_chunks=5):
        """
        Prints out chunks and their metadata for debugging purposes.
        """
        for i, doc in enumerate(docs[:num_chunks]):
            print(f"Chunk {i+1}:")
            print(f"Metadata: {doc.metadata}")
            print(f"Content:\n{doc.page_content[:500]}...\n{'-'*40}")

    def run_pipeline(self):
        heading_docs = self.load_json_documents()
        final_chunks = self.chunk_documents(heading_docs)
        self.save_chunks_to_files(final_chunks)



class Vectoriser:
    """
    Class to create or load a Chroma vectorstore from chunked JSON documents.
    Supports multiple embeddings (OpenAI or PubMedBERT).
    """

    def __init__(
        self,
        chunked_folder_path: str = "./data/tex_translated",
        embedding_choice: str = "openai",
        db_parent_dir: str = "./data/vectorstore"
    ):
        self.chunked_folder_path = chunked_folder_path
        self.embedding_choice = embedding_choice.lower()
        self.db_parent_dir = db_parent_dir
        self.db_name = self._get_db_path()

    def _get_db_path(self) -> str:
        """Determines the path for the vectorstore based on embedding choice."""
        store_name = {
            "openai": "open_ai_vectorstore",
            "pubmedbert": "pubmedbert_vectorstore"
        }.get(self.embedding_choice)

        if store_name is None:
            raise ValueError("Unsupported embedding_choice. Use 'openai' or 'pubmedbert'.")

        return os.path.join(self.db_parent_dir, store_name)

    def load_chunked_documents(self) -> List[Document]:
        """
        Loads chunked JSON files from the specified folder and converts them into Document objects.
        """
        documents = []
        json_files = glob.glob(os.path.join(self.chunked_folder_path, "*.json"))
        for jf in json_files:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)

            doc_id = data["doc_id"]
            chunks = data["chunks"]

            for c in chunks:
                documents.append(Document(page_content=c["text"], metadata=c["metadata"]))

        return documents

    def vectorstore_exists(self) -> bool:
        """Checks if a vectorstore already exists at the specified location."""
        return os.path.exists(self.db_name) and len(os.listdir(self.db_name)) > 0

    def get_embeddings(self):
        """Returns the embedding model based on embedding_choice."""
        if self.embedding_choice == "openai":
            return OpenAIEmbeddings()
        elif self.embedding_choice == "pubmedbert":
            return HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

    def create_vectorstore(self, docs: List[Document]):
        """
        Creates and persists a Chroma vectorstore from provided documents.
        """
        if os.path.exists(self.db_name):
            shutil.rmtree(self.db_name)
        os.makedirs(self.db_name, exist_ok=True)

        embeddings = self.get_embeddings()
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=self.db_name
        )
        vectorstore.persist()
        print(f"Chroma vectorstore created with {len(docs)} embedded chunks at {self.db_name}.")
        return vectorstore

    def load_vectorstore(self):
        """
        Loads an existing Chroma vectorstore from disk.
        """
        embeddings = self.get_embeddings()
        vectorstore = Chroma(
            persist_directory=self.db_name,
            embedding_function=embeddings
        )
        print(f"Chroma vectorstore loaded from {self.db_name}.")
        return vectorstore

    def run_pipeline(self):
        """
        Checks for existing vectorstore. If it exists, loads it. Otherwise, creates it.
        """
        if self.vectorstore_exists():
            print("Vectorstore already exists. Loading it now.")
            return self.load_vectorstore()
        else:
            print("No vectorstore found. Creating a new one.")
            docs = self.load_chunked_documents()
            return self.create_vectorstore(docs)
        
    def visualize_vectorstore(self, vectorstore):
        """
        Creates a t-SNE visualization of the vectorstore embeddings.
        Embeddings are color-coded based on the 'country' field in the metadata.
        """
        # Retrieve embeddings, documents, and metadatas from the vectorstore.
        result = vectorstore.get(include=['embeddings', 'documents', 'metadatas'])
        vectors = np.array(result['embeddings'])
        documents = result['documents']
        
        # Safely extract the 'country' from metadata; default to 'Unknown' if not present.
        doc_countries = []
        for metadata in result['metadatas']:
            if isinstance(metadata, dict):
                doc_countries.append(metadata.get('country', 'Unknown'))
            else:
                doc_countries.append('Unknown')
        
        # Define a color map for countries.
        color_map = {'SE': 'blue', 'NL': 'green', 'DE': 'red', 'EN': 'orange', 'EU': 'yellow'}
        colors = [color_map.get(country, 'grey') for country in doc_countries]

        # Use t-SNE for dimensionality reduction to 2D.
        tsne = TSNE(n_components=2, random_state=42)
        reduced_vectors = tsne.fit_transform(vectors)

        # Create an interactive scatter plot with Plotly.
        fig = go.Figure(data=[go.Scatter(
            x=reduced_vectors[:, 0],
            y=reduced_vectors[:, 1],
            mode='markers',
            marker=dict(size=5, color=colors, opacity=0.8),
            text=[f"Country: {country}<br>Text: {doc[:100]}..." 
                for country, doc in zip(doc_countries, documents)],
            hoverinfo='text'
        )])

        fig.update_layout(
            title='2D Chroma Vector Store Visualization',
            width=800,
            height=600,
            margin=dict(r=20, b=10, l=10, t=40)
        )

        fig.show()
