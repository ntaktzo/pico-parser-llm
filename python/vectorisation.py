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
import plotly.express as px

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
    Creates or loads a Chroma vectorstore from chunked JSON documents.
    Supports embeddings from OpenAI or PubMedBERT.
    """

    def __init__(
        self,
        chunked_folder_path: str = "./data/text_chunked",
        embedding_choice: str = "openai",
        db_parent_dir: str = "./data/vectorstore"
    ):
        self.chunked_folder_path = chunked_folder_path
        self.embedding_choice = embedding_choice.lower()
        self.db_parent_dir = db_parent_dir
        self.db_name = self._get_db_path()

    def _get_db_path(self) -> str:
        store_name = {
            "openai": "open_ai_vectorstore",
            "pubmedbert": "pubmedbert_vectorstore"
        }.get(self.embedding_choice)
        if store_name is None:
            raise ValueError("Unsupported embedding_choice. Use 'openai' or 'pubmedbert'.")
        return os.path.join(self.db_parent_dir, store_name)

    def load_chunked_documents(self) -> List[Document]:
        """
        Loads chunked JSON files into Document objects.
        """
        documents = []
        json_files = glob.glob(os.path.join(self.chunked_folder_path, "*.json"))
        print(f"Found {len(json_files)} JSON files in {self.chunked_folder_path}.")

        for jf in json_files:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)

            for c in data.get("chunks", []):
                text_content = c.get("text", "").strip()
                if not text_content:
                    continue
                metadata = c.get("metadata", {})
                documents.append(Document(page_content=text_content, metadata=metadata))

        print(f"Loaded {len(documents)} valid document chunks.")
        return documents

    def vectorstore_exists(self) -> bool:
        """
        Checks if vectorstore already exists.
        """
        return os.path.exists(self.db_name) and len(os.listdir(self.db_name)) > 0

    def get_embeddings(self):
        """
        Gets the embedding function based on user's choice.
        """
        if self.embedding_choice == "openai":
            return OpenAIEmbeddings()
        elif self.embedding_choice == "pubmedbert":
            return HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
        else:
            raise ValueError("Unsupported embedding_choice. Use 'openai' or 'pubmedbert'.")

    def create_vectorstore(self, docs: List[Document]):
        """
        Creates and persists a new Chroma vectorstore.
        """
        if os.path.exists(self.db_name):
            shutil.rmtree(self.db_name)
        os.makedirs(self.db_name, exist_ok=True)
        os.chmod(self.db_name, 0o755)

        embeddings = self.get_embeddings()
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=self.db_name
        )
        vectorstore.persist()

        for root, dirs, files in os.walk(self.db_name):
            for d in dirs:
                os.chmod(os.path.join(root, d), 0o755)
            for f in files:
                os.chmod(os.path.join(root, f), 0o644)

        print(f"Vectorstore created with {len(docs)} chunks at '{self.db_name}'.")
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
        print(f"Vectorstore loaded from '{self.db_name}'.")
        return vectorstore

    def run_pipeline(self):
        """
        Main pipeline method to either load or create the vectorstore.
        """
        if self.vectorstore_exists():
            print("Vectorstore exists. Loading now...")
            return self.load_vectorstore()
        else:
            print("No vectorstore found. Creating new one...")
            docs = self.load_chunked_documents()
            if not docs:
                raise ValueError("No valid documents found for vectorization.")
            return self.create_vectorstore(docs)

    def visualize_vectorstore(self, vectorstore):
        """
        Visualizes the vectorstore using t-SNE, grouped by doc_id.
        """
        result = vectorstore.get(include=['embeddings', 'documents', 'metadatas'])
        vectors = np.array(result['embeddings'])
        documents = result['documents']
        metadatas = result['metadatas']

        doc_ids = [
            md.get('doc_id', 'unknown') if isinstance(md, dict) else 'unknown'
            for md in metadatas
        ]

        unique_docs = sorted(set(doc_ids))
        colors_palette = px.colors.qualitative.Safe
        color_dict = {
            doc: colors_palette[i % len(colors_palette)]
            for i, doc in enumerate(unique_docs)
        }
        colors = [color_dict[doc] for doc in doc_ids]

        tsne = TSNE(n_components=2, random_state=42)
        reduced_vectors = tsne.fit_transform(vectors)

        fig = go.Figure()

        for doc in unique_docs:
            indices = [i for i, d in enumerate(doc_ids) if d == doc]
            fig.add_trace(go.Scatter(
                x=reduced_vectors[indices, 0],
                y=reduced_vectors[indices, 1],
                mode='markers',
                name=doc,
                marker=dict(size=6, opacity=0.8, color=color_dict[doc]),
                text=[f"Doc ID: {doc}<br>{documents[i][:150]}..." for i in indices],
                hoverinfo='text'
            ))

        fig.update_layout(
            title='2D Vectorstore Visualization (t-SNE) grouped by Document',
            legend_title="Document ID",
            width=900,
            height=700,
            margin=dict(l=20, r=20, t=50, b=20)
        )

        fig.show()