import os
import json
import glob
import shutil
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant


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
    def __init__(
        self,
        chunked_folder_path="./data/text_chunked",
        db_name="vector_db"
    ):
        self.chunked_folder_path = chunked_folder_path
        self.db_name = db_name

    def load_chunked_documents(self) -> List[Document]:
        """
        Loads chunked JSON files from the specified folder and converts them to Document objects.
        """
        documents = []
        json_files = glob.glob(os.path.join(self.chunked_folder_path, "*.json"))
        print(f"Found {len(json_files)} chunked JSON files.")

        for jf in json_files:
            print(f"Loading chunked file: {jf}")
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "doc_id" not in data or "chunks" not in data:
                print(f"Skipping file {jf}: missing 'doc_id' or 'chunks'.")
                continue

            doc_id = data["doc_id"]
            chunks = data["chunks"]

            for c in chunks:
                if "text" not in c or "metadata" not in c:
                    print(f"Skipping a chunk in {jf}: missing 'text' or 'metadata'.")
                    continue
                documents.append(Document(page_content=c["text"], metadata=c["metadata"]))

        print(f"Loaded {len(documents)} chunked documents.")
        return documents

    def create_vectorstore(self, docs: List[Document]):
        """
        Creates a vector store from the provided documents using OpenAIEmbeddings and Chroma.
        """
        if os.path.exists(self.db_name):
            shutil.rmtree(self.db_name)
        os.makedirs(self.db_name, exist_ok=True)

        embeddings = OpenAIEmbeddings()
        vectorstore = Qdrant.from_documents(
            documents=docs,          # your list of chunked Documents
            embedding=embeddings,
            collection_name="my_collection",
            url="http://localhost:6333"  # assuming Qdrant is running locally
        )
        vectorstore.persist()
        print(f"Vectorstore created with {vectorstore._collection.count()} embedded chunks.")

        return vectorstore

    def run_pipeline(self):
        docs = self.load_chunked_documents()
        self.create_vectorstore(docs)






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
