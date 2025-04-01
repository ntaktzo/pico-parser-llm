import os
from typing import List
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
import glob

class FolderTree:
    def __init__(self, root_path: str, show_hidden: bool = False, max_depth: int = None):
        self.root_path = os.path.abspath(root_path)
        self.show_hidden = show_hidden
        self.max_depth = max_depth

    def generate(self):
        self._print_tree(self.root_path)

    def _print_tree(self, current_path: str, prefix: str = "", depth: int = 0):
        if self.max_depth is not None and depth > self.max_depth:
            return

        try:
            entries = sorted(os.listdir(current_path))
        except PermissionError:
            print(prefix + "└── [Permission Denied]")
            return

        if not self.show_hidden:
            entries = [e for e in entries if not e.startswith('.')]

        for index, entry in enumerate(entries):
            path = os.path.join(current_path, entry)
            connector = "├── " if index < len(entries) - 1 else "└── "
            print(prefix + connector + entry)

            if os.path.isdir(path):
                extension = "│   " if index < len(entries) - 1 else "    "
                self._print_tree(path, prefix + extension, depth + 1)


class TestRetrieval:
    """
    Class to test retrieval from an existing Chroma vectorstore.
    """

    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore

    def similarity_search(self, query: str, k: int = 5):
        """
        Performs a similarity search and returns the top k results.
        """
        results = self.vectorstore.similarity_search(query, k=k)
        self.display_results(results)

    def similarity_search_with_metadata_filter(self, query: str, metadata_filter: dict, k: int = 5):
        """
        Performs a similarity search with metadata filtering and returns the top k results.
        """
        results = self.vectorstore.similarity_search(query, k=k, filter=metadata_filter)
        self.display_results(results)

    @staticmethod
    def display_results(results: List[Document]):
        """
        Prints the retrieved document chunks and their metadata.
        """
        for i, doc in enumerate(results, start=1):
            print(f"--- Result {i} ---")
            print(f"Content:\n{doc.page_content}\n")
            print("Metadata:")
            for key, value in doc.metadata.items():
                print(f"  {key}: {value}")
            print("-------------------------\n")


    @staticmethod
    def print_headings_from_cleaned_documents(base_folder="./data/text_cleaned"):
        json_files = glob.glob(os.path.join(base_folder, "**/*.json"), recursive=True)

        for json_file in json_files:
            print(f"\nDocument: {os.path.basename(json_file)}")
            print("Headings:")
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                chunks = data.get("chunks", [])
                if not chunks:
                    print("  No headings found (empty chunks).")
                    continue

                for chunk in chunks:
                    heading = chunk.get("heading", "(No Heading)")
                    print(f"  - {heading}")

            except Exception as e:
                print(f"  Error reading {json_file}: {e}")


class HeadingPrinter:
    def __init__(self, base_folder="./data/text_translated"):
        self.base_folder = base_folder

    def print_all_headings(self):
        json_files = glob.glob(os.path.join(self.base_folder, "**/*.json"), recursive=True)

        for json_file in json_files:
            self.print_headings_from_file(json_file)

    def print_headings_from_file(self, json_file):
        print(f"\nDocument: {os.path.basename(json_file)}")
        print("Headings:")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            chunks = data.get("chunks", [])
            if not chunks:
                print("  No headings found (empty chunks).")
                return

            for chunk in chunks:
                heading = chunk.get("heading", "(No Heading)")
                print(f"  - {heading}")

        except Exception as e:
            print(f"  Error reading {json_file}: {e}")
