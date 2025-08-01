# Pico Parser LLM Pipeline

This repository provides a workflow for extracting information from PDF documents and preparing the data for retrieval augmented generation (RAG) tasks.  It targets health technology assessment (HTA) submissions and clinical guidelines but can be used on any medical PDF.

## Project Structure

```
application/       Application specific files
python/            Core processing modules
    process.py     PDF handling, table detection and translation logic
    retrieve.py    Retrieval utilities and PICO extraction
    vectorise.py   Document chunking and vector store creation
    utils.py       Helper classes
    run.py         High level pipeline orchestration
results/           Example outputs
```

The `main_rag.py` script in the project root launches the entire pipeline through the classes defined in `python/run.py`.

## Core Classes

### TableDetector (`python/process.py`)
Detects tables inside PDF pages.  It analyses text patterns and the PDF layout to decide whether extracted data represents a real table.  It can work with different sensitivity levels and supports domain specific checks for medical documents.

### PDFProcessor (`python/process.py`)
Loads a PDF file and extracts cleaned text.  The processor identifies headings, removes boilerplate and footnotes and uses `TableDetector` to locate tables.  The output is a set of chunks containing the text and any detected tables.

### Translator (`python/process.py`)
Translates extracted JSON documents to English.  It supports multiple translation models, handles GPU availability and preserves medical terminology when possible.  Translation quality is assessed before deciding whether to retry with a higher tier model.

### Chunker (`python/vectorise.py`)
Splits translated JSON documents into smaller pieces either semantically or by fixed size.  The resulting segments are saved back to disk and can be fed into a vectoriser.

### Vectoriser (`python/vectorise.py`)
Creates embeddings for document chunks using OpenAI or BioBERT models and stores them in a Chroma vector database.  The vectoriser can also project embeddings with t‑SNE for inspection.

### ChunkRetriever (`python/retrieve.py`)
Performs similarity search over vector stores.  It implements utilities for deduplication and heading analysis and is used by the PICO extractor.

### PICOExtractor (`python/retrieve.py`)
Given a user query and retrieved context, this class interacts with an OpenAI model to produce PICO (Population, Intervention, Comparator, Outcome) summaries.

### FolderTree, HeadingPrinter (`python/utils.py`)
Small helper utilities to display folder structures and print headings detected in translated documents.

### RagHTASubmission (`python/run.py`)
Orchestrates the full RAG pipeline.  It ties together PDF processing, translation, chunking, vectorisation and retrieval to generate final PICO outputs.

## Usage

1. Place PDFs in `data/PDF` following a country based folder structure.
2. Adjust paths and parameters in `main_rag.py` or create your own script using `RagHTASubmission`.
3. Run the pipeline to produce cleaned, translated and vectorised documents ready for question answering.

## License

This project is distributed under the MIT License.
