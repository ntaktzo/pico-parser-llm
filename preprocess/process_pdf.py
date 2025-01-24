from PyPDF2 import PdfReader
import os

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file using PyPDF2.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.
    """
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return None

def process_pdfs(input_dir, output_dir):
    """
    Processes all PDF files in the input directory, extracts their text,
    and saves the extracted text into a structured output directory.

    Args:
        input_dir (str): Path to the directory containing PDF files.
        output_dir (str): Path to the directory where extracted text files will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)

                # Extract text from the PDF
                text = extract_text_from_pdf(pdf_path)
                if text:
                    
                    # Generate output file path
                    relative_path = os.path.relpath(root, input_dir)
                    output_subdir = os.path.join(output_dir, relative_path)
                    os.makedirs(output_subdir, exist_ok=True)

                    output_file = os.path.join(output_subdir, f"{os.path.splitext(file)[0]}.txt")

                    # Save the extracted text to a .txt file
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(text)

                    print(f"Extracted text saved to {output_file}")
                else:
                    print(f"Failed to extract text from {pdf_path}")

