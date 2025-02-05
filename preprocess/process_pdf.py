import pdfplumber
import os
import re

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file using pdfplumber with better handling of spacing issues.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted and cleaned text from the PDF.
    """
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Use x_tolerance to fix missing spaces, y_tolerance to handle multi-line text
                text += page.extract_text(x_tolerance=2, y_tolerance=3) + "\n"

        # Fix spacing issues using an improved regex function
        text = fix_faulty_spacing(text)
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return None

def fix_faulty_spacing(text):
    """
    Fixes incorrect spacing issues in extracted text.

    Args:
        text (str): Raw extracted text with spacing errors.

    Returns:
        str: Cleaned text with fixed spacing issues.
    """
    # Add a space before lowercase-uppercase word joins (e.g., "inde" -> "in de")
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

    # Add space between numbers and words if needed
    text = re.sub(r"(\d)([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z])(\d)", r"\1 \2", text)

    # Add space where two words have been mistakenly joined
    text = re.sub(r"([a-z]{2,})([A-Z][a-z]+)", r"\1 \2", text)

    # Fix missing spaces before conjunctions and prepositions (e.g., "vaak incombinatie" -> "vaak in combinatie")
    common_prefixes = r"\b(in|de|het|met|zonder|te|van|voor|op|aan|als|door|en|of|om|uit|over|bij)\b"
    text = re.sub(r"(\w)" + common_prefixes, r"\1 \2", text)

    return text

def process_pdfs(input_dir, output_dir):
    """
    Processes all PDF files in the input directory, extracts their text,
    and saves the extracted text into a structured output directory.

    Args:
        input_dir (str): Path to the directory containing PDF files.
        output_dir (str): Path to the directory where extracted text files will be saved.
    """
    print("Processing PDFs...")
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

