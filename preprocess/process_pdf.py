import os
import re
import pdfplumber

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file using pdfplumber with improved handling of 
    multi-column layouts, footers/headers, spacing issues, heading isolation, 
    and reference removal.
    """
    try:
        all_text = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Optionally set layout=True if needed for multi-column pages
                raw_text = page.extract_text(x_tolerance=2, y_tolerance=3)
                if raw_text:
                    cleaned_page_text = remove_header_footer(raw_text)
                    all_text.append(cleaned_page_text)

        text = "\n".join(all_text)
        text = insert_heading_breaks(text)
        text = fix_faulty_spacing(text)
        text = merge_broken_lines(text)

        # >>> Remove references here <<<
        text = remove_references(text)

        return text

    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return None


def remove_references(text):
    """
    Removes references in several common formats:
      - Inline bracketed citations: [1], [2], [1,2], [Smith 2022]
      - Parenthetical citations: (Smith et al., 2023)
      - Entire reference section after a heading like 'References' or 'Bibliography'.
    Adjust the regexes for your document style.
    """
    # 1) Remove bracketed citations like [1], [12, 13], or [Smith 2022]
    #    You can make this more or less strict as needed.
    text = re.sub(r"\[\s*[\w\s,.-]+\s*\]", "", text)

    # 2) Remove parenthetical citations like (Smith et al., 2023) or (Doe, 1999)
    #    This is a simple pattern that looks for parentheses containing letters + digits
    text = re.sub(r"\([A-Za-z,.\s]+\d{4}.*?\)", "", text)

    # 3) Remove everything after a heading like "References" or "Bibliography"
    #    (Case-insensitive). This may remove the entire references section at the end.
    #    If your documents use "Literature," "Bibliografie," etc., add them below.
    text = re.sub(r"(?:^|\n)(References|Bibliography|Literature|Bibliografie|Referenties)\s*\n.*",
                  "",
                  text,
                  flags=re.IGNORECASE | re.DOTALL)

    # Remove extra spaces left behind
    text = re.sub(r"\s{2,}", " ", text)
    return text


def remove_header_footer(page_text):
    disclaimer_phrases = [
        "© NICE",
        "All rights reserved",
        "Subject to Notice of rights",
        "conditions#notice-of-rights",
        "Contents Overview",
        "PDF aangemaakt op 23-01-2025",
        "REGIONALA CANCERCENTRUM",
        "© Leitlinienprogramm Onkologie",
        "Langversion",
        "März 2024",
        "Richtlijnendatabase"
    ]

    lines = page_text.split("\n")
    cleaned_lines = []
    for line in lines:
        # Skip lines that are page numbers or contain known boilerplate phrases
        if re.match(r"^\s*\d+\s*$", line.strip()):
            continue
        if any(phrase in line for phrase in disclaimer_phrases):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def insert_heading_breaks(text):
    """
    Insert a blank line before lines or mid-lines that appear to be headings.
    This is more aggressive than the previous version:
     - Looks for lines that are uppercase
     - Looks for lines that start with digits + dot + space + uppercase (e.g. "2. HEADING")
     - Looks for lines that end with a colon
    Then forces a newline before the heading.

    You can refine the regex as you see fit.
    """
    # 1) Insert newline before lines that match patterns (digit-dot or uppercase)
    #    Because some headings might appear mid-line, we handle them carefully.
    
    # Pattern explanation:
    #  - (?<!\n): Negative lookbehind to ensure we don't have multiple newlines
    #  - (?:^|(?<=\n)): Start of string or preceded by newline
    #  - (\d+\.\s[A-Z]+.*|[A-Z\s]+:): 
    #        either "digits dot space uppercase" or "ALLCAPS + colon"
    #        This is just an example. Adjust to your heading style.

    # Insert a special marker before these headings
    text_with_markers = re.sub(
        r"(?:(?<=\n)|^)(\d+\.\s[A-Z]+.*|[A-Z\s]+:)",
        r"\n\1",
        text
    )

    # 2) Now also handle lines that are fully uppercase and longer than 2 chars
    #    We'll look for mid-line uppercase runs as well.
    def add_newline_before_uppercase(m):
        return "\n" + m.group(1)

    text_with_markers = re.sub(
        r"(?:(?<=\n)|^)([A-Z][A-Z\s]{2,})",  # A line of uppercase (3+ chars)
        add_newline_before_uppercase,
        text_with_markers
    )

    return text_with_markers



def fix_faulty_spacing(text):
    # Remove hyphens at line breaks (e.g., "medi-\nastinal" -> "mediastinal")
    text = re.sub(r"-\n", "", text)
    # If a line ends with a hyphen but isn't followed by whitespace, join them
    text = re.sub(r"-\s+", "", text)
    # Add a space before lowercase-uppercase transitions
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    # Add space between numbers and words if they are directly concatenated
    text = re.sub(r"(\d)([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z])(\d)", r"\1 \2", text)
    # Merge obviously joined words (heuristic)
    text = re.sub(r"([a-z]{2,})([A-Z][a-z]+)", r"\1 \2", text)
    # Fix missing spaces before common Dutch words/prepositions
    common_prefixes = r"\b(in|de|het|met|zonder|te|van|voor|op|aan|als|door|en|of|om|uit|over|bij)\b"
    text = re.sub(r"(\w)" + common_prefixes, r"\1 \2", text)
    return text


def merge_broken_lines(text):
    lines = text.split("\n")
    merged_lines = []
    buffer_line = ""

    for line in lines:
        line = line.strip()
        if not line:
            if buffer_line:
                merged_lines.append(buffer_line)
                buffer_line = ""
            merged_lines.append("")
            continue

        if not buffer_line:
            buffer_line = line
        else:
            if re.search(r"[.!?;:]$", buffer_line):
                merged_lines.append(buffer_line)
                buffer_line = line
            else:
                buffer_line += " " + line

    if buffer_line:
        merged_lines.append(buffer_line)

    return "\n".join(merged_lines)


def process_pdfs(input_dir, output_dir):
    print("Processing PDFs...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                text = extract_text_from_pdf(pdf_path)
                if text:
                    relative_path = os.path.relpath(root, input_dir)
                    output_subdir = os.path.join(output_dir, relative_path)
                    os.makedirs(output_subdir, exist_ok=True)
                    output_file = os.path.join(
                        output_subdir, f"{os.path.splitext(file)[0]}.txt"
                    )
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(text)
                    print(f"Extracted text saved to {output_file}")
                else:
                    print(f"Failed to extract text from {pdf_path}")
