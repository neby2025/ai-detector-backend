import pdfplumber
import docx

def extract_text_from_pdf(file) -> str:
    with pdfplumber.open(file) as pdf:
        return "\n".join([page.extract_text() or "" for page in pdf.pages])

def extract_text_from_docx(file) -> str:
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_file(file, filename: str) -> str:
    if filename.endswith(".pdf"):
        return extract_text_from_pdf(file)
    elif filename.endswith(".docx"):
        return extract_text_from_docx(file)
    elif filename.endswith(".txt"):
        return file.read().decode("utf-8")
    else:
        raise ValueError("Unsupported file format")
