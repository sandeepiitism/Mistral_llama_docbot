# Imports
from PyPDF2 import PdfReader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# Extract Data From the PDF File
def load_pdf_file(file_path):
    """
    Load and extract text from a single PDF file.
    """
    text = ""
    pdf_reader = PdfReader(file_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Split the Data into Text Chunks
def text_split(extracted_text):
    """
    Split the extracted text into smaller chunks using a character-based splitter.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    text_chunks = text_splitter.split_text(extracted_text)
    return text_chunks

# Download the Embeddings from Ollama
def download_ollama_embeddings():
    """
    Load Ollama Embeddings model.
    """
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings
