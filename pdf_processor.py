import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class PDFProcessor:
    def __init__(self, pdf_folder: str = "pdfs"):
        self.pdf_folder = pdf_folder
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_pdfs(self) -> List[str]:
        """Load all PDFs from the folder and return list of documents"""
        documents = []

        if not os.path.exists(self.pdf_folder):
            print(f"Error: Folder '{self.pdf_folder}' not found!")
            print(f"Create a folder named '{self.pdf_folder}' and put your PDFs in it.")
            return documents

        pdf_files = [f for f in os.listdir(self.pdf_folder) if f.lower().endswith('.pdf')]

        if not pdf_files:
            print(f"No PDF files found in '{self.pdf_folder}'!")
            return documents

        print(f"Found {len(pdf_files)} PDF files:")
        for pdf_file in pdf_files:
            print(f"  - {pdf_file}")

        for pdf_file in pdf_files:
            try:
                pdf_path = os.path.join(self.pdf_folder, pdf_file)
                print(f"Loading: {pdf_file}...")
                loader = PyPDFLoader(pdf_path)
                pdf_docs = loader.load()
                documents.extend(pdf_docs)
                print(f"  ✓ Loaded {len(pdf_docs)} pages")
            except Exception as e:
                print(f"  ✗ Error loading {pdf_file}: {str(e)}")

        return documents

    def split_documents(self, documents: List) -> List:
        """Split documents into smaller chunks"""
        if not documents:
            print("No documents to split!")
            return []

        print(f"Splitting {len(documents)} documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        return chunks


if __name__ == "__main__":
    # Test the processor
    processor = PDFProcessor()
    docs = processor.load_pdfs()
    if docs:
        chunks = processor.split_documents(docs)
        print (chunks)
        print(f"\nSample chunk: {chunks[0].page_content[:200]}...")