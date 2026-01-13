from pdf_processor import PDFProcessor
from vector_store import VectorStore
import os


def main():
    print("=" * 50)
    print("Building RAG System - Step 1: Create Embeddings")
    print("=" * 50)

    # Step 1: Process PDFs
    print("\n1. Processing PDFs...")
    processor = PDFProcessor()
    documents = processor.load_pdfs()

    if not documents:
        print("No documents found. Exiting.")
        return

    # Step 2: Split into chunks
    print("\n2. Splitting documents...")
    chunks = processor.split_documents(documents)

    if not chunks:
        print("No chunks created. Exiting.")
        return

    # Step 3: Create vector store
    print("\n3. Creating vector store...")
    store = VectorStore()
    store.create_embeddings(chunks)

    # Step 4: Save for later use
    print("\n4. Saving vector store...")
    store.save()

    print("\n" + "=" * 50)
    print("Embeddings created successfully!")
    print(f"Total chunks: {len(chunks)}")
    print("You can now run the RAG system.")
    print("=" * 50)


if __name__ == "__main__":
    main()