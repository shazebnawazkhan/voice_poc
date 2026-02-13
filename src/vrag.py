
from docling.document_converter import DocumentConverter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS





import os
import json
from typing import List

class VRAG():
    def __init__(self) -> None:
        pass

    def load_and_convert_document(self, file_path):
        converter = DocumentConverter()
        result = converter.convert(file_path)
        return result.document.export_to_markdown()

    def get_markdown_splits(self, markdown_content, metadata=None):
        headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
        chunks = markdown_splitter.split_text(markdown_content)
        if metadata:
            for chunk in chunks:
                chunk.metadata.update(metadata)
        return chunks

    def setup_vector_store(self, chunks):
        embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url="http://localhost:11434")
        single_vector = embeddings.embed_query("this is some text data")
        index = faiss.IndexFlatL2(len(single_vector))
        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        vector_store.add_documents(documents=chunks)
        return vector_store

    def get_vector_store_from_folder(self, folder_path: str, file_ext: str = ".pdf"):
        """
        Index all files with the given extension in the folder for RAG.
        Adds file name as metadata to each chunk.
        Persists the index to disk and supports incremental updates.
        Returns a single vector store containing all documents.
        """
        index_path = self._get_index_path(folder_path)
        indexed_files = self._load_indexed_files(index_path)
        
        # Try to load existing vector store
        existing_store = self._load_vector_store(index_path)
        
        # Find new files to process
        new_files = []
        all_files = []
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(file_ext):
                all_files.append(fname)
                if fname not in indexed_files:
                    new_files.append(fname)
        
        # Process new files if any exist
        if new_files:
            new_chunks = []
            for fname in new_files:
                fpath = os.path.join(folder_path, fname)
                try:
                    markdown_content = self.load_and_convert_document(fpath)
                    chunks = self.get_markdown_splits(markdown_content, metadata={"filename": fname})
                    new_chunks.extend(chunks)
                    print(f"Processed: {fname} ({len(chunks)} chunks)")
                except Exception as e:
                    print(f"Failed to process {fpath}: {e}")
            
            if new_chunks:
                if existing_store:
                    # Merge new chunks with existing store
                    vector_store = self._merge_vector_stores(existing_store, new_chunks)
                else:
                    # Create new store from scratch
                    vector_store = self.setup_vector_store(new_chunks)
                
                # Update indexed files list and save
                indexed_files = all_files
                self._save_indexed_files(index_path, indexed_files)
                self._save_vector_store(vector_store, index_path)
            else:
                vector_store = existing_store
        else:
            if existing_store:
                print("No new files to process. Using existing index.")
                vector_store = existing_store
            else:
                raise ValueError(f"No index found and no {file_ext} files to process in {folder_path}")
        
        if not vector_store:
            raise ValueError(f"Failed to create or load vector store for {folder_path}")
        
        return vector_store

    def _get_index_path(self, folder_path: str):
        """Get the path for storing the FAISS index"""
        index_dir = "indexes"
        os.makedirs(index_dir, exist_ok=True)
        # Create a safe filename from the folder path
        index_name = os.path.basename(folder_path) or "default"
        return os.path.join(index_dir, index_name)

    def _get_indexed_files_path(self, index_path: str):
        """Get the path for tracking indexed files metadata"""
        return f"{index_path}_indexed_files.json"

    def _load_indexed_files(self, index_path: str):
        """Load the list of already indexed files"""
        metadata_file = self._get_indexed_files_path(index_path)
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading indexed files metadata: {e}")
                return []
        return []

    def _save_indexed_files(self, index_path: str, indexed_files: List[str]):
        """Save the list of indexed files metadata"""
        metadata_file = self._get_indexed_files_path(index_path)
        try:
            with open(metadata_file, 'w') as f:
                json.dump(indexed_files, f)
        except Exception as e:
            print(f"Error saving indexed files metadata: {e}")

    def _load_vector_store(self, index_path: str):
        """Load FAISS vector store from disk"""
        try:
            embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url="http://localhost:11434")
            vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            print(f"Loaded existing index from {index_path}")
            return vector_store
        except Exception as e:
            print(f"Could not load existing index: {e}")
            return None

    def _save_vector_store(self, vector_store, index_path: str):
        """Save FAISS vector store to disk"""
        try:
            vector_store.save_local(index_path)
            print(f"Saved index to {index_path}")
        except Exception as e:
            print(f"Error saving index: {e}")

    def _merge_vector_stores(self, existing_store, new_chunks):
        """Merge new chunks into existing vector store"""
        try:
            existing_store.add_documents(documents=new_chunks)
            print(f"Merged {len(new_chunks)} new chunks into existing index")
            return existing_store
        except Exception as e:
            print(f"Error merging vector stores: {e}")
            return None

    def get_vector_store(self, file_path):
        markdown_content = self.load_and_convert_document(file_path)
        chunks = self.get_markdown_splits(markdown_content)
        vector_store = self.setup_vector_store(chunks)
        return vector_store


# Formatting documents for RAG
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def create_rag_chain(retriever):
    prompt = """
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know, do not hallucinate.
        Answer in bullet points. Make sure your answer is relevant to the question and it is answered from the context only.
        Answer the questionin single line.
        ### Question: {question} 

        ### Context: {context} 

        ### Answer:
    """
    model = ChatOllama(model="deepseek-r1:1.5b", base_url="http://localhost:11434")
    prompt_template = ChatPromptTemplate.from_template(prompt)

    chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | model
            | StrOutputParser()
    )
    return chain





if __name__ == "__main__":
    vrag = VRAG()
    # Index all PDFs in the rag_docs folder with file names as metadata
    vs = vrag.get_vector_store_from_folder("rag_docs")
    
    #ctx = vs.similarity_search("Compare skills and experience of Shazeb and Razi in data science")
    #context = vs.search(search_type="mmr", search_kwargs={'k': 3}, query="What is Shazeb's experience with Python?")




    # Setup retriever
    #retriever = vs.as_retriever(search_type="mmr", search_kwargs={'k': 3})
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={'k': 10})

    # Create RAG chain

    rag_chain = create_rag_chain(retriever)


    question = "What are academic qualifications of Shazeb?"
    #question = "Who is the father of Shazeb?"
    #question = "What is the age of Shazeb?"


    while True:
        question = input("Enter something (or type 'quit' to exit): ")

        if question.lower() == 'quit':
            print("Exiting program.")
            break  # Exit the while loop
        else:
            #print(f"You entered: {question}")
            print(f"Question: {question}")
            for chunk in rag_chain.stream(question):
                print(chunk, end="", flush=True)
            print("\n" + "-" * 50 + "\n")



    # print("Similarity Search Results:")
    # for doc in ctx:
    #     print(f"File: {doc.metadata.get('filename', 'Unknown')}")
    #     print(f"Content: {doc.page_content[:100]}...\n")
    
    # print("\nMMR Search Results:")
    # for doc in context:
    #     print(f"File: {doc.metadata.get('filename', 'Unknown')}")
    #     print(f"Content: {doc.page_content[:100]}...\n")