import sys
import os
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage

# CONFIGURATION
PDF_FILE = "D:\\trial\\data\\Lecture_Notes_Unit_2.pdf"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3"

def main():
    # --- STEP 1: LOAD & SPLIT ---
    print(f" Loading {PDF_FILE}...")
    if not os.path.exists(PDF_FILE):
        print(f" Error: File found.")
        return

    loader = PyPDFLoader(PDF_FILE)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    # --- STEP 2: EMBED ---
    print(f" Generating embeddings with {EMBED_MODEL}...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # --- STEP 3: SETUP MEMORY CHAINS ---
    llm = ChatOllama(model=CHAT_MODEL, temperature=0)

    # A. The "Rephraser" Prompt
    # This prompt tells the LLM to rewrite the question based on history
    contextualize_q_system_prompt = """Given a chat history and the latest user question 
    which might reference context in the chat history, formulate a standalone question 
    which can be understood without the chat history. Do NOT answer the question, 
    just reformulate it if needed and otherwise return it as is."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # B. The "History Aware" Retriever
    # This chain will sit between the user and the vector store
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # C. The "Answer" Prompt
    # This is the standard RAG prompt
    qa_system_prompt = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.

    {context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # D. The Final Chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # --- STEP 4: INTERACTIVE LOOP WITH MEMORY ---
    chat_history = [] # <--- This list stores our conversation

    print("\n" + "="*50)
    print(" RAG with Memory Ready! (Ask follow-up questions!)")
    print("="*50)

    while True:
        query = input("\n Your Question: ")
        if query.lower() in ["exit", "quit", "q"]:
            break
        
        print("   Thinking...")
        try:
            # We pass the history along with the question
            response = rag_chain.invoke({"input": query, "chat_history": chat_history})
            
            print(f"\n ANSWER:\n{response['answer']}")
            
            # UPDATE HISTORY
            # We save the interaction so the next loop can see it
            chat_history.append(HumanMessage(content=query))
            chat_history.append(AIMessage(content=response['answer']))
            
        except Exception as e:
            print(f" Error: {e}")

if __name__ == "__main__":
    main()