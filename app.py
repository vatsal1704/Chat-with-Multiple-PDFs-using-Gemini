# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os

# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.vectorstores import FAISS
# import google.generativeai as genai
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv


# # Load environment variables
# load_dotenv()

# # Set up Google Generative AI API key
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# def get_pdf_text(pdf_docs):
#     text=""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text


# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
#     chunks=text_splitter.split_text(text)
#     return chunks



# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks,embeddings)
#     vector_store.save_local("faiss_index")
#     return vector_store



# def get_conversation_chain():
#     prompt_template = """
# You are a helpful assistant. Answer the question as detailed as possible from the provided context.
# Make sure to provide all the details. If the answer is not in the provided context, just say:
# "Answer is not available in the context". Don't make up answers.

# Use the following pieces of context to answer the question at the end.

# Context:
# {context}

# Question:
# {question}
# """
#     # Use the correct model identifier for Gemini chat
#     model = ChatGoogleGenerativeAI(model="models/chat-bison-001", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(llm=model, prompt=prompt, chain_type="stuff")
#     return chain




# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search(user_question)
#     chain = get_conversation_chain()
#     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
#     st.write("### ✅ Answer:")
#     st.write(response['output_text'])




# def main():
#     st.set_page_config(page_title="📚 Chat with Multiple PDFs", layout="wide")
#     st.title("🤖 Chat with your PDFs using Gemini")
#     st.markdown("Upload multiple PDF files and ask questions. Answers are generated using Google's Gemini model.")

#     # Sidebar
#     with st.sidebar:
#         st.header("📄 Upload Your PDFs")
#         pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])
#         process_button = st.button("🔍 Process PDFs")

#     if process_button and pdf_docs:
#         with st.spinner("📚 Reading and indexing your documents..."):
#             raw_text = get_pdf_text(pdf_docs)
#             text_chunks = get_text_chunks(raw_text)
#             get_vector_store(text_chunks)
#             st.success("✅ PDFs processed successfully! You can now ask questions.")

#     # Chat section
#     st.markdown("---")
#     st.subheader("💬 Ask a question about your uploaded PDFs")

#     user_question = st.text_input("Type your question here...")

#     if user_question:
#         with st.spinner("🤔 Thinking..."):
#             user_input(user_question)



# if __name__ == "__main__":
#     main()






import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def get_vector_store(chunks):
    emb = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vs = FAISS.from_texts(chunks, embedding=emb)
    vs.save_local("faiss_index")
    return vs

def get_conversation_chain():
    prompt = """
You are a helpful assistant. Use the provided context to answer the user's question as accurately as possible.

Context:
{context}

Question:
{question}
"""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)
    template = PromptTemplate(template=prompt, input_variables=["context", "question"])
    return load_qa_chain(llm=llm, prompt=template, chain_type="stuff")

def user_input(question):
    emb = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    db = FAISS.load_local("faiss_index", emb, allow_dangerous_deserialization=True)
    docs = db.similarity_search(question)
    chain = get_conversation_chain()
    resp = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    st.write("### ✅ Answer")
    st.write(resp["output_text"])

def main():
    st.set_page_config(page_title="📚 Chat with Multiple PDFs", layout="wide")
    st.title("🤖 Chat with Multiple PDFs using Gemini")
    st.markdown("Upload PDFs, process them, and ask questions")

    with st.sidebar:
        st.header("📄 Upload PDFs")
        pdfs = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)
        go = st.button("🔍 Process PDFs")

    if go and pdfs:
        with st.spinner("📚 Processing PDFs..."):
            raw = get_pdf_text(pdfs)
            chunks = get_text_chunks(raw)
            get_vector_store(chunks)
            st.success("✅ PDFs indexed! Ask your question below.")

    st.markdown("---")
    question = st.text_input("💬 Your question about the PDFs")

    if question:
        if not os.path.exists("faiss_index"):
            st.warning("⚠️ Please upload and process PDFs first.")
        else:
            with st.spinner("🤔 Thinking..."):
                user_input(question)

if __name__ == "__main__":
    main()
