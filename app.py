import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from htmlTemplates import css
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate

def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.header("Chat with multiple PDFs :books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "max_docs_result" not in st.session_state:
        st.session_state.max_docs_result = None
    if "apikey" not in st.session_state:
        st.session_state.apikey = None
    if "temperature" not in st.session_state:
        st.session_state.temperature = None

    api_key = st.text_input("Enter your API key", type = "password")

    if api_key:
        st.session_state.apikey = api_key
        max_result = st.number_input("Max. documents result:", value=2)
        temperature = st.number_input("Desired temperature:", value=1)
        user_question = st.text_input("Ask a question about your documents:")

        if max_result:
            st.session_state.max_docs_result = max_result
        if temperature:
            st.session_state.temperature = temperature

        if user_question:
            handle_userinput(user_question)

        with st.sidebar:
            st.subheader("Upload your documents here!")
            
            pdf_docs = st.file_uploader("Load them and then hit process", accept_multiple_files=True, type="pdf")
            
            if st.button("Process"):
                with st.spinner("Processing"):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    st.session_state.vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain()
    else:
        st.warning("Enter your OPENAI API-KEY. Get your OpenAI API key from [here](https://platform.openai.com/account/api-keys).\n")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.apikey)
    vectorstore = Chroma.from_texts(text_chunks, embeddings, metadatas=[{"source": str(i)} for i in range(len(text_chunks))]).as_retriever()
    return vectorstore

def get_conversation_chain():
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    {context}
    Question: {question}
    Answer in Spanish:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    temperature = st.session_state.temperature
    return load_qa_chain(OpenAI(temperature=temperature, openai_api_key=st.session_state.apikey), chain_type="stuff", prompt=PROMPT, verbose=True)

def handle_userinput(user_question):
    max_docs_result = st.session_state.max_docs_result 
    documents = st.session_state.vectorstore.vectorstore.similarity_search(user_question, k=max_docs_result)

    with get_openai_callback() as cb:
        result = st.session_state.conversation({"input_documents": documents, "question": user_question})
        st.warning(f"Total Tokens: {cb.total_tokens}")
        st.warning(f"Estimated cost: {cb.total_cost}")
        st.success(result['output_text'])
        for doc in result['input_documents']:
            st.info(doc)

if __name__ == '__main__':
    main()