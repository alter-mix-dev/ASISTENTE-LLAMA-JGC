import streamlit as st
import os
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Configuración de la interfaz en Streamlit
st.set_page_config(page_title="Asistente RAG Local", layout="wide")
st.title("🦙 Asistente Virtual DE JGC con RAG (Llama 3 + Tus Documentos PDF)")

# Inicializar modelos locales de Ollama
llm = Ollama(model="llama3", temperature=0.2)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Guardar la base de datos de vectores en la sesión para no recrearla en cada clic
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Barra lateral para subir el archivo PDF
with st.sidebar:
    st.header("📁 Base de Conocimiento")
    uploaded_file = st.file_uploader("Sube un archivo PDF para entrenar al asistente", type=["pdf"])
    
    if uploaded_file is not None:
        # Guardar archivo temporalmente para que el cargador pueda leerlo
        temp_file_path = f"temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner("Procesando y vectorizando el documento..."):
            try:
                # Cargar el PDF
                loader = PyPDFLoader(temp_file_path)
                docs = loader.load()
                
                # Dividir el texto en fragmentos manejables (Chunks)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)
                
                # Crear el almacén de vectores indexado
                st.session_state.vector_store = FAISS.from_documents(splits, embeddings)
                st.success("¡Documento procesado con éxito!")
            except Exception as e:
                st.error(f"Error procesando el archivo: {e}")
            finally:
                # Limpiar el archivo temporal creado
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

# 2. Flujo Principal del Chat
# Mostrar el historial de la conversación
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Entrada de la pregunta del usuario
user_query = st.chat_input("Hazme una pregunta sobre el documento cargado...")

if user_query:
    # Mostrar la pregunta en pantalla y guardarla
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    
    # Validar si hay un documento cargado en el sistema RAG
    if st.session_state.vector_store is None:
        response_text = "Por favor, sube un documento PDF en la barra lateral primero para poder ayudarte con la información."
    else:
        with st.spinner("Buscando en los documentos y generando respuesta..."):
            # Configurar el buscador por similitud (Recuperador)
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
            
            # Diseñar la plantilla de Prompt para delimitar las respuestas al contexto
            prompt_template = """
            Eres un asistente virtual riguroso y experto. Responde la pregunta del usuario basándote ÚNICAMENTE en el contexto proporcionado.
            Si la respuesta no se encuentra en el contexto, di amablemente que no posees esa información en los documentos cargados.
            
            Contexto:
            {context}
            
            Pregunta:
            {question}
            
            Respuesta en español:
            """
            prompt = ChatPromptTemplate.from_template(prompt_template)
            
            # Formatear de forma estructurada los documentos recuperados
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            # Construir la cadena RAG interactiva
            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            # Ejecutar la cadena con la duda del usuario
            response_text = rag_chain.invoke(user_query)
            
    # Mostrar la respuesta final de Llama y guardarla en el historial
    with st.chat_message("assistant"):
        st.write(response_text)
    st.session_state.chat_history.append({"role": "assistant", "content": response_text})
