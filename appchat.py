import streamlit as st
import os
import re
import unicodedata
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURACIÓN DE SEGURIDAD REPARADA PARA LA NUBE ---
if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    GROQ_API_KEY = "api-jgc" 

st.set_page_config(page_title="Asistente RAG", layout="wide")
st.title("🦙 Asistente Virtual de JGC")

if GROQ_API_KEY == "api-jgc":
    st.warning("⚠️ Usando clave de ejemplo. Si estás en la nube, asegúrate de configurar GROQ_API_KEY en Advanced Settings -> Secrets.")

# Inicializar modelos activos en Groq
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.2, groq_api_key=GROQ_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Inicializar estados de sesión
if "vector_store" not in st.session_state: st.session_state.vector_store = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "inicio_documento" not in st.session_state: st.session_state.inicio_documento = ""
if "texto_completo_pdf" not in st.session_state: st.session_state.texto_completo_pdf = ""

# Función auxiliar para normalizar texto (quitar acentos y diéresis)
def normalizar_texto(texto):
    texto_norm = unicodedata.normalize('NFD', texto)
    return "".join([c for c in texto_norm if unicodedata.category(c) != 'Mn']).lower()

with st.sidebar:
    st.header("📁 Base de Conocimiento")
    uploaded_file = st.file_uploader("Sube un archivo PDF", type=["pdf"])
    if uploaded_file is not None:
        temp_file_path = f"temp_{uploaded_file.name}"
        
        with open(temp_file_path, "wb") as f: 
            f.write(uploaded_file.getbuffer())
        
        with st.spinner("Procesando y segmentando documento..."):
            try:
                loader = PyPDFLoader(temp_file_path)
                docs = loader.load()
                
                # Guardar todo el texto del documento para búsquedas y conteos exactos
                st.session_state.texto_completo_pdf = "\n".join([doc.page_content for doc in docs])
                
                # Guardar el texto de las primeras páginas para la portada
                texto_inicial = ""
                for i in range(min(2, len(docs))):
                    texto_inicial += docs[i].page_content + "\n"
                st.session_state.inicio_documento = texto_inicial
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)
                
                st.session_state.vector_store = FAISS.from_documents(splits, embeddings)
                st.success("¡Documento procesado con éxito!")
            except Exception as e: 
                st.error(f"Error al procesar el PDF: {e}")
            finally: 
                if os.path.exists(temp_file_path): 
                    os.remove(temp_file_path)

# Mostrar el historial
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]): 
        st.write(message["content"])

user_query = st.chat_input("Hazme una pregunta sobre tus documentos...")
if user_query:
    with st.chat_message("user"): 
        st.write(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    
    if st.session_state.vector_store is None:
        response_text = "Por favor, primero sube un archivo PDF en la barra lateral para poder extraer el contexto."
    else:
        with st.spinner("Buscando en la base de datos y redactando respuesta..."):
            try:
                # DETECCIÓN DE SOLICITUD DE CONTEO
                conteo_info = ""
                if any(p in user_query.lower() for p in ["cuantas veces", "cuántas veces", "repite"]):
                    # Extraer términos candidatos entre comillas o palabras clave
                    match = re.search(r'["“]([^"”]+)["”]', user_query)
                    termino_a_buscar = match.group(1) if match else user_query.replace("cuantas veces", "").replace("cuántas veces", "").replace("se repite", "").replace("en espanol o ingles", "").replace("en español o inglés", "").strip("?¿ ")
                    
                    if termino_a_buscar:
                        texto_target = normalizar_texto(st.session_state.texto_completo_pdf)
                        
                        # Mapeo inteligente para variaciones comunes (español e inglés)
                        terminos_variaciones = [termino_a_buscar]
                        if "alfa sinucleina" in normalizar_texto(termino_a_buscar) or "alpha synuclein" in normalizar_texto(termino_a_buscar):
                            terminos_variaciones = ["alfa-sinucleina", "alfa sinucleina", "alpha-synuclein", "alpha synuclein", "a-synuclein", "a-sinucleina"]
                        
                        total_conteo = 0
                        detalles_conteo = []
                        for term in terminos_variaciones:
                            term_norm = normalizar_texto(term)
                            veces = texto_target.count(term_norm)
                            if veces > 0:
                                total_conteo += veces
                                detalles_conteo.append(f"'{term}': {veces} veces")
                        
                        conteo_info = f"\n[SISTEMA: El análisis de código matemático detectó que los términos solicitados se repiten un total de {total_conteo} veces en todo el PDF bruto. Desglose analítico: {', '.join(detalles_conteo) if detalles_conteo else '0'}. Transmite este número exacto al usuario de forma amigable sin quejarte por el idioma]."

                # Buscamos los fragmentos más parecidos
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
                docs = retriever.invoke(user_query)
                contexto_busqueda = "\n\n".join(d.page_content for d in docs)
                
                # Contexto integrado
                contexto_completo = f"--- INICIO/PORTADA DEL DOCUMENTO ---\n{st.session_state.inicio_documento}\n\n--- FRAGMENTOS RELEVANTES ---\n{contexto_busqueda}\n\n{conteo_info}"
                
                prompt = ChatPromptTemplate.from_template(
                    "Eres un asistente virtual experto, empático y altamente resolutivo. Responde siempre de forma clara y directa en español.\n"
                    "Si en el contexto se incluye una sección '[SISTEMA: ...]' con el conteo exacto de una palabra, confía ciegamente en ese número y dáselo directamente al usuario de manera integrada, natural y amable.\n"
                    "Considera que el texto original puede estar en inglés ('alpha-synuclein') pero el usuario pregunta en español ('alfa sinucleína'); sé inteligente y asocia ambos conceptos con naturalidad.\n\n"
                    "Contexto disponible:\n{context}\n\n"
                    "Pregunta del usuario:\n{question}\n\n"
                    "Respuesta analítica en español:"
                )
                
                rag_chain = prompt | llm | StrOutputParser()
                response_text = rag_chain.invoke({"context": contexto_completo, "question": user_query})
            except Exception as e:
                response_text = f"Ocurrió un error al procesar la consulta con el modelo de lenguaje: {e}"
            
    with st.chat_message("assistant"): 
        st.write(response_text)
    st.session_state.chat_history.append({"role": "assistant", "content": response_text})

