import streamlit as st
import json
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# --- 1. CONFIGURAÇÃO VISUAL (Estilo Deep Blue) ---
st.set_page_config(page_title="Consultor Fiscal AI", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #050a14; color: #e0e0e0; }
    [data-testid="stChatMessage"] {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 15px;
        margin-bottom: 10px;
    }
    h1, h2, h3 { color: #60a5fa !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CONFIGURAÇÃO DE AMBIENTE ---
os.environ["OPENAI_API_KEY"] = "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"

# --- 3. LÓGICA DE DADOS ---
@st.cache_resource
def inicializar_consultor():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docs_finais = []

    if os.path.exists("créditos_pis_cofins.pdf"):
        loader = PyPDFLoader("créditos_pis_cofins.pdf")
        pdf_docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs_finais.extend(splitter.split_documents(pdf_docs))
    
    if os.path.exists("contas_formatadas.json"):
        with open('contas_formatadas.json', 'r', encoding='utf-8') as f:
            dados = json.load(f)
            for item in dados:
                content = f"CÓDIGO: {item['codigo']} | NOME_CONTA: {item['descricao']} | INFO: {item['detalhamento']}"
                docs_finais.append(Document(page_content=content, metadata={"fonte": "json"}))

    return Chroma.from_documents(
        documents=docs_finais, 
        embedding=embeddings,
        collection_name="pis_cofins_collection"
    )

try:
    vector_db = inicializar_consultor()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
except Exception as e:
    st.error(f"Erro: {e}")
    st.stop()

# --- 4. INTERFACE ---
st.title("⚖️ PIS e COFINS AI")
st.caption("Respostas sobre créditos e contas contábeis de forma objetiva e direta.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ex: Posso aplicar crédito em determinado serviço/compra e em qual a conta?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando..."):
            docs = vector_db.similarity_search(prompt, k=5)
            contexto = "\n\n".join([d.page_content for d in docs])
            
            # PROMPT AJUSTADO PARA RESPOSTAS CURTAS E SEM JSON
            prompt_final = f"""
            Você é um Especialista Fiscal e Expert apenas em PIS e COFINS de uma empresa de transporte e logítica.
            Todos os serviços ou compra de materiais que são relacionados a transporte e logística, manutenção e veículos, serviços que são considerados como essenciais para que toda operação ocorre é passível de PIS e COFINS.
            Responda de forma breve mas com fundamentação com base no créditos_pis_cofins.pdf à pergunta usando o CONTEXTO abaixo.

            REGRAS DE FORMATAÇÃO:
            1. Para o crédito (PDF): Diga apenas se é permitido ou não é permitido, e o porquê com uma fundamentação legal do arquivo em PDF.
            2. Para a conta (JSON): Forneça apenas uma conta com seu número do código e o nome da conta (Ex: 444444 - Nome da Conta). 
            3. PROIBIDO retornar blocos de código ou formato JSON.

            CONTEXTO:
            {contexto}

            PERGUNTA:
            {prompt}
            """
            
            response = llm.invoke(prompt_final)
            full_response = response.content
            st.markdown(full_response)
            
    st.session_state.messages.append({"role": "assistant", "content": full_response})
