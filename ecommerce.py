import streamlit as st
import os
import cv2
import numpy as np
import pytesseract
from PIL import Image
import pyperclip
import xml.etree.ElementTree as ET
from langchain_community.document_loaders import WebBaseLoader, UnstructuredFileLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from pathlib import Path
import tempfile

# Configurações iniciais
st.set_page_config(
    page_title="🛒 Assistente VTEX/Shopify",
    page_icon="🤖",
    layout="wide"
)

# ==============================================
# 1. Configuração do Ambiente
# ==============================================

# Configuração do Tesseract OCR
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'
except Exception as e:
    st.error(f"❌ Erro na configuração do Tesseract: {str(e)}")
    st.stop()

# ==============================================
# 2. Funções Principais (Atualizadas)
# ==============================================

def load_docs(xml_files, urls, openai_api_key):
    """Carrega documentos de XMLs e URLs"""
    try:
        documents = []
        
        # Processa XMLs
        for xml_file in xml_files:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xml') as tmp:
                    tmp.write(xml_file.getvalue())
                    tmp_path = tmp.name
                
                tree = ET.parse(tmp_path)
                root = tree.getroot()
                text = ET.tostring(root, encoding='unicode', method='text')
                documents.append({"page_content": text, "metadata": {"source": xml_file.name}})
                os.unlink(tmp_path)
            except Exception as e:
                st.error(f"Erro ao processar {xml_file.name}: {str(e)}")
                continue
        
        # Processa URLs
        if urls:
            try:
                valid_urls = [url for url in urls if url.startswith(('http://', 'https://'))]
                if valid_urls:
                    loader = WebBaseLoader(valid_urls)
                    documents.extend(loader.load())
            except Exception as e:
                st.error(f"Erro ao carregar URLs: {str(e)}")
        
        if not documents:
            st.warning("Nenhum documento válido encontrado")
            return None
            
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        db = FAISS.from_documents(documents, embeddings)
        db.save_local("faiss_index")
        return db
        
    except Exception as e:
        st.error(f"Erro ao carregar documentos: {str(e)}")
        return None

def extract_text_from_image(image_file):
    """Extrai texto com pré-processamento avançado"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
            tmp.write(image_file.getvalue())
            tmp_path = tmp.name
        
        # Pré-processamento
        img = cv2.imread(tmp_path)
        if img is None:
            raise ValueError("Não foi possível ler a imagem")
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR com configuração otimizada
        custom_config = r'--oem 3 --psm 6 -l por+eng'
        text = pytesseract.image_to_string(thresh, config=custom_config)
        
        os.unlink(tmp_path)
        return text.strip()
    except Exception as e:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        st.error(f"Erro no OCR: {str(e)}")
        return ""

# ==============================================
# 3. Interface e Lógica Principal (Atualizada)
# ==============================================

def main():
    st.title("🤖 Assistente de Implementação VTEX/Shopify")
    st.caption("Transforme designs em código pronto para produção")
    
    with st.sidebar:
        st.header("⚙️ Configurações")
        platform = st.selectbox("Plataforma", ["VTEX", "Shopify"], index=0)
        
        # Configuração da API OpenAI
        st.subheader("🔑 Configuração da API OpenAI")
        model = st.selectbox(
            "Modelo OpenAI",
            ["gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
            index=0,
            help="Selecione o modelo da OpenAI a ser utilizado"
        )
        
        openai_api_key = st.text_input(
            "Chave da API OpenAI",
            type="password",
            help="Insira sua chave da API OpenAI aqui"
        )
        
        st.subheader("📚 Documentação Técnica")
        
        # Upload de XMLs
        xml_files = st.file_uploader(
            "Envie arquivos XML",
            type=["xml"],
            accept_multiple_files=True,
            help="Arquivos XML com documentação técnica"
        )
        
        # Input para URLs
        url_input = st.text_area(
            "Cole URLs de documentação (uma por linha)",
            height=100,
            help="Exemplo: https://developers.vtex.com/docs/guides"
        )
        urls = [url.strip() for url in url_input.split('\n') if url.strip()]
        
        st.subheader("🖼️ Layout do Produto")
        image_file = st.file_uploader(
            "Envie o screenshot", 
            type=["png", "jpg", "jpeg"],
            help="Imagem do layout que deseja converter"
        )
    
    if (xml_files or urls) and image_file:
        if not openai_api_key:
            st.error("❌ Por favor, insira sua chave da API OpenAI na barra lateral")
            st.stop()
            
        with st.spinner("Processando..."):
            try:
                # Processa documentação
                db = load_docs(xml_files, urls, openai_api_key)
                if not db:
                    st.stop()
                
                # Processa imagem
                text = extract_text_from_image(image_file)
                
                if not text:
                    st.warning("Não foi possível extrair texto da imagem")
                    st.stop()
                
                # Gera código com exemplos específicos
                examples = {
                    "VTEX": """
                    ```json
                    // Exemplo de configuração VTEX
                    {
                      "storeSettings": {
                        "theme": "store-theme",
                        "currency": "BRL"
                      }
                    }
                    ```
                    """,
                    "Shopify": """
                    ```liquid
                    <!-- Exemplo de template Shopify -->
                    {% for product in collections.frontpage.products limit:4 %}
                      <div class="product-card">
                        {{ product.title | escape }}
                      </div>
                    {% endfor %}
                    ```
                    """
                }.get(platform, "")
                
                prompt = f"""
                Você é um especialista em {platform}. Com base em:
                
                1. Documentação técnica fornecida
                2. Texto extraído do layout:
                   {text}
                
                Gere:
                
                A. Código {platform} PRONTO PARA USO
                B. Configurações necessárias
                C. Boas práticas para implementação
                
                {examples}
                
                Formato de resposta:
                ```{platform.lower()}
                [código aqui]
                ```
                ```config
                [configurações aqui]
                ```
                ```tips
                [boas práticas aqui]
                ```
                """
                
                # Executa a consulta
                qa = RetrievalQA.from_chain_type(
                    llm=ChatOpenAI(
                        model=model,
                        temperature=0.3,
                        openai_api_key=openai_api_key
                    ),
                    retriever=db.as_retriever(),
                    chain_type="stuff"
                )
                response = qa.run(prompt)
                
                # Exibe resultados
                st.subheader(f"🧑‍💻 Código Gerado para {platform}")
                st.code(response, language=platform.lower())
                
                # Ações
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📋 Copiar Código", use_container_width=True):
                        pyperclip.copy(response)
                        st.toast("Código copiado!", icon="✅")
                with col2:
                    st.download_button(
                        "💾 Baixar Código",
                        data=response,
                        file_name=f"codigo_{platform.lower()}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
            except Exception as e:
                st.error(f"Erro durante o processamento: {str(e)}")
    else:
        st.warning("Por favor, envie pelo menos um XML ou URL de documentação + uma imagem de layout")

if __name__ == "__main__":
    main()