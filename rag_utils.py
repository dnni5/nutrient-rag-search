import os
import pandas as pd
import pickle
from pathlib import Path
import requests
import time

# Importaciones de Haystack 
from haystack import Pipeline, Document
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.converters import PyPDFToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.writers import DocumentWriter
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.utils import Secret

# Importaciones de Qdrant
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever

# Importación del Generador de Mistral AI
from haystack_integrations.components.generators.mistral import MistralChatGenerator

def save_embeddings_cache(documents, cache_file="embeddings_cache.pkl"):
    """Guarda los documentos con embeddings en un archivo cache"""
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(documents, f)
        print(f"Cache de embeddings guardado en: {cache_file}")
    except Exception as e:
        print(f"Error al guardar cache: {e}")

def load_embeddings_cache(cache_file="embeddings_cache.pkl"):
    """Carga los documentos con embeddings desde el cache"""
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                documents = pickle.load(f)
            print(f"Cache de embeddings cargado desde: {cache_file}")
            return documents
        else:
            print(f"ℹNo se encontró cache en: {cache_file}")
            return None
    except Exception as e:
        print(f"Error al cargar cache: {e}")
        return None

def index_documents_to_qdrant(qdrant_url: str, pdf_dir: str, metadata_path: str, collection_name: str, embedder_model: str, use_cache=True):
    """
    Procesa y guarda los documentos en un servidor de base de datos Qdrant.
    """
    print("\n INICIANDO FASE DE INDEXACIÓN OFFLINE")
    cache_file = f"embeddings_cache_{collection_name}.pkl"
    embedded_documents = None
    
    # Verificar cache
    if use_cache:
        embedded_documents = load_embeddings_cache(cache_file)
        if embedded_documents:
            print(f"Usando embeddings del cache ({len(embedded_documents)} documentos)")
            return write_to_qdrant_only(qdrant_url, embedded_documents, collection_name)
    
    
    # Verificar conexión con Qdrant
    print(f"Verificando conexión con Qdrant en {qdrant_url}...")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{qdrant_url}/health", timeout=10)
            if response.status_code == 200:
                print(f"Qdrant está disponible (intento {attempt + 1})")
                break
        except Exception as e:
            print(f"Intento {attempt + 1} fallido: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)

    # Cargar metadatos
    try:
        metadata_df = pd.read_csv(metadata_path, dtype=str).fillna("")
        if 'FileName' in metadata_df.columns:
            metadata_df.rename(columns={'FileName': 'filename'}, inplace=True)
        assert 'filename' in metadata_df.columns, "La columna 'filename' es obligatoria."
        print(f"Metadatos cargados: {len(metadata_df)} registros")
    except Exception as e:
        print(f"Error al cargar metadatos: {e}")
        return
    
    metadata_map = {row['filename']: row.to_dict() for _, row in metadata_df.iterrows()}
    
    # Buscar PDFs válidos
    valid_pdfs = []
    for filename_from_csv in metadata_map.keys():
        pdf_name = filename_from_csv if filename_from_csv.endswith('.pdf') else filename_from_csv + ".pdf"
        path_to_check = os.path.join(pdf_dir, pdf_name)
        if os.path.isfile(path_to_check):
            valid_pdfs.append(path_to_check)
    
    if not valid_pdfs:
        print(f"No se encontraron PDFs para indexar en '{pdf_dir}'.")
        return
    print(f"Procesando {len(valid_pdfs)} archivos PDF...")
    
    # Convertir PDFs a documentos
    try:
        converter = PyPDFToDocument()
        docs_from_pdf = converter.run(sources=valid_pdfs)["documents"]
        print(f"Convertidos {len(docs_from_pdf)} páginas de PDF")
    except Exception as e:
        print(f"Error al convertir PDFs: {e}")
        return
    
    # Agregar metadatos
    for doc in docs_from_pdf:
        filename_full = os.path.basename(doc.meta['file_path'])
        filename_base = os.path.splitext(filename_full)[0]
        if filename_full in metadata_map:
            doc.meta.update(metadata_map[filename_full])
        elif filename_base in metadata_map:
            doc.meta.update(metadata_map[filename_base])
    
    # Dividir documentos
    splitter = DocumentSplitter(split_by="word", split_length=200, split_overlap=50)
    split_docs = splitter.run(documents=docs_from_pdf)["documents"]
    print(f"Divididos en {len(split_docs)} fragmentos")
    
    # Generar embeddings
    try:
        embedder = SentenceTransformersDocumentEmbedder(model=embedder_model)
        embedder.warm_up()
        embedded_docs = embedder.run(documents=split_docs)["documents"]
        print(f"Embeddings generados para {len(embedded_docs)} fragmentos")
        
        # GUARDAR CACHE después de generar embeddings
        if use_cache:
            save_embeddings_cache(embedded_docs, cache_file)
    except Exception as e:
        print(f"Error al generar embeddings: {e}")
        return
    
    # Escribir a Qdrant
    return write_to_qdrant_only(qdrant_url, embedded_docs, collection_name)

def write_to_qdrant_only(qdrant_url: str, embedded_documents, collection_name: str):
    """Escribe documentos ya procesados a Qdrant"""
    print(f"\n Escribiendo {len(embedded_documents)} documentos a Qdrant...")
    try:
        # Inicializar document store 
        document_store = QdrantDocumentStore(
            url=qdrant_url,
            embedding_dim=768
        )

        # Escribir documentos
        writer = DocumentWriter(document_store=document_store, policy="OVERWRITE")
        result = writer.run(documents=embedded_documents)
        
        doc_count = document_store.count_documents()
        print(f"\n Escritura completada exitosamente!")
        print(f"{doc_count} fragmentos guardados en Qdrant")
        return True
        
    except Exception as e:
        print(f"Error al escribir a Qdrant: {e}")
        return False

def resume_indexing_from_cache(qdrant_url: str, collection_name: str):
    """Función para reanudar solo la escritura a Qdrant usando el cache"""
    cache_file = f"embeddings_cache_{collection_name}.pkl"
    embedded_documents = load_embeddings_cache(cache_file)
    
    if not embedded_documents:
        print("No se encontró cache de embeddings.")
        print("   Ejecuta la indexación completa primero.")
        return False
    
    return write_to_qdrant_only(qdrant_url, embedded_documents, collection_name)