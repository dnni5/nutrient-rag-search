from rag_utils import index_documents_to_qdrant

# Configuración para la indexación local
# Conexión al servidor Qdrant que se está ejecutando en Docker
QDRANT_URL = "http://localhost:6333"
PDF_DIR = "ruta/pdfs"
METADATA_FILE = "ruta/articles"
QDRANT_COLLECTION = "nutricion_v1"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

if __name__ == "__main__":
    # Se llama a la función con el argumento 'qdrant_url'
    index_documents_to_qdrant(
        qdrant_url=QDRANT_URL,
        pdf_dir=PDF_DIR,
        metadata_path=METADATA_FILE,
        collection_name=QDRANT_COLLECTION,
        embedder_model=EMBEDDING_MODEL
    )