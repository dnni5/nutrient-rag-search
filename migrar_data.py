import pickle
import requests
from getpass import getpass

def simple_migrate():
    """Migración usando API REST directa"""
    
    cloud_url = input("URL de Qdrant Cloud")
    api_key = getpass("API Key: ")
    
    # Cargar documentos
    with open("embeddings_cache_nutricion_v1.pkl", 'rb') as f:
        documents = pickle.load(f)
    
    print(f"Cargados {len(documents)} documentos")
    
    # API REST para crear colección
    headers = {"api-key": api_key, "Content-Type": "application/json"}
    
    # Crear colección con el nombre
    collection_config = {
        "vectors": {
            "size": 768,
            "distance": "Cosine"
        }
    }
    
    response = requests.put(
        f"{cloud_url}/collections/nutricion_v1",
        json=collection_config,
        headers=headers
    )
    
    print(f"Colección creada: {response.status_code}")
    
    # Subir documentos usando haystack con la colección 
    from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
    from haystack.components.writers import DocumentWriter
    from haystack.utils import Secret
    

    store = QdrantDocumentStore(
        url=cloud_url + ":6333",  
        api_key=Secret.from_token(api_key),
        index="nutricion_v1"  
    )
    
    writer = DocumentWriter(document_store=store)
    
    # Subir en lotes de 50
    for i in range(0, len(documents), 50):
        batch = documents[i:i+50]
        print(f"Subiendo lote {i//50 + 1}...")
        try:
            writer.run(documents=batch)
            print(f"Lote {i//50 + 1} completado")
        except Exception as e:
            print(f"Error en lote {i//50 + 1}: {e}")

    # Verificar documentos en la colección 
    final_count = store.count_documents()
    print(f"Migración completada!")
    print(f"Documentos en colección 'nutricion_v1': {final_count}")

    # Verificar via API REST
    try:
        response = requests.get(
            f"{cloud_url}/collections/nutricion_v1",
            headers=headers
        )
        
        if response.status_code == 200:
            collection_info = response.json()
            result = collection_info.get("result", {})
            
            vector_count = (result.get("vectors_count") or 
                          result.get("points_count") or 
                          result.get("indexed_vectors_count") or 
                          "No disponible")
            
            print(f"Verificación via API REST: {vector_count} vectores")
            print(f"Info completa: {result}")
        else:
            print(f"No se pudo verificar via API REST (código: {response.status_code})")

    except Exception as e:
        print(f"Error en verificación via API REST: {e}")
        print("Pero la migración con Haystack fue exitosa!")

    return final_count

# Función para verificar la colección
def verify_collection():
    """Verificar el estado de la colección"""
    cloud_url = input("URL de Qdrant Cloud (sin :6333): ")
    api_key = getpass("API Key: ")
    
    headers = {"api-key": api_key, "Content-Type": "application/json"}
    
    # Listar todas las colecciones
    response = requests.get(f"{cloud_url}/collections", headers=headers)
    
    if response.status_code == 200:
        collections = response.json()["result"]["collections"]
        print("Colecciones disponibles:")
        for col in collections:
            print(f"  - {col['name']}: {col.get('vectors_count', 0)} vectores")
    
    # Verificar colección específica
    response = requests.get(f"{cloud_url}/collections/nutricion_v1", headers=headers)
    
    if response.status_code == 200:
        info = response.json()["result"]
        print(f"\nDetalles de 'nutricion_v1':")
        print(f"  Vectores: {info.get('vectors_count', 0)}")
        print(f"  Estado: {info.get('status', 'unknown')}")
    else:
        print(f"Error accediendo a colección: {response.status_code}")

if __name__ == "__main__":
    print("1. Migrar documentos")
    print("2. Verificar colección")
    choice = input("Selecciona opción (1 o 2): ")
    
    if choice == "1":
        simple_migrate()
    elif choice == "2":
        verify_collection()
    else:
        print("Opción no válida")