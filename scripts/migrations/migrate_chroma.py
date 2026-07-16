import chromadb
import os
from config import AppConfig

config = AppConfig()

old_path = os.path.join(config.DATA_PATH, "database")
old_client = chromadb.PersistentClient(path=old_path)
old_collection = old_client.get_collection("langchain")

all_data = old_collection.get(include=["embeddings", "documents", "metadatas"])

new_path = config.DATABASE_PATH
new_client = chromadb.PersistentClient(path=new_path)
new_collection = new_client.create_collection(
    name="langchain",
    metadata={"hnsw:space": "cosine"}
)

batch_size = 5000
for i in range(0, len(all_data["ids"]), batch_size):
    new_collection.add(
        ids=all_data["ids"][i:i+batch_size],
        embeddings=all_data["embeddings"][i:i+batch_size],
        documents=all_data["documents"][i:i+batch_size],
        metadatas=all_data["metadatas"][i:i+batch_size],
    )

print(f"Migrated {len(all_data['ids'])} vectors to a cosine-configured collection at {new_path} — $0 cost")
