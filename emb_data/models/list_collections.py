import sys

import chromadb

def list_collections(db_path):
    client = chromadb.PersistentClient(db_path)
    collections = client.list_collections()
    for c in collections:
        print(c.name)

if __name__ == "__main__":
    model = "mxbai-l"
    db_path = f"/Users/ck/Monarch/elder/emb_data/models/{model}"
    list_collections(db_path)