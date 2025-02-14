import logging
from typing import List, Dict, Any

import chromadb
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def head_data(collection, db_path):
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(collection)
    res = collection.get(include=["embeddings", "metadatas", "documents"], limit=1)
    for metadata in res.get("metadatas", []):
        print(metadata)

    for embedding in res.get("embeddings", []):
        print(embedding)

    print(res)
    # Prepare lists to accumulate data.
    metadata_list: List[Any] = []
    embeddings_list: List[Any] = []
    documents_list: List[Any] = []

    # Depending on the ChromaDB API, the result may be a dict or an iterable.
    if isinstance(res, dict):
        metadata_list = res.get('metadatas', [])
        embeddings_list = res.get('embeddings', []).tolist()
        documents_list = res.get('documents', [])
    else:
        # If result is an iterable of tuples/lists, unpack each item.
        for item in res:
            try:
                meta, embed, doc = item
                metadata_list.append(meta)
                embeddings_list.append(embed)
                documents_list.append(doc)
            except Exception as e:
                logger.error("Unexpected item format: %s. Error: %s", item, e)

    # Wrap all the collected data into the Pydantic model.
    return ChromaResult(
        metadata=metadata_list,
        embeddings=embeddings_list,
        documents=documents_list
    )

class ChromaResult(BaseModel):
     metadata: Any
     embeddings: Any
     documents: Any


if __name__ == "__main__":
    model_name ="ada002"
    COLLECTION_NAME = "ada002_lrd_hpo_embeddings"
    # COLLECTION_NAME = "lrd_hpo"
    DB_PATH = f"/Users/ck/Monarch/elder/emb_data/models/{model_name}"
    # DB_PATH = "/Users/ck/Desktop/chromadb/ada-002-hp"


    result  = head_data(COLLECTION_NAME, DB_PATH)
    print(result.model_dump_json(indent=2))
    # except Exception as error:
    #     logger.exception("An error occurred while fetching head data from ChromaDB.")
