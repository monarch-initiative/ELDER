import json

from typing import Optional

import numpy as np

from venomx.model.venomx import Index, Model
from pheval_elder.metadata.metadata import Metadata


def normalize_metadata(metadata):
    """
    Normalize metadata downloaded from huggingface. Transformation to parquet forces nested
    lists to be turned into array type so we flatten those again.
    """

    def convert_array(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, list):
            return [convert_array(item) for item in obj]
        if isinstance(obj, dict):
            return {k: convert_array(v) for k, v in obj.items()}
        return obj

    return object_metadata(convert_array(metadata))


def object_metadata(obj):
    """
    Transform an object into metadata suitable for storage in chromadb.

    chromadb does not allow nested objects, so in addition to storing the
    top level keys that are primitive, we also store a json representation
    of the entire object at the top level with the key "_json".

    :param obj:
    :return:
    """
    dict_obj = _dict(obj)
    dict_obj["_json"] = json.dumps(dict_obj)
    return {
        k: v for k, v in dict_obj.items() if not isinstance(v, (dict, list)) and v is not None
    }


def _dict(obj):
    if isinstance(obj, dict):
        return obj
    else:
        raise ValueError(f"Cannot convert {obj} to dict")


@staticmethod
def populate_venomx(collection: Optional[str], model: Optional[str], existing_venomx: Index) -> Index:
    venomx = Index(
        id=f"{collection}",
        embedding_model=Model(
            name=model
        )
    )
    if existing_venomx:
        existing_venomx = Metadata(venomx=existing_venomx.model_dump(exclude_none=True))
        venomx = venomx.model_copy(update=existing_venomx.model_dump())
    return venomx
