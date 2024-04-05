import pytest
from pheval_elder.prepare.core.graph_embedding_extractor import GraphEmbeddingExtractor
from pheval_elder.prepare.core.graph_data_processor import GraphDataProcessor
from pheval_elder.prepare.core.chromadb_manager import ChromaDBManager
from pheval_elder.prepare.core.data_processor import DataProcessor


def test_parse_deep_embeddings():
    extractor = GraphEmbeddingExtractor()
    embeddings = extractor.parse_deep_embeddings()
    head = extractor.filtered_deepwalk_df
    assert embeddings is not None
    assert len(embeddings) > 0
    # print(head)


def test_parse_line_embeddings():
    extractor = GraphEmbeddingExtractor()
    embeddings = extractor.parse_line_embeddings()
    # head = extractor.filtered_line_df
    head = extractor.filtered_line_df
    assert embeddings is not None
    assert len(embeddings) > 0
    # print(head)


def test_dict():
    db_manager = ChromaDBManager()
    extractor = GraphEmbeddingExtractor()
    graph_processor = GraphDataProcessor(manager=db_manager, extractor=extractor)
    graph_processor.line_graph_embeddings()
    base_processor = DataProcessor(db_manager=db_manager)
    bd = base_processor.hp_embeddings
    print(len(bd))
    d = graph_processor.get_line_embeddings_dict()
    for i, (k, v) in enumerate(d.items()):
        if i > 5:
            break
    assert len(d) == 17884
    assert len(d) > 0
    assert len(bd) > 0
    assert len(bd) == 29912
    assert type(d) == dict
