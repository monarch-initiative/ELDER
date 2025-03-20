# Indexing Ontologies with CurateGPT

This document explains how to use the `curate-index` command to index ontologies using CurateGPT's functionality, for later use with ELDER's analysis tools.

## Overview

The `curate-index` command provides a direct integration with CurateGPT's ontology indexing capabilities, allowing you to:

1. Index ontologies with standard descriptions
2. Index ontologies with enhanced descriptions using OpenAI's o1 model
3. Search indexed collections
4. View information about indexed collections

This tool is particularly valuable for creating high-quality embeddings of the Human Phenotype Ontology (HP), which are used by ELDER's disease similarity analysis tools.

## Prerequisites

- CurateGPT installed (included as a dependency in ELDER)
- OpenAI API key (if using enhanced descriptions)
- An ontology to index (HP ontology recommended)

## Setting Up

Set your OpenAI API key if you plan to use enhanced descriptions:

```bash
export OPENAI_API_KEY=your-api-key-here
```

## Basic Usage

### Indexing an Ontology

```bash
# Index HP ontology with standard descriptions
curate-index index-ontology --db-path ./my_db --collection hp_standard

# Index with enhanced descriptions for richer semantic understanding
curate-index index-ontology --enhanced-descriptions --collection hp_enhanced

# Specify which fields to include in the embedding
curate-index index-ontology --index-fields "label,definition,relationships,aliases"

# Use a different embedding model
curate-index index-ontology --model ada002
curate-index index-ontology --model large3
curate-index index-ontology --model bge-m3
```

### Searching Indexed Terms

```bash
# Search for terms in a collection
curate-index search --collection hp_standard "cardiac arrhythmia"

# Search in a collection that was indexed with enhanced descriptions
curate-index search --collection hp_enhanced --enhanced "heart defect"
```

### Viewing Collection Information

```bash
# List all collections in a database
curate-index info

# Show details of a specific collection
curate-index info --collection hp_enhanced
```

## Model Options

The `curate-index` tool supports various embedding models through shorthand names:

| Shorthand | Model |
|-----------|-------|
| `ada` or `ada002` | OpenAI's text-embedding-ada-002 |
| `small3` | OpenAI's text-embedding-3-small |
| `large3` | OpenAI's text-embedding-3-large |
| `bge-m3` | BAAI/bge-m3 |
| `nomic` | nomic-embed-text-v1.5 |
| `mxbai-l` | mxbai/embed-large-v1 |
| `all-MiniLM-L6-v2` | SentenceTransformers model |
| `all-mpnet-base-v2` | SentenceTransformers model |

You can also use Ollama models with the prefix `ollama:`, e.g., `ollama:llama2`.

## Using with ELDER

After indexing an ontology, you can use the resulting collection with ELDER's disease similarity analysis tools:

```bash
# Run ELDER with the indexed collection
elder average --db-path ./my_db --collection hp_enhanced --model large3
```

## Enhanced Descriptions

When using the `--enhanced-descriptions` flag, the tool will:

1. Use OpenAI's o1 model to generate rich, detailed descriptions for each term
2. Incorporate these descriptions into the embeddings
3. Result in more semantically meaningful vectors that capture clinical context

This is particularly valuable for rare phenotypes that might have limited information in the standard HP definitions.

## Performance Considerations

- Indexing with enhanced descriptions will be slower and incur OpenAI API costs
- The first-time indexing with enhanced descriptions will be the slowest
- For large ontologies, consider indexing in smaller batches