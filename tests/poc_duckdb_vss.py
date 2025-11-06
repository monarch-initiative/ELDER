"""
Proof of Concept: DuckDB VSS for ELDER
Tests whether DuckDB Vector Similarity Search works on this platform
"""
import duckdb
import numpy as np
import sys


def test_duckdb_vss():
    """Test DuckDB VSS with ELDER-like data"""
    print("=" * 60)
    print("DuckDB VSS Proof of Concept")
    print("=" * 60)

    try:
        # Connect to in-memory database
        print("\n1. Connecting to DuckDB...")
        conn = duckdb.connect(':memory:')
        print("   ✓ Connected successfully")

        # Install VSS extension
        print("\n2. Installing VSS extension...")
        conn.execute("INSTALL vss")
        print("   ✓ VSS extension installed")

        print("\n3. Loading VSS extension...")
        conn.execute("LOAD vss")
        print("   ✓ VSS extension loaded")

        # Create embeddings table (using 1536 dimensions like OpenAI ada-002)
        print("\n4. Creating embeddings table (1536 dimensions)...")
        conn.execute("""
            CREATE TABLE disease_embeddings (
                id VARCHAR PRIMARY KEY,
                disease_id VARCHAR,
                disease_name VARCHAR,
                embedding FLOAT[1536],
                metadata JSON
            )
        """)
        print("   ✓ Table created")

        # Insert test embeddings (mock disease embeddings)
        print("\n5. Inserting 100 test embeddings...")
        test_embeddings = np.random.rand(100, 1536).astype(np.float32)

        for i, emb in enumerate(test_embeddings):
            conn.execute("""
                INSERT INTO disease_embeddings VALUES (?, ?, ?, ?, ?)
            """, (
                f"disease_{i}",
                f"OMIM:{1000+i}",
                f"Disease {i}",
                emb.tolist(),
                {"source": "test", "category": "rare_disease"}
            ))
        print("   ✓ 100 embeddings inserted")

        # Create HNSW index with cosine similarity
        print("\n6. Creating HNSW index (cosine metric)...")
        conn.execute("""
            CREATE INDEX emb_idx ON disease_embeddings
            USING HNSW (embedding)
            WITH (metric = 'cosine')
        """)
        print("   ✓ HNSW index created")

        # Query similar embeddings
        print("\n7. Testing similarity query...")
        query_vector = np.random.rand(1536).astype(np.float32)

        results = conn.execute("""
            SELECT
                disease_id,
                disease_name,
                array_cosine_similarity(embedding, ?::FLOAT[1536]) as similarity_score
            FROM disease_embeddings
            ORDER BY similarity_score DESC
            LIMIT 10
        """, (query_vector.tolist(),)).fetchall()

        print("   ✓ Query executed successfully")
        print("\n   Top 10 similar diseases:")
        for disease_id, disease_name, score in results:
            print(f"      {disease_id}: {disease_name} - {score:.4f}")

        # Test other similarity measures
        print("\n8. Testing L2 (Euclidean) distance...")
        conn.execute("DROP INDEX emb_idx")
        conn.execute("""
            CREATE INDEX emb_idx_l2 ON disease_embeddings
            USING HNSW (embedding)
            WITH (metric = 'l2sq')
        """)

        results_l2 = conn.execute("""
            SELECT disease_id, array_distance(embedding, ?::FLOAT[1536]) as distance
            FROM disease_embeddings
            ORDER BY distance ASC
            LIMIT 5
        """, (query_vector.tolist(),)).fetchall()
        print("   ✓ L2 distance works")

        print("\n9. Testing Inner Product...")
        conn.execute("DROP INDEX emb_idx_l2")
        conn.execute("""
            CREATE INDEX emb_idx_ip ON disease_embeddings
            USING HNSW (embedding)
            WITH (metric = 'ip')
        """)

        results_ip = conn.execute("""
            SELECT disease_id, array_inner_product(embedding, ?::FLOAT[1536]) as score
            FROM disease_embeddings
            ORDER BY score DESC
            LIMIT 5
        """, (query_vector.tolist(),)).fetchall()
        print("   ✓ Inner product works")

        # Test metadata filtering
        print("\n10. Testing metadata filtering...")
        results_filtered = conn.execute("""
            SELECT disease_id, array_cosine_similarity(embedding, ?::FLOAT[1536]) as score
            FROM disease_embeddings
            WHERE JSON_EXTRACT_STRING(metadata, '$.source') = 'test'
            ORDER BY score DESC
            LIMIT 5
        """, (query_vector.tolist(),)).fetchall()
        print(f"   ✓ Metadata filtering works ({len(results_filtered)} results)")

        conn.close()

        print("\n" + "=" * 60)
        print("✅ SUCCESS: DuckDB VSS is fully functional!")
        print("=" * 60)
        print("\nValidation Results:")
        print("  ✓ DuckDB VSS installs correctly")
        print("  ✓ Can create tables with 1536-dim vectors")
        print("  ✓ HNSW indexes work for all metrics (cosine, L2, IP)")
        print("  ✓ Similarity queries execute successfully")
        print("  ✓ Metadata filtering works")
        print("\nConclusion: DuckDB VSS migration is FEASIBLE ✅")

        return True

    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        print("\nValidation FAILED - DuckDB VSS migration not feasible")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_duckdb_vss()
    sys.exit(0 if success else 1)
