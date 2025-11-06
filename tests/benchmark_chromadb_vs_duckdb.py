"""
Performance Benchmark: ChromaDB vs DuckDB VSS
Compares insert and query performance for ELDER-scale workloads
"""
import time
import chromadb
import duckdb
import numpy as np


def benchmark_chromadb(n_embeddings=10000, n_queries=100, dim=1536):
    """Benchmark ChromaDB performance"""
    print(f"\n📊 Benchmarking ChromaDB ({n_embeddings} embeddings, {n_queries} queries, {dim} dims)...")

    client = chromadb.Client()
    # Use unique collection name to avoid conflicts
    collection_name = f"test_{n_embeddings}_{int(time.time())}"
    collection = client.create_collection(collection_name)

    # Benchmark insert
    print("   Inserting embeddings...")
    start = time.time()
    batch_size = 100
    for batch_start in range(0, n_embeddings, batch_size):
        batch_end = min(batch_start + batch_size, n_embeddings)
        batch_ids = [f"id_{i}" for i in range(batch_start, batch_end)]
        batch_embeddings = [np.random.rand(dim).tolist() for _ in range(batch_end - batch_start)]
        batch_metadatas = [{"disease_id": f"OMIM:{i}"} for i in range(batch_start, batch_end)]

        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas
        )
    insert_time = time.time() - start

    # Benchmark query
    print("   Querying embeddings...")
    start = time.time()
    for _ in range(n_queries):
        results = collection.query(
            query_embeddings=[np.random.rand(dim).tolist()],
            n_results=10
        )
    query_time = time.time() - start

    return {
        "insert_time": insert_time,
        "query_time": query_time,
        "insert_per_sec": n_embeddings / insert_time,
        "query_per_sec": n_queries / query_time
    }


def benchmark_duckdb(n_embeddings=10000, n_queries=100, dim=1536):
    """Benchmark DuckDB VSS performance"""
    print(f"\n📊 Benchmarking DuckDB VSS ({n_embeddings} embeddings, {n_queries} queries, {dim} dims)...")

    conn = duckdb.connect(':memory:')
    conn.execute("INSTALL vss")
    conn.execute("LOAD vss")

    conn.execute(f"""
        CREATE TABLE embeddings (
            id VARCHAR PRIMARY KEY,
            embedding FLOAT[{dim}],
            metadata JSON
        )
    """)

    # Benchmark insert
    print("   Inserting embeddings...")
    start = time.time()
    batch_size = 100
    for batch_start in range(0, n_embeddings, batch_size):
        batch_end = min(batch_start + batch_size, n_embeddings)
        for i in range(batch_start, batch_end):
            conn.execute("""
                INSERT INTO embeddings VALUES (?, ?, ?)
            """, (f"id_{i}", np.random.rand(dim).tolist(), f'{{"disease_id": "OMIM:{i}"}}'))
    insert_time = time.time() - start

    # Create index
    print("   Creating HNSW index...")
    index_start = time.time()
    conn.execute("""
        CREATE INDEX emb_idx ON embeddings
        USING HNSW (embedding) WITH (metric = 'cosine')
    """)
    index_time = time.time() - index_start

    # Benchmark query
    print("   Querying embeddings...")
    start = time.time()
    for _ in range(n_queries):
        results = conn.execute(f"""
            SELECT id, array_cosine_similarity(embedding, ?::FLOAT[{dim}]) as score
            FROM embeddings
            ORDER BY score DESC
            LIMIT 10
        """, (np.random.rand(dim).tolist(),)).fetchall()
    query_time = time.time() - start

    conn.close()
    return {
        "insert_time": insert_time,
        "index_time": index_time,
        "query_time": query_time,
        "insert_per_sec": n_embeddings / insert_time,
        "query_per_sec": n_queries / query_time
    }


def run_benchmarks():
    """Run all benchmarks and display results"""
    print("=" * 70)
    print("Performance Benchmark: ChromaDB vs DuckDB VSS")
    print("=" * 70)

    # Small scale test (fast)
    print("\n🔬 Small Scale Test (1,000 embeddings, 50 queries)")
    print("-" * 70)
    chroma_small = benchmark_chromadb(n_embeddings=1000, n_queries=50, dim=1536)
    duckdb_small = benchmark_duckdb(n_embeddings=1000, n_queries=50, dim=1536)

    # Medium scale test (ELDER-like)
    print("\n🔬 Medium Scale Test (10,000 embeddings, 100 queries)")
    print("-" * 70)
    chroma_medium = benchmark_chromadb(n_embeddings=10000, n_queries=100, dim=1536)
    duckdb_medium = benchmark_duckdb(n_embeddings=10000, n_queries=100, dim=1536)

    # Results table
    print("\n" + "=" * 70)
    print("📈 BENCHMARK RESULTS")
    print("=" * 70)

    results = []

    # Small scale results
    results.append(["Small Scale", "ChromaDB", "Insert",
                   f"{chroma_small['insert_time']:.2f}s",
                   f"{chroma_small['insert_per_sec']:.0f}/s", "-"])
    results.append(["", "DuckDB", "Insert",
                   f"{duckdb_small['insert_time']:.2f}s",
                   f"{duckdb_small['insert_per_sec']:.0f}/s",
                   f"{chroma_small['insert_time']/duckdb_small['insert_time']:.2f}x"])

    results.append(["", "ChromaDB", "Query",
                   f"{chroma_small['query_time']:.2f}s",
                   f"{chroma_small['query_per_sec']:.0f}/s", "-"])
    results.append(["", "DuckDB", "Query",
                   f"{duckdb_small['query_time']:.2f}s",
                   f"{duckdb_small['query_per_sec']:.0f}/s",
                   f"{chroma_small['query_time']/duckdb_small['query_time']:.2f}x"])

    # Medium scale results
    results.append(["Medium Scale", "ChromaDB", "Insert",
                   f"{chroma_medium['insert_time']:.2f}s",
                   f"{chroma_medium['insert_per_sec']:.0f}/s", "-"])
    results.append(["", "DuckDB", "Insert",
                   f"{duckdb_medium['insert_time']:.2f}s",
                   f"{duckdb_medium['insert_per_sec']:.0f}/s",
                   f"{chroma_medium['insert_time']/duckdb_medium['insert_time']:.2f}x"])
    results.append(["", "DuckDB", "Index Creation",
                   f"{duckdb_medium['index_time']:.2f}s", "-", "-"])

    results.append(["", "ChromaDB", "Query",
                   f"{chroma_medium['query_time']:.2f}s",
                   f"{chroma_medium['query_per_sec']:.0f}/s", "-"])
    results.append(["", "DuckDB", "Query",
                   f"{duckdb_medium['query_time']:.2f}s",
                   f"{duckdb_medium['query_per_sec']:.0f}/s",
                   f"{chroma_medium['query_time']/duckdb_medium['query_time']:.2f}x"])

    # Print results in a formatted table
    print(f"\n{'Scale':<15} {'Database':<10} {'Operation':<15} {'Time':<10} {'Rate':<12} {'Speedup':<10}")
    print("-" * 80)
    for row in results:
        print(f"{row[0]:<15} {row[1]:<10} {row[2]:<15} {row[3]:<10} {row[4]:<12} {row[5]:<10}")

    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)

    avg_insert_speedup = (chroma_small['insert_time']/duckdb_small['insert_time'] +
                          chroma_medium['insert_time']/duckdb_medium['insert_time']) / 2
    avg_query_speedup = (chroma_small['query_time']/duckdb_small['query_time'] +
                         chroma_medium['query_time']/duckdb_medium['query_time']) / 2

    print(f"\nAverage Insert Performance:")
    if avg_insert_speedup > 1:
        print(f"  ✅ DuckDB is {avg_insert_speedup:.2f}x FASTER than ChromaDB")
    else:
        print(f"  ⚠️  ChromaDB is {1/avg_insert_speedup:.2f}x faster than DuckDB")

    print(f"\nAverage Query Performance:")
    if avg_query_speedup > 1:
        print(f"  ✅ DuckDB is {avg_query_speedup:.2f}x FASTER than ChromaDB")
    elif avg_query_speedup > 0.5:
        print(f"  ⚠️  DuckDB is {1/avg_query_speedup:.2f}x slower (but within acceptable 2x range)")
    else:
        print(f"  ❌ DuckDB is {1/avg_query_speedup:.2f}x slower (CONCERN)")

    print(f"\nAdditional DuckDB Overhead:")
    print(f"  Index creation: ~{duckdb_medium['index_time']:.1f}s for 10K embeddings")

    # Validation decision
    print("\n" + "=" * 70)
    print("🎯 VALIDATION DECISION")
    print("=" * 70)

    # Check if query performance is within 2x
    if avg_query_speedup > 0.5:  # DuckDB within 2x of ChromaDB
        print("\n✅ PASS: DuckDB VSS meets performance requirements")
        print("   Query performance is comparable to ChromaDB (within 2x)")
        print("\n🚦 RECOMMENDATION: PROCEED with migration")
    else:
        print("\n❌ FAIL: DuckDB VSS does not meet performance requirements")
        print(f"   Query performance is {1/avg_query_speedup:.1f}x slower (exceeds 2x threshold)")
        print("\n🚦 RECOMMENDATION: DO NOT PROCEED with migration")

    print("=" * 70)


if __name__ == "__main__":
    run_benchmarks()
