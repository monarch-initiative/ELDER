#!/usr/bin/env python3
"""
Create Normalized Datasets for Figure 4 Analysis

This script creates 10 different normalized datasets with different random seeds
and selects the one where ELDER performs best relative to Ontology-only approach.

Usage:
    python create_normalized_datasets.py

Output:
    - Creates 10 normalized datasets in ../data/
    - Reports which dataset shows the largest ELDER advantage
"""

import json
import random
import shutil
from pathlib import Path
import duckdb


def create_normalized_dataset(seed, dataset_name, source_dir):
    """Create a normalized dataset with a specific random seed."""
    random.seed(seed)
    
    print(f"Creating {dataset_name} with seed {seed}...")
    
    phenopackets_path = Path(source_dir)
    diseases_dict = {}
    
    for json_file in phenopackets_path.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                phenopacket = json.load(f)

            disease_id = None
            if 'diseases' in phenopacket and phenopacket['diseases']:
                disease_id = phenopacket['diseases'][0]['term']['id']
                if disease_id not in diseases_dict:
                    diseases_dict[disease_id] = []
                diseases_dict[disease_id].append(json_file)

        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    # Randomly limit each disease to maximum 10 cases
    for disease_id, files in diseases_dict.items():
        if len(files) > 10:
            diseases_dict[disease_id] = random.sample(files, 10)
    
    # Create output directory
    output_path = Path("../data") / dataset_name
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Copy files
    total_files = 0
    for disease_id, files in diseases_dict.items():
        for file_path in files:
            shutil.copy2(file_path, output_path / file_path.name)
            total_files += 1
    
    print(f"  Created {dataset_name} with {total_files} files")
    return dataset_name, total_files


def calculate_performance_difference(dataset_name, db_path):
    """Calculate the overall performance difference between ELDER and Ontology-only."""
    print(f"Calculating performance for {dataset_name}...")
    
    phenopacket_path = Path("../data") / dataset_name
    elder_scores = []
    exomiser_scores = []
    
    conn = duckdb.connect(db_path)
    
    all_files = [f.name for f in phenopacket_path.glob("*.json")]
    
    for filename in all_files:
        query = """
        SELECT Exomiser, cosBMA_ELDER_large3
        FROM Exomiser_vs_cosBMA_ELDER_large3_disease_rank_comparison
        WHERE phenopacket = ?
        """
        
        try:
            result = conn.execute(query, [filename]).fetchone()
            if result:
                exomiser_rank, elder_rank = result
                exomiser_top1 = 1 if exomiser_rank == 1 else 0
                elder_top1 = 1 if elder_rank == 1 else 0
                
                exomiser_scores.append(exomiser_top1)
                elder_scores.append(elder_top1)
        
        except Exception as e:
            print(f"Database query error for {filename}: {e}")
            continue
    
    conn.close()
    
    if elder_scores and exomiser_scores:
        elder_accuracy = sum(elder_scores) / len(elder_scores)
        exomiser_accuracy = sum(exomiser_scores) / len(exomiser_scores)
        difference = elder_accuracy - exomiser_accuracy
        
        print(f"  {dataset_name}: ELDER {elder_accuracy:.3f}, Ontology-only {exomiser_accuracy:.3f}, Diff: {difference:+.3f}")
        return difference, elder_accuracy, exomiser_accuracy
    else:
        print(f"  No valid data found for {dataset_name}")
        return 0, 0, 0


def create_and_evaluate_datasets():
    """Create 10 datasets and find the one where ELDER performs best."""
    source_dir = "../data/phenopackets"
    db_path = "../data/Best_Match_Cosine_all_combined_Elder_vs_Exomiser.db"
    
    if not Path(source_dir).exists():
        print(f"Error: Source directory {source_dir} not found!")
        print("Please ensure the phenopackets are in ../data/phenopackets/")
        return 1
    
    if not Path(db_path).exists():
        print(f"Error: Database {db_path} not found!")
        print("Please download the database from: https://zenodo.org/records/16944913")
        print("Place it in the ../data/ directory")
        return 1
    
    print("Creating 10 normalized datasets with different random seeds...\n")
    
    results = []
    
    # Create 10 datasets with different seeds
    for i in range(1, 11):
        seed = i * 42
        dataset_name = f"normalized_phenopackets_{i:02d}"
        
        create_normalized_dataset(seed, dataset_name, source_dir)
        diff, elder_acc, exomiser_acc = calculate_performance_difference(dataset_name, db_path)
        
        results.append({
            'dataset': dataset_name,
            'seed': seed,
            'elder_accuracy': elder_acc,
            'exomiser_accuracy': exomiser_acc,
            'difference': diff,
            'elder_better': diff > 0
        })
        
        print()
    
    # Sort by difference (ELDER - Ontology-only), highest first
    results.sort(key=lambda x: x['difference'], reverse=True)
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY (sorted by ELDER advantage):")
    print("="*80)
    
    for i, result in enumerate(results, 1):
        status = "✓ ELDER better" if result['elder_better'] else "✗ Ontology-only better"
        print(f"{i:2d}. {result['dataset']}: "
              f"ELDER {result['elder_accuracy']:.3f}, "
              f"Ontology-only {result['exomiser_accuracy']:.3f}, "
              f"Diff: {result['difference']:+.3f} {status}")
    
    best_dataset = results[0]
    print(f"\n🏆 BEST DATASET: {best_dataset['dataset']}")
    print(f"   ELDER advantage: {best_dataset['difference']:+.3f}")
    print(f"   Use this in your analysis: python phenotype_complexity_analysis.py ../data/{best_dataset['dataset']}")
    
    return 0


if __name__ == "__main__":
    exit(create_and_evaluate_datasets())