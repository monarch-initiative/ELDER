#!/usr/bin/env python3
"""
Phenotype Complexity Analysis - Figure 4

This script analyzes the performance of ELDER vs Ontology-only approaches
across different phenotype complexity bins.

Usage:
    python phenotype_complexity_analysis.py [dataset_name]

Example:
    python phenotype_complexity_analysis.py normalized_phenopackets_07
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import duckdb


def stratify_dataset_by_phenotypes_per_case(normalized_phenopacket_dir, db_path):
    """Analyze performance across phenotype complexity bins."""
    # bins = {
    #     "1-5": [],
    #     "10-15": [],
    #     "15-20": [],
    #     "20-25": [],
    #     "25-30": [],
    #     "30-35": [],
    #     "35-40": [],
    #     "40-45": [],
    #     "45+": [],
    # }

    bins = {
        "1-10": [],
        "10-20": [],
        "20-30": [],
        "30-40": [],
        "40+": [],
    }
    
    phenopacket_path = Path(normalized_phenopacket_dir)
    elder_per_case = {}
    exomiser_per_case = {}
    file_to_disease = {}
    
    for json_file in phenopacket_path.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                phenopacket = json.load(f)
            
            if 'diseases' in phenopacket and phenopacket['diseases']:
                disease_id = phenopacket['diseases'][0]['term']['id']
                file_to_disease[json_file.name] = disease_id
            
            phenotype_count = 0
            if 'phenotypicFeatures' in phenopacket:
                phenotype_count = len(phenopacket['phenotypicFeatures'])
            
            filename = json_file.name
            # if 1 <= phenotype_count <= 5:
            #     bins["1-5"].append(filename)
            # elif 10 <= phenotype_count <= 15:
            #     bins["10-15"].append(filename)
            # elif 16 <= phenotype_count <= 20:
            #     bins["15-20"].append(filename)
            # elif 21 <= phenotype_count <= 25:
            #     bins["20-25"].append(filename)
            # elif 26 <= phenotype_count <= 30:
            #     bins["25-30"].append(filename)
            # elif 31 <= phenotype_count <= 35:
            #     bins["30-35"].append(filename)
            # elif 36 <= phenotype_count <= 40:
            #     bins["35-40"].append(filename)
            # elif 41 <= phenotype_count <= 45:
            #     bins["40-45"].append(filename)
            # elif phenotype_count > 45:
            #     bins["45+"].append(filename)
            if 1 <= phenotype_count <= 10:
                bins["1-10"].append(filename)
            elif 10 <= phenotype_count <= 20:
                bins["10-20"].append(filename)
            elif 21 <= phenotype_count <= 30:
                bins["20-30"].append(filename)
            elif 31 <= phenotype_count <= 40:
                bins["30-40"].append(filename)
            elif phenotype_count > 40:
                bins["40+"].append(filename)
        
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    conn = duckdb.connect(db_path)
    
    all_files = []
    for bin_files in bins.values():
        all_files.extend(bin_files)
    
    if all_files:
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
                    exomiser_per_case[filename] = exomiser_top1
                    elder_per_case[filename] = elder_top1
                else:
                    exomiser_per_case[filename] = 0
                    elder_per_case[filename] = 0
            
            except Exception as e:
                print(f"Database query error for {filename}: {e}")
                exomiser_per_case[filename] = 0
                elder_per_case[filename] = 0
    
    conn.close()
    
    elder_bin_accuracies = {}
    exomiser_bin_accuracies = {}
    
    for bin_name, files in bins.items():
        if files:
            elder_accuracies = [elder_per_case.get(f, 0) for f in files]
            exomiser_accuracies = [exomiser_per_case.get(f, 0) for f in files]
            
            elder_bin_accuracies[bin_name] = sum(elder_accuracies) / len(elder_accuracies) if elder_accuracies else 0
            exomiser_bin_accuracies[bin_name] = sum(exomiser_accuracies) / len(exomiser_accuracies) if exomiser_accuracies else 0
            
            print(f"Bin {bin_name}: {len(files)} files")
            print(f"  ELDER avg top-1 accuracy: {elder_bin_accuracies[bin_name]:.3f}")
            print(f"  Ontology-only avg top-1 accuracy: {exomiser_bin_accuracies[bin_name]:.3f}")
    
    plt.figure(figsize=(14, 8))
    bin_names = list(elder_bin_accuracies.keys())
    elder_accuracies = list(elder_bin_accuracies.values())
    exomiser_accuracies = list(exomiser_bin_accuracies.values())
    
    x = np.arange(len(bin_names))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, exomiser_accuracies, width, label='Ontology-only', 
                   color='lightcoral', edgecolor='black', alpha=0.7)
    bars2 = plt.bar(x + width/2, elder_accuracies, width, label='ELDER', 
                   color='skyblue', edgecolor='black', alpha=0.7)
    
    plt.xlabel('Phenotype Count Bins')
    plt.ylabel('Average Top-1 Accuracy')
    plt.ylim(0, 1.0)
    plt.xticks(x, bin_names)
    plt.legend()
    
    for bar, accuracy in zip(bars1, exomiser_accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{accuracy:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    for bar, accuracy in zip(bars2, elder_accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{accuracy:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    output_dir = Path("../results")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'figure_4_performance_as_func_of_nr_of_pheno_p_case.svg', bbox_inches='tight')
    plt.savefig(output_dir / 'figure_4_performance_as_func_of_nr_of_pheno_p_case.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return elder_per_case, exomiser_per_case, elder_bin_accuracies, exomiser_bin_accuracies


def main():
    """Main function to run the analysis."""
    if len(sys.argv) < 2:
        dataset_name = "../data/normalized_phenopackets_07"
        print(f"No dataset specified, using default: {dataset_name}")
    else:
        dataset_name = sys.argv[1]
    
    db_path = "../data/Best_Match_Cosine_all_combined_Elder_vs_Exomiser.db"
    
    if not Path(dataset_name).exists():
        print(f"Error: Dataset {dataset_name} not found!")
        print("Available datasets in ../data/:")
        data_dir = Path("../data")
        if data_dir.exists():
            for item in data_dir.iterdir():
                if item.is_dir() and item.name.startswith("normalized_phenopackets"):
                    print(f"  {item.name}")
        return 1
    
    if not Path(db_path).exists():
        print(f"Error: Database {db_path} not found!")
        print("Please download the database from: https://zenodo.org/records/16944913")
        print("Place it in the ../data/ directory")
        return 1
    
    print(f"Running analysis on dataset: {dataset_name}")
    stratify_dataset_by_phenotypes_per_case(dataset_name, db_path)
    print("Analysis complete! Check ../results/ for output files.")
    return 0


if __name__ == "__main__":
    exit(main())