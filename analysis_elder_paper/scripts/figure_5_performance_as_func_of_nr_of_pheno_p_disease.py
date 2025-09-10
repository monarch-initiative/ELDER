#!/usr/bin/env python3
"""
Figure 5 Analysis: Disease-Phenotype Complexity from HPOA Data

This script analyzes phenotype complexity directly from the phenotype.hpoa file,
counting actual phenotypes per disease and comparing ELDER vs ontology-only performance
across different phenotype complexity bins.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import duckdb
from pathlib import Path
from collections import defaultdict, Counter

def parse_hpoa_file(hpoa_path):
    """
    Parse the HPOA file and count phenotypes per disease.
    
    Returns:
        disease_phenotype_counts: Dict mapping disease_id -> phenotype count
        disease_names: Dict mapping disease_id -> disease name
    """
    disease_phenotypes = defaultdict(set)  # Use set to avoid duplicate phenotypes
    disease_names = {}
    
    print(f"Parsing HPOA file: {hpoa_path}")
    
    with open(hpoa_path, 'r', encoding='utf-8') as f:
        # Skip header lines that start with #
        for line in f:
            if line.startswith('#'):
                continue
            
            # Split the line by tabs
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            
            database_id = parts[0]  # e.g., OMIM:619340
            disease_name = parts[1]
            qualifier = parts[2]  # Can be empty
            hpo_id = parts[3]  # e.g., HP:0011097
            
            # Skip entries with qualifiers (negated phenotypes)
            if qualifier.strip():
                continue
            
            # Store disease name
            disease_names[database_id] = disease_name
            
            # Add phenotype to disease (set automatically handles duplicates)
            disease_phenotypes[database_id].add(hpo_id)
    
    # Convert sets to counts
    disease_phenotype_counts = {disease_id: len(phenotypes) 
                               for disease_id, phenotypes in disease_phenotypes.items()}
    
    print(f"Parsed {len(disease_phenotype_counts)} diseases")
    print(f"Total phenotype-disease associations: {sum(disease_phenotype_counts.values())}")
    
    # Show some statistics
    phenotype_counts = list(disease_phenotype_counts.values())
    print(f"\nPhenotype count statistics:")
    print(f"  Min: {min(phenotype_counts)}")
    print(f"  Max: {max(phenotype_counts)}")
    print(f"  Mean: {np.mean(phenotype_counts):.1f}")
    print(f"  Median: {np.median(phenotype_counts):.1f}")
    
    return disease_phenotype_counts, disease_names

def get_diseases_with_results(db_path):
    """
    Query the database to get all diseases that have performance results.
    
    Returns:
        disease_performance: Dict mapping disease_id -> {elder_rank, exomiser_rank}
    """
    print(f"Querying database: {db_path}")
    
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return {}
    
    conn = duckdb.connect(str(db_path))
    
    # Query using the correct disease_identifier column
    query = """
    SELECT disease_identifier, Exomiser, cosBMA_ELDER_large3
    FROM Exomiser_vs_cosBMA_ELDER_large3_disease_rank_comparison
    """
    
    try:
        results = conn.execute(query).fetchall()
        print(f"Found {len(results)} performance records")
        
        disease_performance = {}
        
        for disease_id, exomiser_rank, elder_rank in results:
            if disease_id:  # Skip if disease_id is None or empty
                if disease_id not in disease_performance:
                    # For diseases with multiple records, aggregate performance
                    disease_performance[disease_id] = {
                        'elder_ranks': [],
                        'exomiser_ranks': []
                    }
                
                disease_performance[disease_id]['elder_ranks'].append(elder_rank)
                disease_performance[disease_id]['exomiser_ranks'].append(exomiser_rank)
        
        # Calculate aggregated performance metrics (using best rank per disease)
        for disease_id in disease_performance:
            elder_ranks = disease_performance[disease_id]['elder_ranks']
            exomiser_ranks = disease_performance[disease_id]['exomiser_ranks']
            
            # Use best (lowest) rank for each method
            best_elder_rank = min(elder_ranks)
            best_exomiser_rank = min(exomiser_ranks)
            
            disease_performance[disease_id] = {
                'elder_rank': best_elder_rank,
                'exomiser_rank': best_exomiser_rank,
                'elder_top1': 1 if best_elder_rank == 1 else 0,
                'exomiser_top1': 1 if best_exomiser_rank == 1 else 0,
                'num_cases': len(elder_ranks)
            }
        
        conn.close()
        print(f"Extracted {len(disease_performance)} unique diseases with performance data")
        
        # Show sample disease IDs
        if disease_performance:
            sample_diseases = list(disease_performance.keys())[:5]
            print(f"Sample disease IDs: {sample_diseases}")
        
        return disease_performance
        
    except Exception as e:
        print(f"Database query error: {e}")
        conn.close()
        return {}

def create_disease_complexity_table(disease_phenotype_counts, disease_names, disease_performance):
    """
    Create a comprehensive table combining HPOA data with performance results.
    
    Returns:
        df: DataFrame with columns [disease_id, disease_name, phenotype_count, elder_top1, exomiser_top1]
    """
    table_data = []
    
    # Find intersection of diseases in both HPOA and database
    hpoa_diseases = set(disease_phenotype_counts.keys())
    db_diseases = set(disease_performance.keys())
    
    print(f"Diseases in HPOA: {len(hpoa_diseases)}")
    print(f"Diseases in database: {len(db_diseases)}")
    
    # Find overlap
    overlapping_diseases = hpoa_diseases.intersection(db_diseases)
    print(f"Overlapping diseases: {len(overlapping_diseases)}")
    
    # Create table with overlapping diseases
    for disease_id in overlapping_diseases:
        table_data.append({
            'disease_id': disease_id,
            'disease_name': disease_names.get(disease_id, 'Unknown'),
            'phenotype_count': disease_phenotype_counts[disease_id],
            'elder_top1': disease_performance[disease_id]['elder_top1'],
            'exomiser_top1': disease_performance[disease_id]['exomiser_top1'],
            'num_cases': disease_performance[disease_id]['num_cases']
        })
    
    df = pd.DataFrame(table_data)
    print(f"\nCreated table with {len(df)} diseases")
    
    if len(df) > 0:
        print(f"\nPhenotype count distribution:")
        print(df['phenotype_count'].describe())
        
        print(f"\nOverall accuracy:")
        print(f"  ELDER: {df['elder_top1'].mean():.3f}")
        print(f"  Exomiser: {df['exomiser_top1'].mean():.3f}")
        print(f"  Total cases: {df['num_cases'].sum()}")
    
    return df

def bin_diseases_by_phenotype_count(df):
    """
    Bin diseases by phenotype count using the same bins as the original analysis.
    
    Returns:
        binned_data: Dict mapping bin names to DataFrames of diseases in each bin
    """
    if len(df) == 0:
        return {}
    
    # Define the same bins as in the original analysis
    # bins = {
    #     "1-5": (1, 5),
    #     "10-15": (10, 15),
    #     "15-20": (16, 20),  # Note: 16-20, not 15-20
    #     "20-25": (21, 25),
    #     "25-30": (26, 30),
    #     "30-35": (31, 35),
    #     "35-40": (36, 40),
    #     "40-45": (41, 45),
    #     "45+": (46, float('inf'))
    # }

    bins = {
        "1-10": (1, 10),
        "10-20": (11, 20),
        "20-30": (21, 30),
        "30-40": (31, 40),
        "40+": (41, float('inf')),
        # "25-30": [],
        # "30-35": [],
        # "35-40": [],
        # "40-45": [],
        # "45+": [],
    }
    
    binned_data = {}
    
    print(f"\nBinning {len(df)} diseases by phenotype count:")
    
    for bin_name, (min_count, max_count) in bins.items():
        if max_count == float('inf'):
            bin_df = df[df['phenotype_count'] >= min_count].copy()
        else:
            bin_df = df[(df['phenotype_count'] >= min_count) & (df['phenotype_count'] <= max_count)].copy()
        
        binned_data[bin_name] = bin_df
        
        if len(bin_df) > 0:
            elder_acc = bin_df['elder_top1'].mean()
            exomiser_acc = bin_df['exomiser_top1'].mean()
            print(f"  {bin_name}: {len(bin_df)} diseases, ELDER: {elder_acc:.3f}, Exomiser: {exomiser_acc:.3f}")
        else:
            print(f"  {bin_name}: 0 diseases")
    
    return binned_data

def create_phenotype_complexity_plot(binned_data, results_dir, save_plots=True):
    """
    Create a bar chart showing accuracy by phenotype complexity bins.
    
    Parameters:
        binned_data: Dictionary mapping bin names to DataFrames
        results_dir: Path to save results
        save_plots: Whether to save the plots to files
    """
    if not binned_data:
        print("No data to plot!")
        return
    
    plt.figure(figsize=(14, 8))
    
    # Prepare data for plotting
    bin_names = list(binned_data.keys())
    elder_accuracies = []
    exomiser_accuracies = []
    bin_counts = []
    
    for bin_name in bin_names:
        bin_df = binned_data[bin_name]
        if len(bin_df) > 0:
            elder_accuracies.append(bin_df['elder_top1'].mean())
            exomiser_accuracies.append(bin_df['exomiser_top1'].mean())
            bin_counts.append(len(bin_df))
        else:
            elder_accuracies.append(0)
            exomiser_accuracies.append(0)
            bin_counts.append(0)
    
    # Create the plot
    x = np.arange(len(bin_names))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, exomiser_accuracies, width, label='Ontology-only',
                   color='lightcoral', edgecolor='black', alpha=0.7)
    bars2 = plt.bar(x + width/2, elder_accuracies, width, label='ELDER', 
                   color='skyblue', edgecolor='black', alpha=0.7)
    
    plt.xlabel('Phenotype Count Bins', fontsize=12)
    plt.ylabel('Top-1 Accuracy', fontsize=12)
    # plt.title('Performance vs Phenotype Complexity (HPOA Data)', fontsize=14, fontweight='bold')
    plt.ylim(0, 1.0)
    plt.xticks(x, bin_names)
    plt.legend(fontsize=11)
    
    # Add value labels on bars
    for i, (bar1, bar2, count) in enumerate(zip(bars1, bars2, bin_counts)):
        if count > 0:  # Only label bars with data
            plt.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.01,
                    f'{exomiser_accuracies[i]:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
            plt.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.01,
                    f'{elder_accuracies[i]:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Add count labels below bars
        plt.text(i, -0.08, f'n={count}', ha='center', va='top', fontsize=8, color='gray')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Add footer with total diseases
    total_diseases = sum(bin_counts)
    # plt.figtext(0.5, 0.02, f'Total diseases analyzed: {total_diseases}', ha='center', fontsize=10, style='italic')
    
    if save_plots:
        svg_path = results_dir / 'figure_5_performance_as_func_of_nr_of_pheno_p_disease.svg'
        png_path = results_dir / 'figure_5_performance_as_func_of_nr_of_pheno_p_disease.png'
        
        plt.savefig(svg_path, bbox_inches='tight')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        
        print(f"\nPlots saved:")
        print(f"  SVG: {svg_path}")
        print(f"  PNG: {png_path}")
    
    plt.show()
    
    # Calculate overall statistics
    non_empty_bins = [(name, elder_accuracies[i], exomiser_accuracies[i]) 
                     for i, name in enumerate(bin_names) if bin_counts[i] > 0]
    
    if non_empty_bins:
        avg_elder = np.mean([acc for _, acc, _ in non_empty_bins])
        avg_exomiser = np.mean([acc for _, _, acc in non_empty_bins])
        difference = avg_elder - avg_exomiser
        
        print(f"\nOverall Performance Summary:")
        print(f"  ELDER average: {avg_elder:.3f}")
        print(f"  Ontology-only average: {avg_exomiser:.3f}")
        print(f"  Difference: {difference:+.3f} ({'ELDER better' if difference > 0 else 'Ontology-only better'})")
        print(f"  Bins with data: {len(non_empty_bins)}/{len(bin_names)}")

def main():
    """Main analysis function"""
    
    # Set up paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    results_dir = script_dir.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    # File paths
    hpoa_file = data_dir / "phenotype.hpoa"
    db_path = data_dir / "Best_Match_Cosine_all_combined_Elder_vs_Exomiser.db"
    
    print("=" * 60)
    print("FIGURE 5 ANALYSIS: HPOA PHENOTYPE COMPLEXITY")
    print("=" * 60)
    print(f"HPOA file: {hpoa_file}")
    print(f"Database: {db_path}")
    print(f"Results dir: {results_dir}")
    
    # Step 1: Parse HPOA file
    print("\n" + "="*50)
    print("STEP 1: PARSING HPOA FILE")
    print("="*50)
    disease_phenotype_counts, disease_names = parse_hpoa_file(hpoa_file)
    
    # Step 2: Get diseases with performance results
    print("\n" + "="*50)
    print("STEP 2: QUERYING PERFORMANCE DATABASE")
    print("="*50)
    disease_performance = get_diseases_with_results(db_path)
    
    # Step 3: Create combined table
    print("\n" + "="*50)
    print("STEP 3: CREATING DISEASE COMPLEXITY TABLE")
    print("="*50)
    disease_table = create_disease_complexity_table(
        disease_phenotype_counts, disease_names, disease_performance
    )
    
    if len(disease_table) == 0:
        print("No overlapping diseases found. Cannot proceed with analysis.")
        return
    
    # Step 4: Bin diseases by phenotype count
    print("\n" + "="*50)
    print("STEP 4: BINNING DISEASES BY PHENOTYPE COUNT")
    print("="*50)
    binned_diseases = bin_diseases_by_phenotype_count(disease_table)
    
    # Step 5: Create plot
    print("\n" + "="*50)
    print("STEP 5: CREATING PHENOTYPE COMPLEXITY PLOT")
    print("="*50)
    create_phenotype_complexity_plot(binned_diseases, results_dir, save_plots=True)
    
    # Step 6: Save results
    print("\n" + "="*50)
    print("STEP 6: SAVING RESULTS")
    print("="*50)
    
    # Save disease table
    output_csv = results_dir / 'figure_5_performance_as_func_of_nr_of_pheno_p_disease.csv'
    disease_table.to_csv(output_csv, index=False)
    print(f"Disease complexity table saved to: {output_csv}")
    
    # Save bin summary
    bin_summary = []
    for bin_name, bin_df in binned_diseases.items():
        if len(bin_df) > 0:
            bin_summary.append({
                'bin': bin_name,
                'count': len(bin_df),
                'elder_accuracy': bin_df['elder_top1'].mean(),
                'exomiser_accuracy': bin_df['exomiser_top1'].mean(),
                'elder_advantage': bin_df['elder_top1'].mean() - bin_df['exomiser_top1'].mean()
            })
    
    if bin_summary:
        bin_summary_df = pd.DataFrame(bin_summary)
        bin_csv = results_dir / 'figure_5_hpoa_bin_summary.csv'
        bin_summary_df.to_csv(bin_csv, index=False)
        print(f"Bin summary saved to: {bin_csv}")
        
        print(f"\nFinal Bin Summary:")
        print(bin_summary_df.to_string(index=False))
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()