#!/usr/bin/env python3
"""
Simple runner for Figure 5 analysis
"""

import sys
from pathlib import Path
from figure_5_performance_as_func_of_nr_of_pheno_p_disease import (
    parse_hpoa_file, 
    get_diseases_with_results, 
    create_disease_complexity_table,
    bin_diseases_by_phenotype_count,
    create_phenotype_complexity_plot
)

def main():
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    results_dir = script_dir.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    hpoa_file = data_dir / "phenotype.hpoa"
    db_path = data_dir / "Best_Match_Cosine_all_combined_Elder_vs_Exomiser.db"
    
    print("FIGURE 5 ANALYSIS: HPOA PHENOTYPE COMPLEXITY")
    print("=" * 50)
    
    # Parse HPOA
    print("1. Parsing HPOA file...")
    disease_phenotype_counts, disease_names = parse_hpoa_file(hpoa_file)
    
    # Query database
    print("\n2. Querying database...")
    disease_performance = get_diseases_with_results(db_path)
    
    # Create table
    print("\n3. Creating disease table...")
    disease_table = create_disease_complexity_table(
        disease_phenotype_counts, disease_names, disease_performance
    )
    
    if len(disease_table) == 0:
        print("No overlapping diseases found!")
        return
    
    # Bin diseases
    print("\n4. Binning by phenotype count...")
    binned_diseases = bin_diseases_by_phenotype_count(disease_table)
    
    # Create plot (disable showing for script)
    print("\n5. Creating plot...")
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    create_phenotype_complexity_plot(binned_diseases, results_dir, save_plots=True)
    
    # Save results
    print("\n6. Saving results...")
    output_csv = results_dir / 'figure_5_elder_vs_ont-only_pheno_p_disease.csv'
    disease_table.to_csv(output_csv, index=False)
    print(f"Disease table saved: {output_csv}")
    
    # Bin summary
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
        import pandas as pd
        bin_summary_df = pd.DataFrame(bin_summary)
        bin_csv = results_dir / 'figure_5_hpoa_bin_summary.csv'
        bin_summary_df.to_csv(bin_csv, index=False)
        print(f"Bin summary saved: {bin_csv}")
        
        print(f"\nFINAL RESULTS:")
        print("=" * 40)
        print(bin_summary_df.to_string(index=False))
    
    print(f"\nAnalysis complete! Results in: {results_dir}")

if __name__ == "__main__":
    main()