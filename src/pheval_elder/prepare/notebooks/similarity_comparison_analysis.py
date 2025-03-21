import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple

original_high_resnik = pd.read_csv('/Users/ck/Monarch/elder/src/pheval_elder/prepare/notebooks/original_high_resnik_low_cosine.csv')
original_low_resnik = pd.read_csv('/Users/ck/Monarch/elder/src/pheval_elder/prepare/notebooks/original_low_resnik_high_cosine.csv')
enhanced_high_resnik = pd.read_csv('/Users/ck/Monarch/elder/src/pheval_elder/prepare/notebooks/enhanced_high_resnik_low_cosine.csv')
enhanced_low_resnik = pd.read_csv('/Users/ck/Monarch/elder/src/pheval_elder/prepare/notebooks/enhanced_low_resnik_high_cosine.csv')

# create unique term pair identifier
def create_pair_id(row):
    terms = sorted([row['subject'], row['object']])
    return f"{terms[0]}_{terms[1]}"

# Add pair IDs to all dataframes
for df in [original_high_resnik, original_low_resnik, enhanced_high_resnik, enhanced_low_resnik]:
    df['pair_id'] = df.apply(create_pair_id, axis=1)

# Analysis 1: Compare cosine similarity value ranges
def compare_cosine_ranges():
    orig_high = original_high_resnik['cosine'].describe()
    enh_high = enhanced_high_resnik['cosine'].describe()
    
    orig_low = original_low_resnik['cosine'].describe()
    enh_low = enhanced_low_resnik['cosine'].describe()
    
    print("High Resnik / Low Cosine Statistics:")
    comparison = pd.DataFrame({
        'Original': orig_high,
        'Enhanced': enh_high,
        'Difference': enh_high - orig_high
    })
    print(comparison)
    
    print("\nLow Resnik / High Cosine Statistics:")
    comparison = pd.DataFrame({
        'Original': orig_low,
        'Enhanced': enh_low,
        'Difference': enh_low - orig_low
    })
    print(comparison)

# Analysis 2: Check for overlapping term pairs between original and enhanced datasets
def analyze_overlapping_pairs():
    # For high Resnik/low cosine
    high_resnik_overlap = set(original_high_resnik['pair_id']).intersection(set(enhanced_high_resnik['pair_id']))
    
    # For low Resnik/high cosine
    low_resnik_overlap = set(original_low_resnik['pair_id']).intersection(set(enhanced_low_resnik['pair_id']))
    
    print(f"High Resnik/Low Cosine overlapping pairs: {len(high_resnik_overlap)} out of {len(original_high_resnik)} original and {len(enhanced_high_resnik)} enhanced pairs")
    print(f"Low Resnik/High Cosine overlapping pairs: {len(low_resnik_overlap)} out of {len(original_low_resnik)} original and {len(enhanced_low_resnik)} enhanced pairs")
    
    return high_resnik_overlap, low_resnik_overlap

# Analysis 3: For overlapping pairs, how much did the cosine similarity change?
def analyze_similarity_improvements(high_resnik_overlap, low_resnik_overlap):
    # Create lookup dictionaries for faster access
    orig_high_lookup = original_high_resnik.set_index('pair_id')['cosine'].to_dict()
    enh_high_lookup = enhanced_high_resnik.set_index('pair_id')['cosine'].to_dict()
    
    orig_low_lookup = original_low_resnik.set_index('pair_id')['cosine'].to_dict()
    enh_low_lookup = enhanced_low_resnik.set_index('pair_id')['cosine'].to_dict()
    
    # Calculate changes for high Resnik pairs
    high_changes = []
    for pair_id in high_resnik_overlap:
        high_changes.append(enh_high_lookup[pair_id] - orig_high_lookup[pair_id])
    
    # Calculate changes for low Resnik pairs
    low_changes = []
    for pair_id in low_resnik_overlap:
        low_changes.append(enh_low_lookup[pair_id] - orig_low_lookup[pair_id])
    
    print("\nHigh Resnik/Low Cosine cosine similarity changes:")
    print(f"Mean change: {np.mean(high_changes):.4f}")
    print(f"Median change: {np.median(high_changes):.4f}")
    print(f"Min change: {min(high_changes):.4f}")
    print(f"Max change: {max(high_changes):.4f}")
    
    print("\nLow Resnik/High Cosine cosine similarity changes:")
    print(f"Mean change: {np.mean(low_changes):.4f}")
    print(f"Median change: {np.median(low_changes):.4f}")
    print(f"Min change: {min(low_changes):.4f}")
    print(f"Max change: {max(low_changes):.4f}")
    
    return high_changes, low_changes

# Analysis 4: Identify the top term pairs that showed the most improvement
def identify_top_improvements(high_resnik_overlap, low_resnik_overlap, n=10):
    # Create dataframes with all information about overlapping pairs
    high_overlap_data = []
    for pair_id in high_resnik_overlap:
        # Get first matching row for each DataFrame
        orig_row = original_high_resnik[original_high_resnik['pair_id'] == pair_id].iloc[0]
        enh_row = enhanced_high_resnik[enhanced_high_resnik['pair_id'] == pair_id].iloc[0]
        
        high_overlap_data.append({
            'pair_id': pair_id,
            'subject_label': orig_row['subject_label'],
            'object_label': orig_row['object_label'],
            'original_cosine': orig_row['cosine'],
            'enhanced_cosine': enh_row['cosine'],
            'cosine_change': enh_row['cosine'] - orig_row['cosine'],
            'resnik': orig_row['resnik']
        })
    
    high_overlap_df = pd.DataFrame(high_overlap_data)
    
    # Sort by improvement (greatest positive change in cosine similarity)
    high_improved = high_overlap_df.sort_values('cosine_change', ascending=False).head(n)
    high_worsened = high_overlap_df.sort_values('cosine_change', ascending=True).head(n)
    
    print("\nTop High Resnik/Low Cosine term pairs with greatest improvement:")
    for _, row in high_improved.iterrows():
        print(f"{row['subject_label']} - {row['object_label']}: {row['original_cosine']:.4f} → {row['enhanced_cosine']:.4f} (change: +{row['cosine_change']:.4f})")
    
    print("\nTop High Resnik/Low Cosine term pairs with greatest decline:")
    for _, row in high_worsened.iterrows():
        print(f"{row['subject_label']} - {row['object_label']}: {row['original_cosine']:.4f} → {row['enhanced_cosine']:.4f} (change: {row['cosine_change']:.4f})")
    
    # Do the same for low Resnik/high cosine pairs if there are overlapping pairs
    if len(low_resnik_overlap) > 0:
        low_overlap_data = []
        for pair_id in low_resnik_overlap:
            # Get first matching row for each DataFrame
            orig_row = original_low_resnik[original_low_resnik['pair_id'] == pair_id].iloc[0]
            enh_row = enhanced_low_resnik[enhanced_low_resnik['pair_id'] == pair_id].iloc[0]
            
            low_overlap_data.append({
                'pair_id': pair_id,
                'subject_label': orig_row['subject_label'],
                'object_label': orig_row['object_label'],
                'original_cosine': orig_row['cosine'],
                'enhanced_cosine': enh_row['cosine'],
                'cosine_change': enh_row['cosine'] - orig_row['cosine'],
                'resnik': orig_row['resnik']
            })
        
        low_overlap_df = pd.DataFrame(low_overlap_data)
        
        low_improved = low_overlap_df.sort_values('cosine_change', ascending=False).head(n)
        low_worsened = low_overlap_df.sort_values('cosine_change', ascending=True).head(n)
        
        print("\nTop Low Resnik/High Cosine term pairs with greatest improvement:")
        for _, row in low_improved.iterrows():
            print(f"{row['subject_label']} - {row['object_label']}: {row['original_cosine']:.4f} → {row['enhanced_cosine']:.4f} (change: +{row['cosine_change']:.4f})")
        
        print("\nTop Low Resnik/High Cosine term pairs with greatest decline:")
        for _, row in low_worsened.iterrows():
            print(f"{row['subject_label']} - {row['object_label']}: {row['original_cosine']:.4f} → {row['enhanced_cosine']:.4f} (change: {row['cosine_change']:.4f})")

# Analysis 5: Analyze new term pairs that only appear in enhanced embeddings
def analyze_new_pairs():
    # For high Resnik/low cosine
    original_high_pairs = set(original_high_resnik['pair_id'])
    enhanced_high_pairs = set(enhanced_high_resnik['pair_id'])
    
    new_high_pairs = enhanced_high_pairs - original_high_pairs
    
    # For low Resnik/high cosine
    original_low_pairs = set(original_low_resnik['pair_id'])
    enhanced_low_pairs = set(enhanced_low_resnik['pair_id'])
    
    new_low_pairs = enhanced_low_pairs - original_low_pairs
    
    print(f"\nNew High Resnik/Low Cosine pairs in enhanced embeddings: {len(new_high_pairs)} out of {len(enhanced_high_pairs)} total")
    print(f"New Low Resnik/High Cosine pairs in enhanced embeddings: {len(new_low_pairs)} out of {len(enhanced_low_pairs)} total")
    
    # Look at top 5 new high Resnik/low cosine pairs
    if len(new_high_pairs) > 0:
        new_high_df = enhanced_high_resnik[enhanced_high_resnik['pair_id'].isin(new_high_pairs)]
        print("\nSample of new High Resnik/Low Cosine pairs:")
        for _, row in new_high_df.head(5).iterrows():
            print(f"{row['subject_label']} - {row['object_label']}: Resnik = {row['resnik']:.4f}, Cosine = {row['cosine']:.4f}")
    
    # Look at top 5 new low Resnik/high cosine pairs
    if len(new_low_pairs) > 0:
        new_low_df = enhanced_low_resnik[enhanced_low_resnik['pair_id'].isin(new_low_pairs)]
        print("\nSample of new Low Resnik/High Cosine pairs:")
        for _, row in new_low_df.head(5).iterrows():
            print(f"{row['subject_label']} - {row['object_label']}: Resnik = {row['resnik']:.4f}, Cosine = {row['cosine']:.4f}")

def main():
    print("=== COMPARISON ANALYSIS: ORIGINAL VS ENHANCED EMBEDDINGS ===\n")
    
    # Run analysis 1: Compare value ranges
    print("\n--- ANALYSIS 1: COSINE SIMILARITY VALUE RANGES ---")
    compare_cosine_ranges()
    
    # Run analysis 2: Check for overlapping pairs
    print("\n--- ANALYSIS 2: OVERLAPPING TERM PAIRS ---")
    high_resnik_overlap, low_resnik_overlap = analyze_overlapping_pairs()
    
    # Run analysis 3: Analyze similarity improvements
    print("\n--- ANALYSIS 3: COSINE SIMILARITY CHANGES ---")
    high_changes, low_changes = analyze_similarity_improvements(high_resnik_overlap, low_resnik_overlap)
    
    # Run analysis 4: Identify top improvements
    print("\n--- ANALYSIS 4: TOP IMPROVEMENTS AND DECLINES ---")
    identify_top_improvements(high_resnik_overlap, low_resnik_overlap)
    
    # Run analysis 5: Analyze new pairs
    print("\n--- ANALYSIS 5: NEW TERM PAIRS IN ENHANCED EMBEDDINGS ---")
    analyze_new_pairs()
    
    # Create summary statement
    print("\n=== SUMMARY ===")
    avg_high_change = np.mean(high_changes) if high_changes else 0
    avg_low_change = np.mean(low_changes) if low_changes else 0
    
    if avg_high_change > 0:
        high_msg = f"improved by {avg_high_change:.4f} on average"
    else:
        high_msg = f"decreased by {abs(avg_high_change):.4f} on average"
    
    if avg_low_change > 0:
        low_msg = f"improved by {avg_low_change:.4f} on average"
    else:
        low_msg = f"decreased by {abs(avg_low_change):.4f} on average"
    
    print(f"• For High Resnik/Low Cosine pairs, cosine similarity {high_msg}")
    print(f"• For Low Resnik/High Cosine pairs, cosine similarity {low_msg}")
    
    # Calculate percentage of improved pairs
    improved_high = sum(1 for change in high_changes if change > 0)
    improved_low = sum(1 for change in low_changes if change > 0)
    
    if high_changes:
        print(f"• {improved_high/len(high_changes)*100:.1f}% of High Resnik/Low Cosine pairs had improved cosine similarity")
    if low_changes:
        print(f"• {improved_low/len(low_changes)*100:.1f}% of Low Resnik/High Cosine pairs had improved cosine similarity")
    
    # # Overall conclusion
    # if avg_high_change > 0 and avg_low_change > 0:
    #     print("\nCONCLUSION: The enhanced embeddings generally show improved alignment with Resnik similarity,")
    #     print("though the magnitude of improvements appears modest.")
    # elif avg_high_change > 0 or avg_low_change > 0:
    #     print("\nCONCLUSION: The enhanced embeddings show mixed results, with improvements in some areas")
    #     print("but not in others. The overall change is relatively modest.")
    # else:
    #     print("\nCONCLUSION: The enhanced embeddings do not show substantial improvements in alignment with")
    #     print("Resnik semantic similarity. Further refinement of the embedding approach may be needed.")

if __name__ == "__main__":
    main()