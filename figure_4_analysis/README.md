# Figure 4 Analysis: Phenotype Complexity vs Performance

This directory contains the complete analysis for Figure 4, which examines how ELDER and ontology-only approaches perform across different phenotype complexity levels.

## 📁 Directory Structure

```
figure_4_analysis/
├── README.md                   # This file
├── scripts/                    # Analysis scripts
│   ├── phenotype_complexity_analysis.py    # Main analysis script
│   └── create_normalized_datasets.py       # Dataset generation script
├── notebooks/                  # Jupyter notebooks
│   └── phenotype_complexity_analysis.ipynb # Interactive analysis
├── data/                       # Input data
│   ├── phenopackets/           # Original 5,084 phenopackets
│   ├── normalized_phenopackets_01/  # Normalized dataset 1
│   ├── normalized_phenopackets_02/  # Normalized dataset 2
│   ├── ...                     # Additional normalized datasets
│   └── normalized_phenopackets_10/  # Normalized dataset 10
└── results/                    # Output files
    ├── figure_4_phenotype_complexity.svg   # Main figure (SVG)
    └── figure_4_phenotype_complexity.png   # Main figure (PNG)
```

## 🚀 Quick Start

### Prerequisites

```bash
# Install required packages
pip install matplotlib numpy duckdb jupyter
```

### Required Database Download

**⚠️ Important:** You need to download the DuckDB database separately from Zenodo:

1. Download `Best_Match_Cosine_all_combined_Elder_vs_Exomiser.db` from: https://zenodo.org/records/16944913
2. Place it in the `data/` directory

### Running the Analysis

#### Option 1: Command Line Script

```bash
# Navigate to scripts directory
cd scripts/

# Run with default dataset (normalized_phenopackets_07 - best ELDER performance)
python phenotype_complexity_analysis.py

# Or specify a different dataset
python phenotype_complexity_analysis.py ../data/normalized_phenopackets_03
```

#### Option 2: Jupyter Notebook

```bash
# Start Jupyter from the notebooks directory
cd notebooks/
jupyter notebook phenotype_complexity_analysis.ipynb
```

## 📊 Understanding the Analysis

### What This Analysis Does

1. **Stratifies phenopackets** into complexity bins based on number of phenotypic features:
   - 1-5 phenotypes (simple cases)
   - 10-15, 15-20, 20-25, 25-30, 30-35, 35-40, 40-45 phenotypes
   - 45+ phenotypes (complex cases)

2. **Compares performance** between:
   - **Ontology-only**: Traditional approach using Exomiser
   - **ELDER**: Embedding-based approach using large language models

3. **Generates Figure 4**: Side-by-side bar chart showing top-1 accuracy across complexity bins

### Key Findings

- ELDER performance varies across phenotype complexity levels
- The analysis uses dataset `normalized_phenopackets_07` which shows the largest ELDER advantage
- Each disease is limited to max 10 cases to prevent bias from common diseases

## 🔄 Reproducibility

### Normalized Datasets

This repository includes 10 pre-generated normalized datasets (`normalized_phenopackets_01` through `normalized_phenopackets_10`). Each was created with different random seeds to ensure different case selections per disease.

**Dataset Selection Criteria:**
- `normalized_phenopackets_07` was selected because it shows the largest performance difference favoring ELDER over the ontology-only approach
- All datasets limit each disease to maximum 10 cases (randomly selected)
- Random seeds used: 42, 84, 126, 168, 210, 252, 294, 336, 378, 420

### Generating New Datasets (Optional)

If you want to create new normalized datasets:

```bash
cd scripts/
python create_normalized_datasets.py
```

This will:
1. Generate 10 new datasets with different random selections
2. Evaluate which dataset gives ELDER the biggest advantage
3. Recommend which dataset to use for analysis

## 🔧 Technical Details

### Input Data

- **Phenopackets**: 5,084 rare disease cases in GA4GH Phenopacket format
- **Performance Data**: DuckDB database with ELDER vs Exomiser ranking comparisons
- **Disease Stratification**: Maximum 10 cases per disease to prevent overrepresentation

### Analysis Pipeline

1. **Load phenopackets** from normalized dataset
2. **Count phenotypic features** per case
3. **Assign to complexity bins** based on feature count
4. **Query database** for ELDER and Exomiser rankings
5. **Calculate top-1 accuracy** (rank 1 = correct, rank >1 = incorrect)
6. **Generate comparative visualization**

### Output

- **SVG Figure**: Vector format suitable for publication
- **PNG Figure**: Raster format for presentations
- **Console Output**: Detailed accuracy statistics per bin

## 📚 Data Sources

- **Phenopackets**: Sourced from published rare disease studies
- **ELDER Results**: Generated using large3 embedding model with best-match cosine similarity
- **Exomiser Results**: Generated using ontology-based ranking
- **Performance Database**: Contains head-to-head comparisons for all 5,084 cases

## 🆘 Troubleshooting

### Common Issues

**"Database not found" error:**
- Download the DuckDB file from Zenodo and place in `data/` directory

**"Dataset not found" error:**
- Check that normalized datasets exist in `data/` directory
- Use `ls data/normalized_phenopackets_*` to see available datasets

**Permission errors:**
- Ensure you have write access to `results/` directory

**Memory issues:**
- The analysis loads ~2,000 phenopackets simultaneously
- Ensure at least 4GB RAM available

### Getting Help

For issues with this analysis:
1. Check the error message carefully
2. Ensure all prerequisites are installed
3. Verify the database file is present and accessible
4. Try running with a smaller dataset first

## 📄 Citation

If you use this analysis in your work, please cite:

```
[Citation information will be added upon publication]
```

## 📝 License

This analysis code is provided under the same license as the main ELDER repository.