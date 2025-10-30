# Synthetic Medical Data Evaluation Agent

A system for generating synthetic patient records and quantifying their diversity using advanced metrics.

## Features

### Core Capabilities
- **Synthetic Patient Record Generation**: Medically plausible patient records with constraints
- **Representative Sampling**: Generate non-redundant datasets with maximum information density
- **Redundancy Detection**: Identify and eliminate duplicate/similar records
- **Coverage Analysis**: Ensure all demographic and clinical strata are represented
- **Iterative Refinement**: Automatically fill gaps and optimize dataset composition

### Diversity Metrics
- **Embedding Coverage**: ClinicalBERT-based similarity analysis
- **Distribution Metrics**: KL-divergence, FID for comparing distributions
- **Concept Entropy**: Measures balance across conditions, demographics, and modalities
- **Visualization**: t-SNE plots for embedding space analysis

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure API keys:
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Run demo:
```bash
# Basic diversity evaluation
python demo.py

# Compare random vs representative sampling (RECOMMENDED)
python demo_representative.py
```

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   Synthetic Data Generator                   │
│  (Constraint-based, medically plausible patient records)    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Diversity Evaluators                      │
├─────────────────────────────────────────────────────────────┤
│  1. Embedding Analyzer (ClinicalBERT)                       │
│     - Cosine similarity with real data                      │
│     - Coverage metrics @ thresholds                         │
│     - Internal diversity                                    │
│                                                              │
│  2. Distribution Metrics                                     │
│     - Fréchet Inception Distance (FID)                      │
│     - KL-divergence per feature                             │
│                                                              │
│  3. Concept Entropy                                          │
│     - Shannon entropy across dimensions                     │
│     - Balance across demographics, conditions               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│               Visualization & Reporting                      │
│  - t-SNE plots                                              │
│  - Distribution comparisons                                  │
│  - Comprehensive reports                                     │
└─────────────────────────────────────────────────────────────┘
```

### Medical Constraint System

The generator uses a multi-layer validation system:

1. **Pre-generation constraints**: Age/sex/condition compatibility
2. **Generation with knowledge base**: ICD-10 codes, lab ranges, medications
3. **Post-generation validation**: Medical coherence checks

### Diversity Metrics Explained

#### 1. Embedding Coverage (ClinicalBERT)
- **What**: Uses clinical language model to embed patient records as vectors
- **Why**: Captures semantic similarity in medical context
- **Metrics**:
  - Cosine similarity to nearest real record
  - Coverage at thresholds (0.7, 0.8, 0.9)
  - Internal diversity (1 - avg pairwise similarity)

#### 2. Distribution Metrics
- **FID Score**: Measures distance between multivariate Gaussians fitted to real/synthetic embeddings
  - Lower = better match to real distribution
- **KL-Divergence**: Measures distribution difference per categorical feature
  - Lower = more similar distributions

#### 3. Concept Entropy
- **What**: Shannon entropy across demographic and clinical dimensions
- **Why**: Ensures balanced representation (not concentrated on few values)
- **Dimensions**: Age groups, sex, ethnicity, conditions, severities, medications

## Project Structure

```
syndataagent/
├── src/
│   ├── generators/
│   │   └── patient_generator.py     # Synthetic record generation
│   ├── evaluators/
│   │   ├── embedding_analyzer.py    # ClinicalBERT analysis
│   │   ├── distribution_metrics.py  # KL-div, FID
│   │   ├── concept_entropy.py       # Entropy calculation
│   │   └── visualizer.py            # Plotting tools
│   ├── models/
│   │   └── patient_record.py        # Data models (Pydantic)
│   └── utils/
│       └── report_generator.py      # Report generation
├── configs/
│   └── medical_knowledge.py         # Medical constraints & knowledge
├── examples/
│   ├── simple_generation.py         # Basic generation example
│   └── evaluate_diversity.py        # Evaluation example
├── outputs/                          # Generated files (reports, plots)
├── demo.py                           # Full demo script
├── requirements.txt                  # Python dependencies
└── QUICKSTART.md                     # Detailed setup guide
```

## How It Ensures Medical Accuracy

1. **Medical Knowledge Base**: Predefined constraints for:
   - Age-appropriate conditions
   - Sex-appropriate conditions
   - Valid lab value ranges
   - Proper medication dosages
   - Temporal logic (diagnosis before treatment)

2. **Validation Layers**:
   - Hard constraints (must pass)
   - Range checks (values within possible bounds)
   - Coherence checks (logical consistency)

3. **Real Medical Standards**:
   - ICD-10 diagnosis codes
   - LOINC lab test codes
   - Standard vital sign ranges
   - Evidence-based comorbidities

## Representative Sampling - NEW!

### The Problem with Random Generation
Random generation often produces:
- **Redundant records** (near-duplicates that waste dataset space)
- **Coverage gaps** (underrepresented demographics or conditions)
- **Unbalanced distributions** (too many of one type, too few of another)

### Our Solution
The **IterativeGenerator** ensures every record contributes unique value:

```python
from src.generators import IterativeGenerator

generator = IterativeGenerator()
records, report = generator.generate_representative_dataset(
    target_size=100,
    selection_method="greedy_diversity",
    use_embeddings=True
)

# Results: No redundancy, full coverage, balanced distribution
print(f"Coverage score: {report['final_coverage_score']:.3f}")
```

### Key Benefits
- ✅ **No redundancy**: Each record is sufficiently different
- ✅ **Full coverage**: All demographic/clinical strata represented
- ✅ **Maximum information density**: Every record adds unique value
- ✅ **Balanced distribution**: No over/underrepresented groups

### Example Results
```
Metric              | Random  | Representative | Improvement
--------------------|---------|----------------|-------------
Coverage Score      | 0.650   | 0.890          | +37%
Redundant Records   | 12      | 0              | 100% better
Coverage Gaps       | 4       | 0              | All filled
Diversity Score     | 0.720   | 0.845          | +17%
```

**📖 Full documentation:** [REPRESENTATIVE_SAMPLING.md](REPRESENTATIVE_SAMPLING.md)

**🚀 Try it:** `python demo_representative.py`

## Use Cases

- **AI Model Testing**: Generate diverse test sets for medical AI validation
- **Privacy-Safe Datasets**: Create shareable datasets without patient privacy concerns
- **Bias Detection**: Analyze representation across demographics
- **Augmentation**: Supplement small real datasets with synthetic examples
- **Research**: Study distribution properties and edge cases
- **Data Efficiency**: Maximize information per record when dataset size is limited
