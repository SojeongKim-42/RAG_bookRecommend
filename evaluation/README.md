# Evaluation System

This directory contains the evaluation framework for the RAG Book Recommendation System. It provides tools for systematic testing, performance measurement, and experiment management.

## Overview

The evaluation system supports:
- **Automated evaluation** with predefined test queries
- **Multiple evaluation metrics** (genre accuracy, diversity, semantic similarity)
- **Experiment presets** for ablation studies and parameter sweeps
- **Result visualization** and analysis tools

## Directory Structure

```
evaluation/
├── evaluate.py           # Main experiment runner
├── metrics.py            # Evaluation metrics (Genre, Retrieval, Semantic)
├── dataset.py            # Test query dataset
├── utils.py              # Result saving and formatting utilities
├── viz.py                # Visualization tools
└── experiment_config.py  # Experiment configuration and presets
```

## Quick Start

### Running Predefined Experiments

```bash
# Baseline experiment
python evaluation/evaluate.py --preset baseline

# Ablation study (test each feature)
python evaluation/evaluate.py --preset ablation

# K-value sweep
python evaluation/evaluate.py --preset k_sweep

# MMR lambda parameter sweep
python evaluation/evaluate.py --preset lambda_sweep
```

### Custom Experiments

```bash
# Custom configuration
python evaluation/evaluate.py --custom \
    --k 5 \
    --mmr-lambda 0.8 \
    --query-type vague \
    --sample 10
```

## Evaluation Metrics

### 1. Genre Metrics
Measures how well the system retrieves books in the expected genres.

- **Precision**: Proportion of retrieved books in correct genres
- **Recall**: Coverage of expected genres
- **F1 Score**: Harmonic mean of precision and recall
- **Diversity**: Number of unique genres / total retrieved books

### 2. Semantic Metrics
Measures semantic similarity between query and retrieved books.

- **Average Similarity**: Mean cosine similarity across all results
- **Max Similarity**: Best match similarity
- **Similarity@K**: Similarity scores at each rank position

### 3. Retrieval Metrics (when ground truth available)
- **Precision@K**: Relevant books in top-K results
- **Recall@K**: Coverage of relevant books in top-K
- **MRR**: Mean Reciprocal Rank of first relevant result

## Test Dataset

The evaluation dataset (`dataset.py`) contains 30+ diverse test queries categorized by type:

### Query Types
- **SPECIFIC**: Clear genre/topic specification (e.g., "SF 소설 추천해줘")
- **EMOTIONAL**: Emotion-based queries (e.g., "요즘 너무 우울해")
- **SITUATIONAL**: Context-based queries (e.g., "군대 가기 전에 읽을 책")
- **VAGUE**: Ambiguous queries (e.g., "재미있는 책 추천")
- **MULTI_INTENT**: Multiple requirements (e.g., "재밌고 의미 있는 소설")

### Example Test Query
```python
TestQuery(
    query_id="S001",
    query="SF 소설 추천해줘",
    query_type=QueryType.SPECIFIC,
    expected_genres=[GenreCategory.NOVEL, GenreCategory.GENRE_NOVEL],
    expected_themes=["SF", "공상과학"],
    notes="명확한 장르 지정"
)
```

## Experiment Configuration

### RetrievalConfig
Configure retrieval parameters:

```python
RetrievalConfig(
    k=5,                    # Number of documents
    use_mmr=True,           # Enable MMR
    mmr_lambda=0.8,         # MMR diversity parameter
    use_reranking=True,     # Enable bestseller reranking
    use_adaptive_k=True,    # Enable adaptive top-k
)
```

### OrchestratorConfig
Configure orchestrator behavior:

```python
OrchestratorConfig(
    enabled=True,
    use_rewriter=True,      # Enable query rewriting
    use_quality_check=True  # Enable quality evaluation
)
```

## Experiment Presets

### Available Presets

1. **baseline**: Default configuration
2. **orchestrator**: With ambiguity-aware orchestrator
3. **ablation**: Progressive feature addition
4. **k_sweep**: Test k ∈ {2, 3, 5, 7, 10}
5. **lambda_sweep**: Test MMR λ ∈ {0.3, 0.5, 0.7, 0.8, 0.9, 0.95}
6. **rerank_sweep**: Test reranking weight combinations
7. **ablation_orchestrator**: Orchestrator component ablation

### Creating Custom Presets

Edit `experiment_config.py`:

```python
@staticmethod
def my_custom_preset() -> ExperimentConfig:
    return ExperimentConfig(
        name="my_experiment",
        description="Custom experiment description",
        retrieval=RetrievalConfig(
            k=7,
            use_mmr=True,
            mmr_lambda=0.6
        ),
        orchestrator=OrchestratorConfig(enabled=False)
    )
```

## Results and Analysis

### Output Files

Results are saved to `experiment_results/` with timestamp:

```
experiment_results/
├── baseline_20231216_123456/
│   ├── config.json                    # Experiment configuration
│   ├── detailed_results_*.json        # Per-query results
│   ├── aggregated_results_*.json      # Aggregated metrics
│   └── summary_report_*.txt           # Human-readable summary
└── comparison_*.json                  # Multi-experiment comparison
```

### Detailed Results Format

```json
{
  "query_id": "S001",
  "query": "SF 소설 추천해줘",
  "query_type": "specific",
  "genre_metrics": {
    "precision": 0.8,
    "recall": 1.0,
    "f1": 0.89,
    "diversity": 0.4
  },
  "semantic_metrics": {
    "avg_similarity": 0.75,
    "max_similarity": 0.92
  },
  "retrieved_books": [...]
}
```

### Aggregated Results Format

```json
{
  "total_queries": 30,
  "genre_metrics": {
    "avg_precision": 0.75,
    "avg_recall": 0.82,
    "avg_f1": 0.78,
    "avg_diversity": 0.45
  },
  "by_query_type": {
    "specific": {...},
    "vague": {...}
  }
}
```

## Visualization

Generate visualizations using `viz.py`:

```bash
python evaluation/viz.py \
    --results-file experiment_results/baseline_*/detailed_results_*.json \
    --output-dir visualizations/
```

### Generated Visualizations
- `metrics_by_query_type.png`: Performance by query type
- `f1_distribution.png`: F1 score distribution
- `genre_distribution.png`: Retrieved genre distribution
- `correlation_matrix.png`: Metric correlations

## Advanced Usage

### Filtering by Query Type

```bash
python evaluation/evaluate.py --preset baseline --query-type vague
```

### Sampling Queries

```bash
python evaluation/evaluate.py --preset baseline --sample 10
```

### Custom Output Directory

```bash
python evaluation/evaluate.py --preset baseline --output-dir my_results/
```

### Running Multiple Experiments

```python
from evaluation.evaluate import ExperimentRunner
from evaluation.experiment_config import ExperimentPresets

runner = ExperimentRunner()
experiments = [
    ExperimentPresets.baseline(),
    ExperimentPresets.orchestrator_enabled()
]
runner.run_experiments(experiments)
```

## Interpreting Results

### Good Performance Indicators
- **F1 > 0.7**: Strong genre matching
- **Diversity > 0.4**: Good genre variety
- **Semantic Similarity > 0.6**: Relevant content retrieval

### Query Type Performance
- **SPECIFIC**: Should have high precision (>0.8)
- **VAGUE**: May have lower precision but higher diversity
- **EMOTIONAL**: Semantic similarity is more important than genre precision

### Comparison Guidelines
- Compare F1 scores across experiments
- Check diversity for vague/multi-intent queries
- Analyze by_query_type breakdown for insights

## Troubleshooting

### Vector Store Not Found
```bash
# Build vector store first
python main.py --rebuild
```

### Memory Issues
Reduce sample size or test fewer query types:
```bash
python evaluation/evaluate.py --preset baseline --sample 5 --query-type specific
```

### Slow Execution
Use lighter models in `src/config.py`:
```python
CHAIN_MODEL_NAME = "google_genai:gemini-2.0-flash-lite"
```

## Contributing

### Adding New Metrics

1. Create evaluator class in `metrics.py`:
```python
class MyCustomEvaluator:
    def evaluate(self, test_query, retrieved_books):
        # Implementation
        return MyCustomMetrics(...)
```

2. Integrate in `EvaluationRunner.evaluate_single_query()`

### Adding New Test Queries

Edit `dataset.py`:
```python
TestQuery(
    query_id="NEW001",
    query="Your test query",
    query_type=QueryType.SPECIFIC,
    expected_genres=[GenreCategory.NOVEL],
    notes="Description"
)
```

## References

- Main README: `../README.md`
- Configuration: `../src/config.py`
- Experiment Details: `EVALUATION_AND_EXPERIMENTS.md`
