# LLM Annotation Validation Framework

This repository implements a validation framework for Large Language Model (LLM) generated annotations when reliable benchmark data is unavailable. The framework tests whether an LLM can reconstruct passages from annotated labels while maintaining semantic consistency with the original text.

## Overview

The validation framework addresses a key challenge: **How can researchers validate LLM measurements when reliable external benchmarks (like human annotations) are unavailable?**

The framework establishes validity through three key tests:

1. **Annotation Backtranslation Test**: Verifies that the annotation function `f^{-1}` and simulation function `f` are mutually consistent (i.e., `f^{-1}(f(β)) = β`)

2. **Separation Test**: Ensures that texts generated from different labels are semantically distinct

3. **Validity Test**: Checks that text generated from the annotated label is semantically similar to the original text

The Annotation Backtranslation Test and Separation Test assess whether the problem satisfy pre-requisite requirements for the Validity Test to work. Together, they mitigate the risk of validating an invalid annotation due to circular reasoning. 


## Installation

### Prerequisites

- Python 3.8 or higher
- An OpenAI API key (or compatible API endpoint)

### Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Configure API credentials in .env file


## Quick Start

Run the example sentiment analysis validation:

```bash
python example_sentiment_news.py
```

This demonstrates the complete validation framework on a sentence from a financial news article.

## Framework Components

### Core Utilities (`utils/`)

- **`test_backtranslation.py`**: Tests the annotation backtranslation property
- **`test_separation.py`**: Tests the separation property  
- **`test_validity.py`**: Tests annotation validity using semantic similarity
- **`llm_api_calls.py`**: Handles LLM API calls for completions and embeddings with automatic logging
- **`cosine_similarity.py`**: Computes cosine similarity between embedding vectors
- **`count_tokens.py`**: Token counting utility
- **`sampling.py`**: Random and stratified sampling utilities

### API Call Logging

All LLM API calls (completions and embeddings) are automatically logged to `api_logs/api_calls_YYYY-MM-DD.jsonl` for debugging and cost tracking. Each log entry includes:
- Timestamp
- Model used
- Call type (completion/embedding)
- Input preview
- Token usage (when available)
- Success/error status

### Example: Sentiment Analysis on News

The included example (`example_sentiment_news.py`) demonstrates validation of sentiment analysis on financial news:

```python
from utils.test_backtranslation import test_backtranslation
from utils.test_separation import test_separation
from utils.test_validity import test_validity

# Define annotation function f^{-1}(text) → label
def annotation_func(text):
    # Your LLM prompt to classify text
    ...

# Define simulation function f(label) → text
def simulation_func(label, num_texts):
    # Your LLM prompt to generate texts with given label
    ...

# Run the three validation tests
backtranslation_results = test_backtranslation(...)
separation_results = test_separation(...)
validity_results = test_validity(...)
```

## How It Works

### 1. Annotation Backtranslation Property

This test generates texts from each label using the simulation function `f`, then annotates them using the annotation function `f^{-1}`, checking if the result matches the original label.

**Example**: If `f(positive)` generates a positive text, then `f^{-1}` should label it as positive.

**Pass criterion**: High accuracy (typically ≥90%) in recovering the original labels.

### 2. Separation Property

This test verifies that texts generated from different labels are semantically distinct using cosine similarity of embeddings.

**Example**: Texts generated with label "positive" should be semantically different from texts generated with label "negative".

**Pass criterion**: Statistical test rejects the null hypothesis that texts from different labels are similar.

### 3. Validity Test

This test checks if text generated from the annotated label is semantically similar to the original text.

**Example**: If the LLM annotates a text as "positive", then text generated with the "positive" label should be semantically similar to the original.

**Pass criterion**: Statistical test fails to reject the null hypothesis that simulated and original texts are similar.

## Customizing for Your Task

To adapt the framework for your own annotation task:

1. **Define your labels**: What categories/values are you measuring?
   ```python
   LABELS = ['category1', 'category2', 'category3']
   ```

2. **Create your annotation function** (`f^{-1}`): How does your LLM classify texts?
   ```python
   def annotation_func(text):
       # Your prompt engineering here
       return predicted_label
   ```

3. **Create your simulation function** (`f`): How does your LLM generate texts with specific characteristics?
   ```python
   def simulation_func(label, num_texts):
       # Your prompt engineering here
       return list_of_generated_texts
   ```

4. **Run the validation tests** on your actual data

## Use Cases

The framework is applicable to various text annotation tasks:

- **Sentiment analysis** (positive/negative/neutral)
- **Topic classification** (does text mention specific topics?)
- **Clarity measurement** (clear/unclear/vague)
- **Temporal focus** (past/present/future)
- **Stance detection** (support/oppose/neutral)
- And more...

## Citation

If you use this framework in your research, please cite:

```
@misc{hansen2026validating,
  title={{Validating Large Language Model Annotations}},
  author={Hansen, Anne Lundgaard},
  year={2026},
  howpublished={FEDS Working Paper No. 2026-20}
}
```

## License

MIT License - See LICENSE file for details

## Questions?

For questions or issues, please open an issue on GitHub or contact the author.
