# Notebooks Directory

This directory contains Jupyter notebooks for exploring and understanding the Self-Compassion RAG system implementation.

## `self_compassion_rag_implementation.ipynb`

A comprehensive notebook demonstrating the complete RAG (Retrieval-Augmented Generation) implementation for the self-compassion system.

### What's Included

1. **Configuration & Setup**
   - Environment setup and API key configuration
   - File path definitions and constants

2. **Index Building**
   - BM25 keyword-based indexing
   - Vector embeddings using OpenAI's text-embedding-3-small
   - Caching mechanisms for efficiency

3. **Search Implementation**
   - **BM25 Search**: Traditional keyword-based retrieval
   - **Vector Search**: Semantic similarity using embeddings
   - **Hybrid Search**: Combination of BM25 and vector scores
   - Search router for easy method switching

4. **Evaluation**
   - Performance metrics (Hit Rate, MRR, Precision@K, Recall@K)
   - Comparison across different search methods
   - Results visualization

5. **QA with Grounding**
   - LLM integration for answer generation
   - Source attribution and citation
   - Example queries and responses

### Key Features

- **Multiple Retrieval Methods**: Compare BM25, vector, and hybrid approaches
- **Caching**: Efficient embedding storage and retrieval
- **Evaluation**: Comprehensive performance assessment
- **Grounding**: Source-attributed answers with citations
- **Modular Design**: Easy to extend and modify

### Usage

1. **Prerequisites**:
   ```bash
   pip install -r ../requirements.txt
   ```

2. **Environment Setup**:
   - Create a `.env` file in the project root
   - Add your OpenAI API key: `OPENAI_API_KEY=your_key_here`

3. **Data Requirements**:
   - Ensure `../processed_documents_clean.json` exists (run `python ../ingestion.py` first)

4. **Run the Notebook**:
   ```bash
   jupyter notebook self_compassion_rag_implementation.ipynb
   ```

### Expected Outputs

- **Search Results**: Comparison of different retrieval methods
- **Performance Metrics**: Quantitative evaluation results
- **QA Examples**: Sample questions and grounded answers
- **Cached Files**: Embeddings and indexes for reuse

### Integration with Main App

The functions and approaches demonstrated in this notebook are integrated into:
- `../streamlit_app.py` - Main user interface
- `../evaluation.py` - Automated evaluation pipeline
- `../ingestion.py` - Data processing and indexing

### Notes

- The notebook is designed for educational and development purposes
- All sensitive information (API keys, personal data) has been removed
- Results may vary based on the specific dataset and evaluation queries
- The notebook includes extensive comments and explanations for learning

### Troubleshooting

- **Missing API Key**: Ensure your `.env` file contains a valid OpenAI API key
- **Missing Data**: Run `python ../ingestion.py` to generate the required data files
- **Import Errors**: Install all requirements with `pip install -r ../requirements.txt`
- **Memory Issues**: The notebook loads the full document corpus into memory

---

For more information about the project, see the main [README.md](../README.md).
