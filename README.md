# Self-Compassion LLM for Neurodiverse Brains ❥ 🧠

<p align="center">
<img width="254" height="276" alt="labubu-selfcompassion" src="https://github.com/user-attachments/assets/fc95f795-bd29-435e-81c3-854555fd22d3" />
</p>  

## ✨ INTRO
- Evidence-based/scientific

---
## 💕 LIVE DEMO
- Try out the **deployed app** (no setup required!) 👉🏻 [![Live Demo](https://img.shields.io/badge/demo-online-brightgreen?style=for-the-badge)](https://selfcompassion.streamlit.app/)

---
## 🛠️ Environment & Configuration

- **Python version:** Python 3.11+ recommended
- **Environment variables:**  
  - `OPENAI_API_KEY` (required for LLM features; set in a `.env` file or your environment)

---
## 🚀 Quickstart (clone and run locally)
- **Clone the repo and install dependencies:**
  ```bash
  git clone https://github.com/<your-username>/selfcompassion-llm.git
  cd selfcompassion-llm
  pip install -r requirements.txt
  ```

- **Run ingestion to build the index:**
  ```bash
  python ingestion.py
  ```

- **Launch the Streamlit app:**
  ```bash
  streamlit run streamlit_app.py
  ```

## 🛥 Alternative: Docker Setup
- Set up API key. Create a `.env` file in project root: 
  ```bash
  OPENAI_API_KEY=your_actual_api_key_here
  ```
- Run **docker**: 
  ```bash
  docker-compose up -- build
  ```
- Access at http://localhost:8503

---
## 📂 Project Structure

```
selfcompassion-llm/
├── streamlit_app.py        # main Streamlit app
├── ingestion.py            # build FAISS/embedding index
├── monitor.py              # simple monitoring dashboard
├── docker-compose.yml      # Docker setup
├── Dockerfile
├── Makefile                # Developer shortcuts 
├── processed_documents_clean.json  # demo dataset
├── requirements.txt
└── notebooks/              # exploration notebook with example code
```

---
## 💡 Features
- Retrieval-Augmented Generation (RAG) pipeline  
- Query rewriting + document re-ranking  
- Simple feedback collection + monitoring dashboard  
- Streamlit UI (press Enter or click **Answer**)  

---
## 🏠 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                               │
│                     Streamlit Web App                               │
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │
│  │   Query     │  │   Settings  │  │  Feedback   │                  │
│  │   Input     │  │   Panel     │  │   Radio     │                  │
│  └─────────────┘  └─────────────┘  └─────────────┘                  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      QUERY PROCESSING                               │
│                                                                     │
│  ┌─────────────┐         ┌─────────────┐                            │
│  │   Query     │────────▶│  Original   │                            │
│  │ Rewriting   │         │   Query     │                            │ 
│  └─────────────┘         └─────────────┘                            │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    RETRIEVAL LAYER                                  │
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │
│  │    BM25     │  │   Vector    │  │   Hybrid    │                  │
│  │   Search    │  │   Search    │  │   Search    │                  │
│  └─────────────┘  └─────────────┘  └─────────────┘                  │
│                               │                                     │
│                               ▼                                     │
│                   ┌─────────────────┐                               │
│                   │   Document      │                               │
│                   │  Re-ranking     │                               │
│                   └─────────────────┘                               │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   KNOWLEDGE BASE                                    │
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │
│  │ Documents   │  │ Embeddings  │  │ BM25 Index  │                  │
│  │    JSON     │  │    .npy     │  │   Memory    │                  │
│  └─────────────┘  └─────────────┘  └─────────────┘                  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LLM GENERATION                                   │
│                                                                     │
│  ┌─────────────┐         ┌─────────────┐                            │
│  │   OpenAI    │◀────────│   Prompt    │                            │
│  │  gpt-4o-mini│         │ Engineering │                            │
│  └─────────────┘         └─────────────┘                            │
│                               │                                     │
│  ┌─────────────┐              │                                     │
│  │   3 Prompt  │◀─────────────┘                                     │
│  │   Styles    │                                                    │
│  └─────────────┘                                                    │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   MONITORING & LOGGING                              │
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │
│  │  Event Log  │  │ Feedback    │  │ Dashboard   │                  │
│  │   JSONL     │  │ Collection  │  │  Charts     │                  │
│  └─────────────┘  └─────────────┘  └─────────────┘                  │
└─────────────────────────────────────────────────────────────────────┘
```

## 🔄 Data Flow

1. **User Input** → Query entered in Streamlit interface
2. **Query Processing** → Optional rewriting with domain terms  
3. **Retrieval** → Multiple search methods (BM25/Vector/Hybrid)
4. **Re-ranking** → Document relevance scoring
5. **Context Building** → Top documents formatted for LLM
6. **Generation** → OpenAI API with custom prompts
7. **Response** → Answer + sources displayed to user
8. **Monitoring** → Events logged for analytics

---
## 📊 Monitoring

- **A separate monitoring page (`monitor.py`) tracks:**
  - Sources per answer  
  - Feedback breakdown  
  - Query rewriting usage  

- **Run it with:**
  ```bash
  streamlit run monitor.py
  ```

---
## ✅ Evaluation Results
### Running Evaluation
  ```bash
  python evaluation.py
  ```
- App is deployed and running [here](https://selfcompassion.streamlit.app/). 
- The **Self-Compassion RAG system** was evaluated on a golden dataset of 20 queries with labeled relevant documents. It tests the RAG system against the golden dataset.

---
### 💪🏻 Performance Metrics

| Retrieval Method | Hit Rate | MRR   | Precision@10 | Recall@10 |
|------------------|----------|-------|--------------|-----------|
| **BM25**         | 1.000    | 0.598 | 0.100        | 1.000     |
| **Vector**       | 0.900    | 0.410 | 0.090        | 0.900     |
| **TF-IDF**       | 0.900    | 0.410 | 0.090        | 0.900     |
| **Hybrid**       | 0.950    | 0.520 | 0.095        | 0.950     |

---
### 💡 Key Findings
- **BM25** consistently achieves perfect hit rate and recall in this dataset, with the best MRR (0.598), ranking relevant documents higher.
- **Hybrid** slightly improves over pure TF-IDF and Vector, balancing precision and recall.
- **Vector** and **TF-IDF** show similar performance, with lower MRR than BM25 but decent recall.
- **BM25** emerges as the most reliable retriever under current conditions, though **Hybrid** provides a good balance. 

---
## ⚡ Data & Reproducibility

- Deterministic ingestion pipeline (`ingestion.py`)  
- `requirements.txt` provided for environment setup  
- Docker + docker-compose included for containerized runs
- Repo includes a small demo dataset (`processed_documents_clean.json`) to immediately run the app and test its functionality.
- To rebuild the larger embedding index files locally, run:
  
  ```bash
  make ingest
  ```
  or 
  ```bash
  python ingestion.py
  ```

---
## 📘 Development Notes

- Exploration notebooks are under `/notebooks/`.  
- Embeddings (`embeddings.npy`, `id_index.json`) are **ignored** from git.  
- Feedback data is stored locally (not shared).  

---
## 📚 Project Criteria
- See [Project Criteria](ProjectEvaluationCriteria.md)
---

#### ❥ Thank you so much for using the app-- I hope you find it useful! May we all be kind to ourselves! 💕🙏🏻
![selfcompassion](https://github.com/user-attachments/assets/6009a37a-be39-48d8-b736-ab19b9b6bcd7)
---
