# Self-Compassion LLM for Neurodiverse Brains 🧠

## 🚀 Quickstart

### Clone the repo and install dependencies:

```bash
git clone https://github.com/<your-username>/selfcompassion-llm.git
cd selfcompassion-llm
pip install -r requirements.txt
```

### Run ingestion to build the index:

```bash
python ingestion.py
```

### Launch the Streamlit app:

```bash
streamlit run streamlit_app.py
```

---

## 📂 Project Structure

```
selfcompassion-llm/
│
├── streamlit_app.py        # main Streamlit app
├── ingestion.py            # build FAISS/embedding index
├── monitor.py              # simple monitoring dashboard
├── docker-compose.yml      # optional Docker setup
├── Dockerfile
├── processed_documents_clean.json  # demo dataset
├── requirements.txt
└── notebooks/              # (optional) exploration notebooks
```

---

## 🧠 Features

- Retrieval-Augmented Generation (RAG) pipeline  
- Query rewriting + document re-ranking  
- Simple feedback collection + monitoring dashboard  
- Streamlit UI (press Enter or click **Answer**)  

---

## 📊 Monitoring

A separate monitoring page (`monitor.py`) tracks:  

- Sources per answer  
- Feedback breakdown  
- Query rewriting usage  

**Run it with:**

```bash
streamlit run monitor.py
```

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

---

## 📘 Development Notes

- Exploration notebooks are under `/notebooks/`.  
- Embeddings (`embeddings.npy`, `id_index.json`) are **ignored** from git.  
- Feedback data is stored locally (not shared).  

---

## 📜 License

MIT License
