# Self-Compassion LLM for Neurodiverse Brains ❥ 🧠

<p align="center">
<img width="254" height="276" alt="labubu-selfcompassion" src="https://github.com/user-attachments/assets/fc95f795-bd29-435e-81c3-854555fd22d3" />
</p>  

## ✨ Introduction
- Research shows that individuals with neurodiverse conditions (e.g., ADHD, autism) often struggle with self-compassion ([Beaton et al., 2002](https://pubmed.ncbi.nlm.nih.gov/35334113/)). Everyday reminders to be kind to oneself can be cognitively demanding and difficult to sustain, especially during stressful situations or when executive function is taxed. This project addresses that gap by offering a lightweight and accessible tool to make self-compassion easier and less effortful. Drawing from leading evidence-based research, this app provides a structured and easy way to practice kindness toward oneself ❥.
<br><br>

---
## 💕 Live Demo
- Try out the **deployed app** (no setup required!) 👉🏻 [![Live Demo](https://img.shields.io/badge/demo-online-brightgreen?style=for-the-badge)](https://selfcompassion.streamlit.app/)

[videowalkthrough.webm](https://github.com/user-attachments/assets/d5fb192b-9fbe-46a0-bf0b-b9fd50163793)


---
## 💡 Features
- **Science-based**: built using leading empirical research and science on neurodiversity and self-compassion.

- **Interactive Web Interface**: easy-to-use Streamlit app with fast, instant responses (press 'Enter' or click 'Answer').
  <p align="center">
  <img width="643" height="139" alt="ScreenFloat Shot of Google Chrome on 2025-08-21 at 14-32-31" src="https://github.com/user-attachments/assets/20e8c75a-6c77-4b04-82ad-ed7073e083d2" />
</p>

- **Smart Search Options**: choose between `hybrid`, `BM25`, or `vector` retrieval to find the most relevant guidance.
  <p align="center">
    <img width="206" height="164" alt="ScreenFloat Shot of Streamlit on 2025-08-21 at 14-47-55" src="https://github.com/user-attachments/assets/53c2e092-2802-45c1-83bc-d765613a0c90" />
  </p>

- **Personalized Experience**: select from 3 prompt styles based on your moods and needs-- 💕 Supportive, 📚 Direct, or 💪🏻 Action-Oriented!
  <p align="center">
    <img width="211" height="187" alt="ScreenFloat Shot of Streamlit on 2025-08-21 at 14-47-04" src="https://github.com/user-attachments/assets/8c315c1f-114e-4e9e-aaf8-7141eb2f7c50" />
</p>

- **Advanced Customization**: adjust search depth (top-k results), LLM creativity (temperature), and choose between GPT models (gpt-4o-mini or gpt-3.5-turbo).
<p align="center">
  <img width="217" height="169" alt="ScreenFloat Shot of Streamlit on 2025-08-21 at 14-55-10" src="https://github.com/user-attachments/assets/ebd9d3ca-6707-4be5-96d3-85ffb87c8273" />
</p>

- **Enhanced Retrieval**: optional query expansion and document re-ranking for better, more relevant results.
<p align="center">
  <img width="190" height="94" alt="ScreenFloat Shot of Streamlit on 2025-08-21 at 14-57-17" src="https://github.com/user-attachments/assets/485ec704-6982-445a-acad-7985af7f2aa1" />
</p>

- **Built-in Analytics**: Real-time feedback collection and monitoring dashboards to track usage and effectiveness.
<p align="center">
<img width="574" height="742" alt="ScreenFloat Shot of Streamlit on 2025-08-21 at 15-00-21" src="https://github.com/user-attachments/assets/a9769c97-9cad-4242-8e14-99ec75c3c037" />
</p>

<br><br>

---
## 🛠️ Environment & Configuration

- **Python version:** Python 3.11+ recommended
- **Environment variables:**  
  - `OPENAI_API_KEY` (required for LLM features; set in a `.env` file or your environment). 
<br><br>
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
<br><br>
---

## 📂 Project Structure

```
selfcompassion-llm/
├── streamlit_app.py        # main Streamlit app
├── ingestion.py            # build FAISS/embedding index
├── monitor.py              # simple monitoring dashboard
├── docker-compose.yml      # docker setup
├── Dockerfile
├── Makefile                # developer shortcuts 
├── processed_documents_clean.json  # demo dataset
├── requirements.txt
├── images                  # images folder
└── notebooks/              # exploration notebook with example code
```
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
<br><br>
---
## 📊 Monitoring
- **Run it with:**
  ```bash
  streamlit run monitor.py
  ```
- **A separate monitoring page (`monitor.py`) tracks:**
  - Sources per answer  
  - Feedback breakdown  
  - Query rewriting usage
    
<p align="center">
  <img width="574" height="734" alt="ScreenFloat Shot of Streamlit on 2025-08-21 at 15-05-29" src="https://github.com/user-attachments/assets/fd289e6b-2ff7-4a8b-bf81-9af05e5d28f9" />
</p>

---
## ✅ Evaluation Results
### 🚘 Running Evaluation
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
## 📝 Disclaimer
- All responses are anonymous — nothing is saved, logged, or tracked by the app.
- This app does not store, share, or distribute full PDFs or copyrighted materials. All content is based on short excerpts, summaries, or personal notes from books I’ve legally purchased, or on publicly available sources such as peer-reviewed research papers and podcast transcripts.
- This app is intended for educational and personal use only. It is not intended to diagnose, treat, cure, or prevent any mental health condition. If you are in emotional distress or experiencing a crisis, please contact a licensed mental health professional.
---

### ❥ *Thank you so much for using the app-- I hope you find it useful! May we all be kind to ourselves!* 💕🙏🏻
![selfcompassion](https://github.com/user-attachments/assets/6009a37a-be39-48d8-b736-ab19b9b6bcd7)

- This project is licensed under the [MIT License](./license)

