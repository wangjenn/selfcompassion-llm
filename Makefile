dev:
\tstreamlit run streamlit_app.py --server.port=8503

up:
\tdocker compose up --build

down:
\tdocker compose down

golden:
\tpython scripts/generate_golden.py

eval:
\tpython -c "from streamlit_app import bm25_search, vector_search, hybrid_search; \
from evaluate_retrieval import evaluate_system; \
import json; \
print('BM25',  evaluate_system(bm25_search)); \
print('Vector',evaluate_system(vector_search)); \
print('Hybrid',evaluate_system(hybrid_search)))"
