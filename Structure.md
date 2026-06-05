# Project Directory Structure (Structure.md)

All Python code generated for Project Aletheia must strictly adhere to this directory structure to ensure absolute import paths work correctly.

` ` `text
MahoRAGa/
├── core/
│   ├── __init__.py
│   ├── config.py           # Pydantic BaseSettings (API keys, thresholds)
│   ├── state.py            # The Pydantic models from State.md
│   └── exceptions.py       # Custom error classes
├── graph/
│   ├── __init__.py
│   ├── orchestrator.py     # The compiled LangGraph StateGraph
│   ├── nodes/              # Individual LangGraph node functions
│   │   ├── __init__.py
│   │   ├── router.py
│   │   ├── retriever.py
│   │   ├── evaluator.py
│   │   ├── search_apis.py
│   │   ├── pruner.py
│   │   └── generator.py
├── memory/
│   ├── __init__.py
│   ├── neo4j_client.py     # Graph database connection
│   ├── vector_store.py     # FAISS implementation
│   └── sparse_store.py     # Disk-backed BM25 implementation
├── ui/
│   └── app.py              # Gradio interface
├── .env                    # Environment variables
└── requirements.txt
` ` `
