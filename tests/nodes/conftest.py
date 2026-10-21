"""
Shared stubs for node-level unit tests.

Importing ``graph.nodes.*`` pulls the package __init__ chain
(graph/__init__ -> graph/nodes/__init__ -> retriever -> memory.*), which in
turn imports heavy third-party modules (neo4j, faiss, sentence_transformers,
torch, numpy). Those are not installed in the lightweight validation venv, so
we insert minimal stand-ins into ``sys.modules`` BEFORE any ``from graph...``
or ``from memory...`` import is resolved.

When the real heavy deps ARE installed (full environment), these stubs are
only inserted for modules that are genuinely absent, so they do not shadow the
real implementations.
"""

import sys
import types


def _stub(name, **attrs):
    if name in sys.modules:
        return sys.modules[name]
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


# neo4j driver + the specific exceptions imported by memory/neo4j_client.py
_stub(
    "neo4j",
    GraphDatabase=type("GraphDatabase", (), {}),
    Session=type("Session", (), {}),
)
_stub(
    "neo4j.exceptions",
    AuthError=type("AuthError", (Exception,), {}),
    ServiceUnavailable=type("ServiceUnavailable", (Exception,), {}),
    SessionExpired=type("SessionExpired", (Exception,), {}),
)

# FAISS / torch are only needed for import resolution in these unit tests.
_stub("faiss")
_stub("torch")

# sentence_transformers: provide the classes referenced at import time.
_stub(
    "sentence_transformers",
    SentenceTransformer=type("SentenceTransformer", (), {}),
    CrossEncoder=type("CrossEncoder", (), {}),
)

# numpy: a tiny stub is enough for import resolution. If real numpy is already
# installed it is reused via the early-return in _stub.
_stub("numpy")
