"""
Smoke test rapido — verifica imports e componentes isolados sem API e sem corpus.
Nao faz chamadas externas.
"""

import sys
import numpy as np


def check(label: str, fn):
    try:
        fn()
        print(f"  [OK] {label}")
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        return False
    return True


print("\n=== Smoke Test — ReformulatEE Fase 0 ===\n")

ok = True

ok &= check("rank_bm25 import", lambda: __import__("rank_bm25"))
ok &= check("sentence_transformers import", lambda: __import__("sentence_transformers"))
ok &= check("faiss import", lambda: __import__("faiss"))
ok &= check("sklearn import", lambda: __import__("sklearn"))
ok &= check("anthropic import", lambda: __import__("anthropic"))
ok &= check("arxiv import", lambda: __import__("arxiv"))

ok &= check("src.corpus.index import", lambda: __import__("src.corpus.index", fromlist=["CorpusIndex"]))
ok &= check("src.ee.nao_trivialidade import", lambda: __import__("src.ee.nao_trivialidade", fromlist=["nao_trivialidade"]))
ok &= check("src.eval.curated import + validacao", lambda: __import__("src.eval.curated", fromlist=["get_curated"]).get_curated())

# Testa funcao sino isolada (sem modelo)
from src.ee.nao_trivialidade import bell
ok &= check("bell(0.45) ~ 1.0", lambda: None if abs(bell(0.45) - 1.0) < 0.01 else (_ for _ in ()).throw(AssertionError(f"bell(0.45)={bell(0.45)}")))
ok &= check("bell(0.0) < 0.1", lambda: None if bell(0.0) < 0.1 else (_ for _ in ()).throw(AssertionError(f"bell(0.0)={bell(0.0)}")))
ok &= check("bell(1.0) < 0.1", lambda: None if bell(1.0) < 0.1 else (_ for _ in ()).throw(AssertionError(f"bell(1.0)={bell(1.0)}")))

# Testa BM25 com corpus minimo
from rank_bm25 import BM25Okapi
def _test_bm25():
    corpus = [["hello", "world"], ["science", "research"], ["question", "research", "method"]]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(["research"])
    assert scores[1] > scores[0], "BM25 nao priorizou documento relevante"
ok &= check("BM25 score ordering", _test_bm25)

print(f"\n{'='*40}")
print(f"  Resultado: {'PASSOU' if ok else 'FALHOU'}")
print(f"{'='*40}\n")
sys.exit(0 if ok else 1)
