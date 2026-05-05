"""
Constrói o índice semântico (embeddings) sobre o corpus BM25 existente.
Executa uma única vez; resultado salvo em data/corpus/bm25_index_semantic.npy.

Pré-requisito: BM25 index já construído (python -m src.corpus.index).

Uso:
  python -m src.corpus.build_semantic_index
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..')))

from dotenv import load_dotenv
load_dotenv()

from src.corpus.index import build_index

corpus_dir = Path(os.getenv("CORPUS_DIR", "data/corpus"))
sem_path   = corpus_dir / "bm25_index_semantic.npy"

if sem_path.exists():
    print(f"Índice semântico já existe: {sem_path}")
    print("Delete o arquivo e rode novamente para reconstruir.")
    sys.exit(0)

print("Carregando índice BM25...")
idx = build_index(corpus_dir)

print(f"\nConstruindo embeddings para {len(idx.papers)} papers...")
print("(all-MiniLM-L6-v2, ~22ms por paper, pode demorar alguns minutos)\n")

idx.build_semantic_index(save_path=sem_path)

print(f"\n✅ Índice semântico pronto: {sem_path}")
print("   A busca híbrida será ativada automaticamente na próxima execução.")
