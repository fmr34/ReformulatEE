import sys
from importlib.metadata import version, PackageNotFoundError

print("Python:", sys.version)

def pkg_version(import_name, pkg_name=None):
    pkg_name = pkg_name or import_name
    def _check():
        __import__(import_name)
        try:
            return version(pkg_name)
        except PackageNotFoundError:
            return getattr(sys.modules[import_name], "__version__", "installed")
    return _check

checks = {
    "torch":                 pkg_version("torch"),
    "transformers":          pkg_version("transformers"),
    "sentence_transformers": pkg_version("sentence_transformers"),
    "rank_bm25":             pkg_version("rank_bm25"),
    "faiss":                 pkg_version("faiss", "faiss-cpu"),
    "sklearn":               pkg_version("sklearn", "scikit-learn"),
    "anthropic":             pkg_version("anthropic"),
    "arxiv":                 pkg_version("arxiv"),
}

ok = True
for name, fn in checks.items():
    try:
        v = fn()
        print(f"  [OK] {name}: {v}")
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        ok = False

import torch
backend = "CUDA" if torch.cuda.is_available() else "CPU"
print(f"\nPyTorch backend: {backend}")

sys.exit(0 if ok else 1)
