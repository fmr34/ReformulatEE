import json
import numpy as np

data = json.loads(open("data/results/fase0_v2_sem_api.json", encoding="utf-8").read())
pq = data["per_question"]
pos = [r for r in pq if r["label"] == 1]
neg = [r for r in pq if r["label"] == 0]

print(f"AUC: {data['auc']}  |  Gate: {'PASSOU' if data['gate_passed'] else 'FALHOU'}")
print()
print("=== Estatisticas por label ===")
ee1 = [r["ee"] for r in pos]
ee0 = [r["ee"] for r in neg]
resp1 = [r["respondibilidade"] for r in pos]
resp0 = [r["respondibilidade"] for r in neg]
print(
    f"label=1 (alta EE)  | EE={np.mean(ee1):.3f} +/- {np.std(ee1):.3f} | resp={np.mean(resp1):.3f}"
)
print(
    f"label=0 (baixa EE) | EE={np.mean(ee0):.3f} +/- {np.std(ee0):.3f} | resp={np.mean(resp0):.3f}"
)

print()
print("=== Top 5 falsos positivos (label=0 com EE alta) ===")
fp = sorted(neg, key=lambda x: x["ee"], reverse=True)[:5]
for r in fp:
    print(f"  resp={r['respondibilidade']:.3f} ee={r['ee']:.3f} | {r['text'][:65]}")

print()
print("=== Top 5 falsos negativos (label=1 com EE baixa) ===")
fn = sorted(pos, key=lambda x: x["ee"])[:5]
for r in fn:
    print(f"  resp={r['respondibilidade']:.3f} ee={r['ee']:.3f} | {r['text'][:65]}")
