"""
Layer 3 — Classificador humano dos pares divergentes.

Uso: .venv\Scripts\python classify_divergent.py

Teclas:
  S  — Sim, concordante (q_good é genuinamente melhor que q_bad)
  N  — Não, rejeitar   (q_good NÃO é genuinamente melhor)
  ?  — Incerto / deixar para depois
  Q  — Salvar e sair   (retoma de onde parou na próxima execução)

Critério de avaliação:
  Uma reformulação é CONCORDANTE quando:
    1. A nova pergunta tem metodologia mais clara — existem ferramentas,
       experimentos ou formalismos conhecidos para avançar nela.
    2. Responder a nova pergunta faz progresso real em direção à original.
    3. Não é apenas uma paráfrase ou troca de vocabulário.
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(override=True)

DIVERGENTES  = Path("data/pairs/pairs_layer2_divergent.jsonl")
CONCORDANTES = Path("data/pairs/pairs_layer2.jsonl")
LOG_HUMANO   = Path("data/pairs/layer3_decisoes_humanas.jsonl")

# Cores ANSI (desabilitadas automaticamente fora de terminal)
def _cor(texto: str, codigo: str) -> str:
    if not sys.stdout.isatty():
        return texto
    return f"{codigo}{texto}\033[0m"

AMARELO = "\033[93m"
VERDE   = "\033[92m"
VERMELHO = "\033[91m"
CIANO   = "\033[96m"
NEGRITO = "\033[1m"
FRACO   = "\033[2m"


def carregar_decididos() -> set[str]:
    if not LOG_HUMANO.exists():
        return set()
    decididos = set()
    for linha in LOG_HUMANO.read_text(encoding="utf-8").splitlines():
        if linha.strip():
            rec = json.loads(linha)
            if rec.get("decisao") in ("concordante", "rejeitado"):
                decididos.add(rec["source_id"])
    return decididos


def salvar_decisao(par: dict, decisao: str) -> None:
    LOG_HUMANO.parent.mkdir(parents=True, exist_ok=True)
    registro = {
        "source_id": par.get("source_id", ""),
        "q_bad":     par["q_bad"],
        "q_good":    par["q_good"],
        "dominio":   par.get("domain", ""),
        "fonte":     par.get("source", ""),
        "decisao":   decisao,  # "concordante" | "rejeitado" | "incerto"
        "n_anotadores": par.get("n_annotators", 0),
        "resumo_anotadores": [
            {
                "modelo":    a["model"].split("-")[0],
                "direcao":   a["direction"],
                "magnitude": a["magnitude"],
            }
            for a in par.get("annotations", [])
            if not a.get("error")
        ],
    }
    with LOG_HUMANO.open("a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")


def aplicar_decisoes() -> tuple[int, int]:
    """Copia pares 'concordante' para pairs_layer2.jsonl."""
    if not LOG_HUMANO.exists():
        return 0, 0

    decisoes = {}
    for linha in LOG_HUMANO.read_text(encoding="utf-8").splitlines():
        if linha.strip():
            rec = json.loads(linha)
            decisoes[rec["source_id"]] = rec["decisao"]

    divergentes = [
        json.loads(l)
        for l in DIVERGENTES.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    adicionados = rejeitados = 0
    with CONCORDANTES.open("a", encoding="utf-8") as f:
        for par in divergentes:
            sid = par.get("source_id", "")
            decisao = decisoes.get(sid)
            if decisao == "concordante":
                par["validado_humano"] = True
                par["passes_layer2"] = True
                f.write(json.dumps(par, ensure_ascii=False) + "\n")
                adicionados += 1
            elif decisao == "rejeitado":
                rejeitados += 1

    return adicionados, rejeitados


def formatar_anotadores(par: dict) -> str:
    anots = [a for a in par.get("annotations", []) if not a.get("error")]
    if not anots:
        return _cor("  Nenhum anotador automático funcionou (falha de API)", FRACO)
    linhas = []
    for a in anots:
        modelo = a["model"].split("-")[0].capitalize()
        direcao = "SIM" if a["direction"] else "NÃO"
        cor = VERDE if a["direction"] else VERMELHO
        linhas.append(f"  {modelo}: {_cor(direcao, cor)}  (magnitude {a['magnitude']:.2f})")
    return "\n".join(linhas)


def limpar_tela() -> None:
    os.system("cls" if os.name == "nt" else "clear")


# Cache de traduções para não repetir chamadas se relançar o script
_cache_traducao: dict[str, dict] = {}


def traduzir_par(par: dict) -> dict:
    """Retorna contexto, q_bad e q_good traduzidos para pt-br via Claude."""
    sid = par.get("source_id", "")
    if sid in _cache_traducao:
        return _cache_traducao[sid]

    import anthropic
    client = anthropic.Anthropic()

    textos = {
        "contexto": (par.get("context") or "").strip()[:400],
        "q_bad":    par["q_bad"],
        "q_good":   par["q_good"],
    }

    prompt = (
        "Traduza os três campos abaixo para o português brasileiro. "
        "Preserve termos técnicos científicos entre parênteses em inglês quando necessário. "
        "Responda SOMENTE com JSON válido, sem markdown.\n\n"
        f'{json.dumps(textos, ensure_ascii=False)}'
    )

    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        import re
        raw = re.sub(r"```[a-z]*\n?", "", msg.content[0].text).strip()
        resultado = json.loads(raw)
    except Exception:
        # Fallback: retorna original sem tradução
        resultado = textos

    _cache_traducao[sid] = resultado
    return resultado


def main() -> None:
    if not DIVERGENTES.exists():
        print("Arquivo pairs_layer2_divergent.jsonl não encontrado.")
        return

    divergentes = [
        json.loads(l)
        for l in DIVERGENTES.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    decididos = carregar_decididos()
    pendentes = [p for p in divergentes if p.get("source_id", "") not in decididos]

    limpar_tela()
    print(_cor("=== Layer 3 — Classificador Humano ===", NEGRITO))
    print(f"Total divergentes : {len(divergentes)}")
    print(f"Já decididos      : {len(decididos)}")
    print(f"Pendentes         : {len(pendentes)}")
    print(_cor("\nTeclas: [S] Concordante   [N] Rejeitar   [?] Incerto   [Q] Sair\n", CIANO))
    input("Pressione ENTER para começar...")

    concordantes_n = rejeitados_n = incertos_n = 0

    for idx, par in enumerate(pendentes, 1):
        n_ann = par.get("n_annotators", 0)

        if n_ann == 0:
            status = _cor(" ⚠ SEM ANOTAÇÃO AUTOMÁTICA", AMARELO)
        elif not par.get("agreement_direction"):
            status = _cor(" ⚠ DIREÇÃO DISCORDANTE", AMARELO)
        else:
            status = ""

        limpar_tela()

        # Traduz antes de exibir
        print(_cor(f"─── Par {idx} de {len(pendentes)}{status} ── traduzindo... ───", FRACO),
              end="\r", flush=True)
        trad = traduzir_par(par)

        limpar_tela()

        # Cabeçalho
        print(_cor(f"─── Par {idx} de {len(pendentes)}{status} ───", NEGRITO))
        print(f"{_cor('Fonte:', FRACO)} {par.get('source','')}  "
              f"{_cor('Domínio:', FRACO)} {par.get('domain','?')[:65]}")

        # Contexto traduzido
        contexto = trad.get("contexto", "").strip()
        if contexto:
            print(f"\n{_cor('Contexto do texto original:', CIANO)}")
            palavras = contexto.split()
            linha = "  "
            for palavra in palavras:
                if len(linha) + len(palavra) > 90:
                    print(linha)
                    linha = "  " + palavra + " "
                else:
                    linha += palavra + " "
            if linha.strip():
                print(linha)

        # Pergunta original traduzida
        print(f"\n{_cor('PERGUNTA ORIGINAL (q_bad):', VERMELHO)}")
        print(f"  {trad.get('q_bad', par['q_bad'])}")

        # Reformulação traduzida
        print(f"\n{_cor('REFORMULAÇÃO (q_good):', VERDE)}")
        print(f"  {trad.get('q_good', par['q_good'])}")

        # Anotadores automáticos
        print(f"\n{_cor('Anotadores automáticos:', FRACO)}")
        print(formatar_anotadores(par))

        # Critério rápido
        print(f"\n{_cor('Critério:', FRACO)} q_good tem metodologia mais clara e "
              "avança em direção à resposta da q_bad?")
        print()

        while True:
            try:
                tecla = input(_cor("  Decisão [S / N / ? / Q]: ", NEGRITO)).strip().upper()
            except (EOFError, KeyboardInterrupt):
                tecla = "Q"

            if tecla == "Q":
                adicionados, rej = aplicar_decisoes()
                print(f"\n💾 Salvo. {adicionados} adicionados à Layer 2, "
                      f"{rej} rejeitados.")
                restantes = len(pendentes) - idx
                if restantes > 0:
                    print(f"   {restantes} par(es) restante(s) — execute novamente para continuar.")
                return

            if tecla == "S":
                salvar_decisao(par, "concordante")
                concordantes_n += 1
                print(_cor("  ✔ Marcado como CONCORDANTE", VERDE))
                break
            if tecla == "N":
                salvar_decisao(par, "rejeitado")
                rejeitados_n += 1
                print(_cor("  ✗ Marcado como REJEITADO", VERMELHO))
                break
            if tecla == "?":
                salvar_decisao(par, "incerto")
                incertos_n += 1
                print(_cor("  ~ Marcado como INCERTO", AMARELO))
                break

            print("  Tecla inválida. Use S, N, ? ou Q.")

    # Todos classificados
    adicionados, rej = aplicar_decisoes()
    limpar_tela()
    print(_cor("=== Classificação concluída! ===\n", NEGRITO))
    print(f"  Concordantes : {concordantes_n}")
    print(f"  Rejeitados   : {rejeitados_n}")
    print(f"  Incertos     : {incertos_n}")
    print(f"\n  {adicionados} par(es) adicionados a pairs_layer2.jsonl")
    print(f"  {rej} par(es) rejeitados")
    print(f"\n  Decisões salvas em: {LOG_HUMANO}")


if __name__ == "__main__":
    main()
