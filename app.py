"""
ReformulatEE — Gradio standalone app.
Equivalente à interface do notebook, mas sem precisar do Jupyter.

Uso:
  .venv\\Scripts\\python app.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv

load_dotenv(override=True)

# Online (HF Space): usa Claude API — HF Inference API não suporta este modelo
# Local: auto-detecta (Ollama se disponível, senão Claude)
if os.getenv("SPACE_ID"):
    os.environ["INFERENCE_BACKEND"] = "claude"
else:
    os.environ.setdefault("INFERENCE_BACKEND", "auto")

import gradio as gr

from src.db.historico import init_db
from src.db.historico import registrar_feedback
from src.db.historico import salvar
from src.db.historico import ultimas

init_db()

_initialized = False


def _init():
    global _initialized
    if _initialized:
        return
    from src.rl.inference import _get_index

    _get_index()
    _initialized = True


_MAX_INPUT_CHARS = 500

# Rate limiting: máx 10 requisições por sessão por janela de 60 segundos
import collections
import threading
import time as _time

_rl_lock = threading.Lock()
_rl_windows: dict[str, collections.deque] = {}
_RL_MAX = 10
_RL_WINDOW = 60


def _check_rate_limit(session_id: str) -> bool:
    """Retorna True se a requisição está dentro do limite, False se excedeu."""
    now = _time.monotonic()
    with _rl_lock:
        if session_id not in _rl_windows:
            _rl_windows[session_id] = collections.deque()
        dq = _rl_windows[session_id]
        while dq and now - dq[0] > _RL_WINDOW:
            dq.popleft()
        if len(dq) >= _RL_MAX:
            return False
        dq.append(now)
        return True


def reformular_ui(pergunta, idioma, request: gr.Request = None):
    """Processa a pergunta e retorna (html, dataset, record_id, feedback_row, btn+, btn-)."""
    _vazio = (
        '<p style="color:orange">Digite uma pergunta.</p>',
        gr.update(),
        None,
        gr.update(visible=False),
        gr.update(interactive=True, value="👍  Boa reformulação"),
        gr.update(interactive=True, value="👎  Pode melhorar"),
    )
    if not pergunta.strip():
        return _vazio

    session_id = getattr(request, "session_hash", "anon") if request else "anon"
    if not _check_rate_limit(session_id):
        return (
            '<p style="color:orange">Limite de requisições atingido. Aguarde 1 minuto e tente novamente.</p>',
            gr.update(),
            None,
            gr.update(visible=False),
            gr.update(interactive=True, value="👍  Boa reformulação"),
            gr.update(interactive=True, value="👎  Pode melhorar"),
        )

    pergunta = pergunta[:_MAX_INPUT_CHARS]

    _init()

    from src.rl.inference import reformular
    from src.rl.inference import reformular_ptbr

    ptbr = idioma == "Português"

    try:
        r = reformular_ptbr(pergunta, n=8) if ptbr else reformular(pergunta, n=8)
    except Exception as e:
        print(f"[reformular_ui] erro: {e}")
        erro = (
            '<p style="color:red">Ocorreu um erro ao processar a pergunta. Tente novamente.</p>',
            gr.update(),
            None,
            gr.update(visible=False),
            gr.update(interactive=True, value="👍  Boa reformulação"),
            gr.update(interactive=True, value="👎  Pode melhorar"),
        )
        return erro

    if ptbr:
        entrada, saida = r.q_bad_pt, r.best_pt
        sub_in = f'<div style="color:#aaa;font-size:.82em;margin-top:3px">🔤 {r.q_bad_en}</div>'
        sub_out = f'<div style="color:#aaa;font-size:.82em;margin-top:3px">🔤 {r.best_en}</div>'
        cands, ee_bad, ee_best, passed = r.candidates, r.ee_bad, r.ee_best, r.stage1_pass
    else:
        entrada, saida = r.q_bad, r.best
        sub_in = sub_out = ""
        cands, ee_bad, ee_best, passed = r.candidates, r.ee_bad, r.ee_best, r.stage1_pass

    # Persiste no banco e obtém o ID para o feedback
    record_id = salvar(
        idioma=idioma,
        pergunta_orig=pergunta,
        pergunta_en=r.q_bad_en if ptbr else None,
        candidatos=cands,
        melhor=r.best_en if ptbr else r.best,
        melhor_pt=r.best_pt if ptbr else None,
        ee_antes=ee_bad,
        ee_depois=ee_best,
        stage1_pass=passed,
    )

    delta = ee_best - ee_bad
    ganho_pct = delta / max(ee_bad, 0.001) * 100
    filtro = "✅ PASS" if passed else "⚠️ Fallback"
    fc = "#27ae60" if passed else "#e67e22"

    def bar(v):
        pct = min(int(v * 100), 100)
        c = "#2ecc71" if pct >= 70 else "#f39c12" if pct >= 40 else "#e74c3c"
        return (
            f'<div style="display:inline-flex;align-items:center;gap:6px">'
            f'<div style="background:#ddd;border-radius:3px;height:8px;width:120px">'
            f'<div style="background:{c};width:{pct}%;height:100%;border-radius:3px"></div></div>'
            f'<code style="font-size:.82em">{v:.3f}</code></div>'
        )

    top = sorted(cands, key=lambda c: c["score"], reverse=True)
    rows = "".join(
        f'<tr style="{"font-weight:600;background:#f0fff4" if i==1 else ""};border-bottom:1px solid #eee;color:#333">'
        f'<td style="padding:4px 8px;text-align:center;color:#333">{"🟢" if c["ee"]>ee_bad+0.05 else "🔴"} {i}</td>'
        f'<td style="padding:4px 8px;font-family:monospace;color:#333">{c["ee"]:.3f}</td>'
        f'<td style="padding:4px 8px;color:#333">{c["text"]}</td></tr>'
        for i, c in enumerate(top, 1)
    )

    html = f"""
<div style="font-family:system-ui,sans-serif;line-height:1.5;max-width:860px">
  <div style="background:#f5f5f5;border-left:4px solid #bbb;padding:10px 14px;border-radius:4px;margin-bottom:10px;color:#333">
    <div style="font-size:.72em;color:#888;text-transform:uppercase">Pergunta original</div>
    <div style="margin-top:4px;color:#333">{entrada}</div>{sub_in}
  </div>
  <div style="background:#eafaf1;border-left:4px solid #2ecc71;padding:10px 14px;border-radius:4px;margin-bottom:14px;color:#333">
    <div style="font-size:.72em;color:#27ae60;text-transform:uppercase">Reformulação epistêmica</div>
    <div style="font-weight:600;font-size:1.04em;margin-top:4px;color:#111">{saida}</div>{sub_out}
  </div>
  <div style="display:flex;gap:28px;flex-wrap:wrap;margin-bottom:14px;font-size:.9em">
    <div><div style="font-size:.72em;color:#888">EE ANTES</div>{bar(ee_bad)}</div>
    <div><div style="font-size:.72em;color:#888">EE DEPOIS</div>{bar(ee_best)}</div>
    <div><div style="font-size:.72em;color:#888">GANHO</div>
      <span style="font-weight:700;color:#27ae60">+{delta:.3f} ({ganho_pct:.0f}%)</span></div>
    <div><div style="font-size:.72em;color:#888">FILTRO</div>
      <span style="color:{fc};font-weight:600">{filtro}</span></div>
  </div>
  <details>
    <summary style="cursor:pointer;color:#2980b9;font-size:.85em">▶ Ver {len(cands)} candidatos</summary>
    <table style="border-collapse:collapse;width:100%;margin-top:6px;font-size:.82em;color:#333">
      <thead><tr style="background:#f0f0f0;color:#333">
        <th style="padding:4px 8px">#</th>
        <th style="padding:4px 8px">EE</th>
        <th style="padding:4px 8px;text-align:left">Reformulação</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </details>
</div>
"""
    return (
        html,
        gr.update(samples=ultimas(8)),
        record_id,
        gr.update(visible=True),
        gr.update(interactive=True, value="👍  Boa reformulação"),
        gr.update(interactive=True, value="👎  Pode melhorar"),
    )


def dar_feedback(record_id, valor: int):
    """Registra feedback e desativa os botões."""
    if record_id is not None:
        registrar_feedback(record_id, valor)
    label = "✅ Obrigado!" if valor == 1 else "✅ Registrado!"
    return (
        gr.update(interactive=False, value=label if valor == 1 else "👍  Boa reformulação"),
        gr.update(interactive=False, value=label if valor == -1 else "👎  Pode melhorar"),
    )


with gr.Blocks(title="ReformulatEE", theme=gr.themes.Soft()) as app:
    gr.Markdown("## 🔬 ReformulatEE — Reformulação Epistêmica")
    gr.Markdown(
        "<p style='font-size:.8em;color:#888;margin:0 0 8px'>"
        "As perguntas submetidas são registradas anonimamente para fins de pesquisa e melhoria do modelo. "
        "Não submeta informações pessoais ou confidenciais."
        "</p>"
    )

    with gr.Row():
        with gr.Column(scale=3):
            inp_q = gr.Textbox(
                label="Pergunta de pesquisa",
                placeholder="Digite sua pergunta de pesquisa aqui...",
                lines=3,
            )
        with gr.Column(scale=1):
            inp_idioma = gr.Radio(
                choices=["Português", "English"], value="Português", label="Idioma"
            )

    btn = gr.Button("🔄 Reformular", variant="primary", size="lg")
    out = gr.HTML()

    # Feedback — oculto até haver resultado
    estado_id = gr.State(value=None)
    with gr.Row(visible=False) as feedback_row:
        btn_pos = gr.Button("👍  Boa reformulação", variant="secondary", size="sm")
        btn_neg = gr.Button("👎  Pode melhorar", variant="secondary", size="sm")

    _EXEMPLOS = [
        ["O que causa o envelhecimento biológico?", "Português"],
        ["O que é consciência?", "Português"],
        ["Livre-arbítrio existe?", "Português"],
        ["What is the meaning of life?", "English"],
        ["Is consciousness fundamental?", "English"],
        ["What causes aging?", "English"],
    ]
    _samples_iniciais = ultimas(8) or _EXEMPLOS

    historico = gr.Dataset(
        components=[inp_q, inp_idioma],
        samples=_samples_iniciais,
        label="📋 Últimas perguntas",
        headers=["Pergunta", "Idioma"],
    )

    # Wiring
    btn.click(
        fn=reformular_ui,
        inputs=[inp_q, inp_idioma],
        outputs=[out, historico, estado_id, feedback_row, btn_pos, btn_neg],
        concurrency_limit=5,
    )
    historico.click(
        fn=lambda x: x,
        inputs=[historico],
        outputs=[inp_q, inp_idioma],
    )
    btn_pos.click(
        fn=lambda rid: dar_feedback(rid, 1),
        inputs=[estado_id],
        outputs=[btn_pos, btn_neg],
    )
    btn_neg.click(
        fn=lambda rid: dar_feedback(rid, -1),
        inputs=[estado_id],
        outputs=[btn_pos, btn_neg],
    )


if __name__ == "__main__":
    import sys

    _on_spaces = bool(os.getenv("SPACE_ID"))  # HuggingFace Spaces detectado

    if not _on_spaces and sys.platform == "win32":
        # Mata processos na porta apenas localmente no Windows
        import subprocess

        def _kill_port(port: int):
            try:
                result = subprocess.run(["netstat", "-ano"], capture_output=True, text=True)
                for line in result.stdout.splitlines():
                    if f":{port} " in line and "LISTENING" in line:
                        pid = line.split()[-1]
                        if pid.isdigit():
                            subprocess.run(["taskkill", "/PID", pid, "/F"], capture_output=True)
            except Exception:
                pass

        _kill_port(7860)

    # HF Spaces requer 0.0.0.0; local usa 127.0.0.1
    host = "0.0.0.0" if _on_spaces else "127.0.0.1"
    app.launch(server_port=7860, server_name=host)
