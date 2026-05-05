# LLM Quest — Patches Técnicos e Reformulações (v3)

Documento complementar ao `Source.txt`. Endereça cada uma das seis direções identificadas na avaliação anterior, na ordem de dependência lógica (patches técnicos primeiro, reformulações depois — porque as reformulações dependem do conteúdo dos patches).

---

## 0. Classificação das seis direções

| # | Direção | Tipo | Depende de | Risco se ausente | Esforço |
|---|---|---|---|---|---|
| A | Especificação do `paradigm_classifier` | Patch técnico | — | **Crítico**: o classifier é black-box no design atual. Sem ele, `Tratabilidade(Q)` é indefinida. | Médio |
| B | Estratégia de dataset sem circularidade | Patch técnico | A (parcial) | **Crítico**: dataset enviesado reproduz os erros que o sistema deveria corrigir. | Alto |
| C | Critério de convergência | Patch técnico | — | **Médio**: pipeline pode rodar indefinidamente ou estagnar em ótimo local marginal. | Baixo |
| D | Reescrita do documento (v3) | Reformulação | A, B, C | Baixo: o original já circula. | Baixo |
| E | Plano de implementação faseado | Reformulação | A, B, C | Baixo, mas alto valor para captar colaboradores ou financiamento. | Médio |
| F | Pitch de uma página | Reformulação | D ou E | Nenhum: é um derivado opcional. | Baixo |

**Eixo de prioridade:** A → B → C são os blockers reais. D, E, F são re-apresentações que ficam triviais depois que A, B, C estão fixos. A operação abaixo segue essa ordem.

---

## A. Especificação do `paradigm_classifier`

### Função
`paradigm_classifier(Q) → [0, 1]` retorna a probabilidade de que a questão `Q` pertença a um domínio com paradigma de pesquisa ativo — ou seja, com metodologia, instrumentos e comunidade que tornam respostas concebíveis a curto/médio prazo. É a operacionalização da distinção de Rescher (ignorância temporária vs. ontológica).

### Arquitetura
Encoder pré-treinado (sentence-BERT ou equivalente em domínio científico, e.g. SciBERT) com cabeça de classificação binária, *acrescida de features estruturais não-textuais* extraídas de um corpus científico. As features estruturais carregam o sinal mais discriminativo:

| Feature | Como medir | O que sinaliza |
|---|---|---|
| **Densidade de citações** | Para os top-k documentos retornados via RAG sobre Q, quantas citações em cada um nos últimos 5 anos | Paradigma ativo gera literatura recente; paradigma morto gera apenas citações históricas |
| **Trajetória temporal** | Inclinação da curva de publicações por ano relacionadas a Q (regressão sobre últimos 20 anos) | Crescente = ativo; flat = maduro/saturado; decrescente ou ausente = ontológico ou abandonado |
| **Existência de instrumentos/datasets** | Match contra ontologia de instrumentos científicos (e.g. "telescópio", "PCR", "fMRI") e datasets públicos | Paradigmas operacionais têm artefatos materiais; especulativos não |
| **Razão termos-operacionais/termos-especulativos** | NER em Q, contagem de termos com referente experimental conhecido vs. termos abstratos | "Mecanismo de auto-replicação" (operacional) vs. "essência da vida" (especulativo) |
| **Diversidade de comunidade** | Número de instituições distintas publicando no domínio nos últimos 5 anos | Comunidade dispersa = paradigma estabelecido; comunidade ausente ou monolítica = problema |

### Calibração de ignorância temporária vs. ontológica
A distinção crítica é que questões podem migrar entre as categorias historicamente. "Qual a estrutura do átomo?" foi ontológica até 1900, depois temporária, hoje resolvida. Isso significa que o classifier não pode ser puramente sincrônico — precisa de um *modo de retropredição*:

- Treinar simultaneamente em (Q, ano, status) onde status ∈ {temporária, ontológica, resolvida} para cada ano.
- Para uma Q nova, gerar predições contrafactuais em t-50, t-25, t. Se o status muda entre os pontos, é evidência de paradigma em formação.
- Questões classificadas como ontológicas em t-50 mas temporárias em t são exatamente o tipo de exemplo positivo que se quer reproduzir — são casos onde uma reformulação produtiva ocorreu na história.

### Saída
Não retornar apenas um escalar, mas um vetor:
```
{
  prob_tractable: float,        # alimenta EE(Q)
  trajectory: 'rising' | 'plateau' | 'declining' | 'absent',
  confidence: float,            # baixa quando o RAG retorna pouco
  nearest_resolved_question: Q_known  # para auditoria
}
```
A confidence é importante: questões fora-de-distribuição (poucos documentos retornados) devem reduzir o peso de `Tratabilidade` na recompensa, não receber um valor inventado.

---

## B. Estratégia de dataset sem circularidade

### O risco
Se o dataset `(Q_mal, Q_bem)` é construído por um único LLM grande anotando pares, o sistema treinado nesse dataset herda os vieses epistêmicos do anotador. O resultado é um modelo que aprende a imitar "o que um LLM acha que é uma pergunta melhor", não "o que de fato é uma pergunta melhor". Isso é a falha de polite-liar elevada a sistema.

### Pipeline em quatro camadas

**Camada 1 — Pares atestados historicamente.** Não geração sintética livre. Extrair pares de fontes onde a transição `Q_mal → Q_bem` é documentada por humanos:

- Stanford Encyclopedia of Philosophy: entradas sobre mudanças de paradigma (ex.: "vitalism", "phlogiston", "ether").
- Estudos de caso de Kuhn, Lakatos, Laudan.
- Revisões sistemáticas (Cochrane, PRISMA): seções de "research question refinement".
- Histórico de prêmios Nobel: laudações frequentemente articulam por que a reformulação foi produtiva.
- Papers com markers explícitos: "we reformulate the question as", "the right question is not X but Y".

Esse núcleo é pequeno (estimativa: 200-500 pares de alta qualidade) mas tem ground truth real.

**Camada 2 — Multi-annotator com famílias divergentes.** Para expandir, usar três ou mais LLMs de famílias arquiteturais diferentes (Claude, GPT, Llama, Gemini) anotando pares candidatos independentemente. Manter apenas pares onde:
- Os anotadores concordam na direção (`Q_bem` é melhor que `Q_mal`).
- Concordam aproximadamente na magnitude (variância < threshold).

A divergência entre anotadores vira sinal de incerteza explícito (campo `disagreement` no exemplo), não é descartada.

**Camada 3 — Validação humana estratificada.** Amostragem por:
- Domínio (física, biologia, ciências sociais, filosofia).
- Era histórica do par.
- Nível de concordância dos anotadores.

Validadores humanos (idealmente filósofos da ciência ou metodólogos) marcam cada amostra como `agree`/`disagree`/`unclear`. Pares que falham a validação humana são removidos *e* usados como exemplos negativos (pares onde os LLMs concordaram entre si mas erraram).

**Camada 4 — Adversarial probes.** Conjunto explícito de armadilhas:
- `Q_bem_falso` = paráfrase sintaticamente sofisticada de `Q_mal` sem ganho epistêmico real.
- `Q_bem_falso` = reformulação que aumenta tratabilidade aparente mas perde proximidade semântica com a intenção original (responder Q_bem não responde Q_mal).
- `Q_bem_falso` = reformulação que substitui termos especulativos por outros igualmente especulativos.

O reward model deve atribuir EE baixa a esses casos. Se não atribui, é evidência de circularidade residual e dispara re-treino.

### Splits para evitar contaminação

- **Split temporal**: treinar em pares atestados pré-2010, testar em pares 2010+. Evita que o modelo vaze conhecimento de reformulações que já estão no corpus de pré-treino dos LLMs anotadores.
- **Split por domínio**: hold-out de um domínio inteiro (ex.: ciências sociais) para testar generalização.

### Reverse validation
Teste de saúde: dado apenas `Q_bem`, um avaliador independente deve conseguir prever que `EE(Q_bem)` é alta sem ver `Q_mal`. Se a previsão depende de ver o par, o sinal está em propriedades relacionais (pode ser superficial) e não em propriedades intrínsecas de `Q_bem`. Isso é um indicador de qualidade do dataset.

---

## C. Critério de convergência do pipeline

O design original tem o loop `Q* → Q_0 da próxima rodada` mas nenhuma condição de parada. Especificação:

### C.1 Convergência intra-rodada (geração de K candidatos a partir de Q_0)
Para cada rodada:
- **Parada por threshold**: se `max_i EE(Q_i) - EE(Q_0) < δ_intra` para todos os K candidatos, marca a rodada como "estagnada" e força exploração no próximo passo (ver C.3).
- **Parada por filtro vazio**: se nenhum Q_i passa o filtro de Estágio 1 (`EE(Q_i) ≤ EE(Q_0) + ε`), aumenta K ou aumenta temperatura de sampling. Se três rodadas consecutivas falham, marca Q_0 como "ponto fixo local" e pausa para inspeção humana.

### C.2 Convergência inter-rodadas (Q_t* → Q_{t+1} de partida)
- **Plateau de EE**: se `|EE(Q_t*) - EE(Q_{t-1}*)| < δ_outer` por N rodadas consecutivas (sugestão: N=3), considera convergido.
- **Detecção de ciclo**: se `Q_t*` cai dentro de uma ε-bola de algum `Q_{t-k}*` anterior, o loop entrou em ciclo. Parar e reportar.
- **Orçamento de exploração**: max_iterations como hard cap (sugestão: 10-20 rodadas).

### C.3 Anti-estagnação (injeção forçada de exploração)
O risco identificado na avaliação — convergência prematura para incremento marginal — é mitigado por:
- **Exploration injection**: a cada M rodadas (sugestão: M=5), forçar α=1 por uma rodada independente do schedule de currículo. Isso obriga geração de candidatos com EE máxima sem restrição de proximidade.
- **Diversidade intra-batch**: nos K candidatos de uma rodada, impor `min_pairwise_distance(Q_i, Q_j) > τ` para evitar que o sampling colapse para um único ponto.
- **KL-penalty contra histórico**: penalizar candidatos próximos a qualquer Q* anterior do trajeto, não só ao Q_0 atual.

### C.4 Detecção de regressão de qualidade
- Se `EE(Q_t*) < EE(Q_{t-1}*)` em uma rodada não-exploratória, é regressão. Rollback do update de policy e re-sample.
- Track de "adversarial probe pass rate" como média móvel. Se cair abaixo de threshold (ex.: 80%), parar treino e re-validar dataset (ver B).

### C.5 Parâmetros sugeridos (ponto de partida para tuning)
| Parâmetro | Valor inicial | Sentido |
|---|---|---|
| δ_intra | 0.05 | Mínima melhoria absoluta para considerar Q_i não-trivial dentro da rodada |
| δ_outer | 0.02 | Plateau threshold inter-rodadas |
| ε (filtro Estágio 1) | 0.10 | Margem mínima sobre EE(Q_0) para passar o filtro |
| τ (diversidade) | 0.3 (cosine) | Distância mínima entre candidatos da mesma rodada |
| M (exploration injection) | 5 | Frequência de rodada exploratória forçada |
| max_iterations | 15 | Hard cap |
| K (candidatos por rodada) | 16-32 | Função de orçamento computacional |

Todos esses são *hipóteses iniciais*. Calibração empírica é o trabalho da Fase 2 do roadmap (ver E).

---

## D. Reescrita do documento (v3 — diff focado)

Em vez de duplicar todo o `Source.txt`, lista de mudanças cirúrgicas que produzem a v3:

**Mantidas sem alteração:** Seções 1, 4, 5, 6.

**Substituições:**
- Seção 2 (operacionalização de EE): substituir o parágrafo sobre `Tratabilidade` pela referência ao patch A. Adicionar nota sobre ponderação dinâmica de `Tratabilidade` proporcional a `confidence` retornado pelo classifier.
- Seção 2 (esboço de código): atualizar a função `reward` para refletir o output rico do classifier:
  ```python
  def reward(Q, Q0, alpha, corpus):
      respondibilidade = rag_score(Q, corpus)
      tract_out = paradigm_classifier(Q)
      tratabilidade = tract_out.prob_tractable * tract_out.confidence
      nao_trivialidade = bell_distance(embed(Q), embed(Q0))

      EE = beta1*respondibilidade + beta2*tratabilidade + beta3*nao_trivialidade
      Prox = cosine_similarity(embed(Q), embed(Q0))

      return alpha * EE + (1 - alpha) * Prox
  ```
- Seção 3: adicionar parágrafo final referenciando os adversarial probes do patch B como mecanismo concreto que operacionaliza a "penalização explícita de pseudo-reformulação".
- Seção 7 (dataset): substituir os dois últimos parágrafos pela seção B inteira.

**Adicionadas:**
- Seção 8 — Critério de convergência (todo o conteúdo de C).
- Seção 9 — Discussão de hiperparâmetros: tornar explícito que β₁, β₂, β₃ são objeto de tuning e propor estratégia (grid search com validação em adversarial probes; ou aprender β como meta-parâmetros via bilevel optimization).
- Apêndice — Glossário operacional (`Q_0`, `Q*`, EE, Prox, K, α, ε, δ).

**Reordenação:** mover seção 7 (dataset) para depois da seção 2 (operacionalização), porque sem dataset o reward model é teoria.

---

## E. Plano de implementação faseado

### Fase 0 — Fundação (semanas 1-4)
**Objetivo:** ter EE(Q) computável end-to-end, mesmo que rudimentar.
- Construir corpus RAG (filosofia da ciência + arXiv subset).
- Implementar `Respondibilidade` via RAG simples (BM25 + cross-encoder).
- Implementar `Não-trivialidade` via cosine + função sino com parâmetros heurísticos.
- `Tratabilidade` stub: classifier zero-shot via prompting.
- **Gate de saída:** EE(Q) avalia 100 questões manualmente curadas (50 boas, 50 ruins) com AUC > 0.7.

### Fase 1 — Dataset e classifier real (semanas 5-12)
**Objetivo:** trocar o stub de Tratabilidade por classifier treinado e ter dataset robusto.
- Executar Camadas 1-4 do pipeline de dataset (B).
- Treinar `paradigm_classifier` com features estruturais (A).
- Construir adversarial probe set (~200 pares).
- **Gate de saída:** classifier passa nos adversarial probes (>85% accuracy); concordância com validadores humanos em pares hold-out > 0.7 Cohen's κ.

### Fase 2 — Reward model integrado (semanas 13-20)
**Objetivo:** EE(Q) calibrado, não treina ainda — só pontua.
- Tuning de β₁, β₂, β₃ via grid search sobre adversarial probes.
- Sanity check: para todo `(Q_mal, Q_bem)` conhecido, `EE(Q_bem) > EE(Q_mal)` em >95% dos casos.
- Implementar pipeline de score sobre output de um LLM gerador off-the-shelf (sem treinar).
- **Gate de saída:** reward model identifica corretamente questões reformuladas-bem em test set hold-out temporal.

### Fase 3 — RL training (semanas 21-32)
**Objetivo:** treinar o LLM gerador via PPO ou DPO.
- Implementar pipeline RLHF com EE como sinal de recompensa.
- Currículo de α (annealing crescente).
- Implementar critério de convergência (C) com monitoramento ao vivo.
- **Gate de saída:** modelo treinado supera baseline em geração de Q* com EE > EE(Q_0) em test set, sem degradar em adversarial probes.

### Fase 4 — Não-trivialidade Nível 3 (semanas 33+)
**Objetivo:** operacionalizar o grafo de dependência inferencial entre questões.
- Construir grafo Q → Q' onde aresta significa "responder Q' contribui para responder Q".
- Métrica de proximidade inferencial (caminho mais curto, peso por confiança da inferência).
- Substituir/aumentar `Não-trivialidade` por essa métrica.
- **Gate de saída:** prova de conceito em domínio restrito (ex.: filosofia da biologia) onde o grafo melhora geração comparado à v2.

### Riscos por fase
| Fase | Risco principal | Mitigação |
|---|---|---|
| 0 | Corpus insuficiente | Começar restrito a um domínio bem documentado |
| 1 | Dataset enviesado | Pipeline de quatro camadas (B) é exatamente o anti-corpo |
| 2 | β-tuning acha ótimo espúrio | Validar contra adversarial probes, não só train loss |
| 3 | Reward hacking (modelo aprende a explorar imperfeições do reward) | Monitoramento contínuo dos probes; rollback automático |
| 4 | Grafo inferencial é não-construível em escala | Restringir a domínios específicos antes de generalizar |

---

## F. Pitch técnico de uma página

> **LLM Quest — geração de perguntas com efetividade epistêmica**
>
> **O problema.** Modelos de linguagem treinados com RLHF padrão geram perguntas que parecem profundas mas frequentemente são tão inatingíveis quanto a pergunta original — apenas com vocabulário mais sofisticado. É o "polite liar" aplicado à formulação de problemas.
>
> **A proposta.** Substituir o sinal de recompensa "humano prefere esta resposta" por um sinal computável de *efetividade epistêmica*: `EE(Q) = β₁·Respondibilidade + β₂·Tratabilidade + β₃·Não-trivialidade`. Combinado a um filtro que rejeita reformulações que não aumentam EE acima da pergunta original, isso força o modelo a produzir transformações genuinamente produtivas, não apenas estilizadas.
>
> **Como.** Pipeline em loop fechado: gerador propõe K candidatos → reward model pontua EE → filtro descarta triviais → currículo de α annealing controla a tensão entre exploração (EE alta) e proximidade (fidelidade à intenção original) → policy update → iteração.
>
> **Os três blockers reais.**
> 1. `paradigm_classifier`: distinguir computacionalmente ignorância temporária (paradigma ativo) de ontológica (impossível em princípio). Resolvido com features estruturais (densidade de citações, trajetória temporal, existência de instrumentos) somadas a retropredição histórica.
> 2. Dataset sem circularidade: pares (Q_mal, Q_bem) atestados historicamente como núcleo, multi-annotator com famílias divergentes, validação humana, adversarial probes, splits temporais.
> 3. Convergência: critério de parada explícito, anti-estagnação por exploration injection forçada a cada M rodadas, detecção de regressão por probe pass rate.
>
> **A contribuição original.** Não é o pipeline RLHF — é a **definição operacional de não-trivialidade epistêmica de Nível 3**: Q é não-trivial em relação a Q_0 se existe caminho inferencial Q → Q_0 no grafo de questões. Nenhum reward model atual implementa isso satisfatoriamente.
>
> **Próximo passo.** Fase 0 (fundação): ter EE(Q) computável em 4 semanas, mesmo rudimentar. Gate: AUC > 0.7 em 100 questões curadas. Sem isso, o resto é especulação.

---

## Apêndice — Notas de rastreamento

- Origem: `Source.txt` (v2), avaliado em Cowork session.
- Patches A, B, C são pré-requisitos para qualquer implementação séria.
- D, E, F são produtos de comunicação derivados, com dependência indicada na seção 0.
- O patch mais arriscado é B (dataset). É também o de maior alavancagem: dataset bom resgata um classifier mediano; dataset ruim trava qualquer classifier.
