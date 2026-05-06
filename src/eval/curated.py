"""
100 questoes anotadas manualmente para o gate da Fase 0.

label=1: questao com EE alta (respondivel, tratavel, nao-trivial)
label=0: questao com EE baixa (vaga, inatingivel, ou trivialmente parafrastica)

Fonte: construidas manualmente inspiradas em:
  - Mudancas de paradigma historicas (Kuhn, Lakatos)
  - Pares do tipo "essencia X" -> "mecanismo X"
  - Questoes bem-formadas da literatura de revisao sistematica
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CuratedQuestion:
    text: str
    label: int  # 1=alta EE, 0=baixa EE
    domain: str
    note: str = ""


CURATED: list[CuratedQuestion] = [
    # ── EE ALTA (label=1) ─────────────────────────────────────────────────────
    CuratedQuestion("What are the molecular mechanisms of CRISPR-Cas9 DNA repair?", 1, "biology"),
    CuratedQuestion(
        "How does transformer architecture scaling affect downstream task performance?", 1, "ml"
    ),
    CuratedQuestion(
        "What is the effect of sleep deprivation on hippocampal neurogenesis in rodents?",
        1,
        "neuroscience",
    ),
    CuratedQuestion(
        "How do ribosomes synthesize proteins from mRNA codon sequences?", 1, "biochemistry"
    ),
    CuratedQuestion(
        "What statistical methods are used to detect publication bias in meta-analyses?",
        1,
        "methodology",
    ),
    CuratedQuestion(
        "How does antibiotic resistance spread between bacterial populations via plasmid transfer?",
        1,
        "microbiology",
    ),
    CuratedQuestion(
        "What is the relationship between synaptic pruning and adolescent cognitive development?",
        1,
        "neuroscience",
    ),
    CuratedQuestion(
        "How do convolutional neural networks learn hierarchical visual representations?", 1, "ml"
    ),
    CuratedQuestion(
        "What are the measurable effects of urban heat islands on local precipitation patterns?",
        1,
        "climate",
    ),
    CuratedQuestion(
        "How does mRNA stability affect protein expression levels in eukaryotic cells?",
        1,
        "biochemistry",
    ),
    CuratedQuestion(
        "What is the dose-response relationship between particulate matter exposure and cardiovascular disease?",
        1,
        "epidemiology",
    ),
    CuratedQuestion(
        "How does working memory capacity correlate with fluid intelligence across age groups?",
        1,
        "psychology",
    ),
    CuratedQuestion(
        "What are the empirical predictors of treaty compliance in international environmental agreements?",
        1,
        "political science",
    ),
    CuratedQuestion(
        "How do feedback loops in the carbon cycle affect atmospheric CO2 concentration projections?",
        1,
        "climate",
    ),
    CuratedQuestion(
        "What is the relationship between gut microbiome diversity and immune response to vaccines?",
        1,
        "immunology",
    ),
    CuratedQuestion(
        "How do phonological awareness deficits predict reading difficulties in early childhood?",
        1,
        "education",
    ),
    CuratedQuestion(
        "What are the measurable cognitive effects of mindfulness-based stress reduction interventions?",
        1,
        "psychology",
    ),
    CuratedQuestion(
        "How does reward prediction error in dopaminergic neurons drive reinforcement learning?",
        1,
        "neuroscience",
    ),
    CuratedQuestion(
        "What computational methods can identify protein folding pathways from amino acid sequences?",
        1,
        "biochemistry",
    ),
    CuratedQuestion(
        "How does institutional trust mediate the relationship between inequality and political participation?",
        1,
        "political science",
    ),
    CuratedQuestion(
        "What are the epidemiological risk factors for type 2 diabetes onset in populations under 40?",
        1,
        "epidemiology",
    ),
    CuratedQuestion(
        "How does transfer learning reduce labeled data requirements in medical image classification?",
        1,
        "ml",
    ),
    CuratedQuestion(
        "What is the effect of minimum wage increases on employment rates in low-skill sectors?",
        1,
        "economics",
    ),
    CuratedQuestion(
        "How do epigenetic modifications inherited across generations affect disease susceptibility?",
        1,
        "genetics",
    ),
    CuratedQuestion(
        "What measurement approaches can operationalize scientific paradigm shifts quantitatively?",
        1,
        "methodology",
    ),
    CuratedQuestion(
        "How does linguistic framing of risk information affect decision-making under uncertainty?",
        1,
        "psychology",
    ),
    CuratedQuestion(
        "What are the detectable precursors of volcanic eruptions in seismic and geodetic signals?",
        1,
        "geology",
    ),
    CuratedQuestion(
        "How does group size affect collective decision accuracy in self-organizing systems?",
        1,
        "complexity",
    ),
    CuratedQuestion(
        "What neural correlates distinguish conscious from unconscious visual processing?",
        1,
        "neuroscience",
    ),
    CuratedQuestion(
        "How do market concentration levels affect innovation rates in pharmaceutical R&D?",
        1,
        "economics",
    ),
    CuratedQuestion(
        "What are the measurable effects of peer tutoring on mathematics achievement in secondary school?",
        1,
        "education",
    ),
    CuratedQuestion(
        "How does climate variability affect conflict onset probability in agriculturally dependent societies?",
        1,
        "political science",
    ),
    CuratedQuestion(
        "What are the operational signatures of active versus dormant scientific paradigms in citation networks?",
        1,
        "scientometrics",
    ),
    CuratedQuestion(
        "How does n-back training transfer to untrained working memory tasks?", 1, "psychology"
    ),
    CuratedQuestion(
        "What mechanisms underlie synaptic long-term potentiation at hippocampal CA1 synapses?",
        1,
        "neuroscience",
    ),
    CuratedQuestion(
        "How do social network structure and tie strength affect information diffusion rates?",
        1,
        "sociology",
    ),
    CuratedQuestion(
        "What are the measurable determinants of replication success across scientific disciplines?",
        1,
        "methodology",
    ),
    CuratedQuestion(
        "How does placebo analgesia modulate pain processing in the anterior cingulate cortex?",
        1,
        "neuroscience",
    ),
    CuratedQuestion(
        "What is the empirical relationship between patent scope and subsequent innovation in biotechnology?",
        1,
        "economics",
    ),
    CuratedQuestion(
        "How do RNA-binding proteins regulate alternative splicing under cellular stress conditions?",
        1,
        "biochemistry",
    ),
    CuratedQuestion(
        "What factors predict spontaneous remission rates in major depressive disorder without treatment?",
        1,
        "psychiatry",
    ),
    CuratedQuestion(
        "How does urban street network topology affect pedestrian injury rates?", 1, "public health"
    ),
    CuratedQuestion(
        "What are the computational signatures of overfitting in neural language model pretraining?",
        1,
        "ml",
    ),
    CuratedQuestion(
        "How does early bilingual experience affect executive function development in children?",
        1,
        "cognitive science",
    ),
    CuratedQuestion(
        "What is the relationship between forest fragmentation and species extinction debt timelines?",
        1,
        "ecology",
    ),
    CuratedQuestion(
        "How do randomized controlled trials detect heterogeneous treatment effects across subgroups?",
        1,
        "methodology",
    ),
    CuratedQuestion(
        "What are the neurophysiological correlates of expert chess pattern recognition?",
        1,
        "cognitive science",
    ),
    CuratedQuestion(
        "How does economic inequality affect intergenerational social mobility across OECD countries?",
        1,
        "economics",
    ),
    CuratedQuestion(
        "What are the measurable effects of teacher expectation on student academic outcomes?",
        1,
        "education",
    ),
    CuratedQuestion(
        "How do attentional blinks reflect temporal limits of conscious perception?",
        1,
        "cognitive science",
    ),
    # ── EE BAIXA (label=0) ────────────────────────────────────────────────────
    CuratedQuestion("What is the essence of life?", 0, "philosophy", "ontologicamente bloqueada"),
    CuratedQuestion(
        "What is the true nature of consciousness?", 0, "philosophy", "sem metodologia consensual"
    ),
    CuratedQuestion("What is the meaning of existence?", 0, "philosophy", "metafisica pura"),
    CuratedQuestion(
        "What is the ultimate purpose of the universe?", 0, "philosophy", "sem referente empirico"
    ),
    CuratedQuestion(
        "What is reality, fundamentally?", 0, "philosophy", "vaga demais para operacionalizar"
    ),
    CuratedQuestion("What is the soul?", 0, "philosophy", "sem metodologia de medicao"),
    CuratedQuestion("Is there a God?", 0, "theology", "nao falsificavel empiricamente"),
    CuratedQuestion("What is the nature of good and evil?", 0, "ethics", "normativa, nao empirica"),
    CuratedQuestion("What is beauty?", 0, "aesthetics", "sem metrica consensual"),
    CuratedQuestion("What is the true self?", 0, "philosophy", "conceito nao operacionalizavel"),
    CuratedQuestion(
        "What is the deeper meaning of suffering?", 0, "philosophy", "axiologica, nao empirica"
    ),
    CuratedQuestion(
        "Does free will exist?",
        0,
        "philosophy",
        "debatida ha seculos sem convergencia metodologica",
    ),
    CuratedQuestion(
        "What is time, really?", 0, "philosophy", "vaga sem especificacao de nivel de analise"
    ),
    CuratedQuestion(
        "What lies beyond the observable universe?", 0, "cosmology", "inacessivel por definicao"
    ),
    CuratedQuestion("What is the true nature of love?", 0, "philosophy", "sem operacionalizacao"),
    CuratedQuestion("What is the essence of justice?", 0, "philosophy", "normativa"),
    CuratedQuestion(
        "Is mathematics invented or discovered?", 0, "philosophy", "sem teste empirico concebivel"
    ),
    CuratedQuestion(
        "What is the relationship between mind and body?",
        0,
        "philosophy",
        "muito vaga; variantes tratáveis existem mas essa nao",
    ),
    CuratedQuestion(
        "What is the ultimate nature of matter?",
        0,
        "physics",
        "vaga demais; versoes especificas sao trataveis",
    ),
    CuratedQuestion(
        "Why is there something rather than nothing?", 0, "metaphysics", "sem metodo de resposta"
    ),
    CuratedQuestion(
        "What is the true source of human happiness?",
        0,
        "philosophy",
        "sem operacionalizacao consensual",
    ),
    CuratedQuestion("What does it mean to live a good life?", 0, "ethics", "normativa"),
    CuratedQuestion(
        "What is the nature of intelligence?",
        0,
        "philosophy",
        "vaga; versoes operacionalizadas existem mas essa nao",
    ),
    CuratedQuestion(
        "What is the deep structure of language?",
        0,
        "linguistics",
        "vaga; variantes especificas sao trataveis",
    ),
    CuratedQuestion(
        "What is the true nature of time and space?",
        0,
        "philosophy",
        "vaga; relatividade responde versoes especificas",
    ),
    CuratedQuestion("What is society, fundamentally?", 0, "sociology", "vaga demais"),
    CuratedQuestion("What is the essence of art?", 0, "aesthetics", "normativa e sem metodo"),
    CuratedQuestion(
        "What is the ultimate cause of everything?",
        0,
        "metaphysics",
        "regressão infinita sem âncora empirica",
    ),
    CuratedQuestion(
        "What is knowledge?",
        0,
        "epistemology",
        "muito vaga; epistemologia formal trata versoes especificas",
    ),
    CuratedQuestion(
        "What is the true nature of morality?", 0, "ethics", "metaética sem resolucao empirica"
    ),
    CuratedQuestion("What is life's purpose?", 0, "philosophy", "axiologica"),
    CuratedQuestion("What is truly real?", 0, "metaphysics", "sem criterio de verdade"),
    CuratedQuestion(
        "What is the nature of emotions?",
        0,
        "philosophy",
        "vaga; versoes especificas sao trataveis",
    ),
    CuratedQuestion(
        "What is the essence of power?", 0, "political philosophy", "sem metodo de teste"
    ),
    CuratedQuestion(
        "What is the true nature of identity?", 0, "philosophy", "sem criterio operacional"
    ),
    CuratedQuestion(
        "What is the fundamental nature of causation?", 0, "philosophy", "metafisica sem resolucao"
    ),
    CuratedQuestion(
        "What is the deepest nature of information?",
        0,
        "philosophy",
        "vaga; teoria da informacao trata versoes especificas",
    ),
    CuratedQuestion(
        "What is the ultimate nature of space?",
        0,
        "philosophy",
        "vaga; fisica trata versoes especificas",
    ),
    CuratedQuestion(
        "What is the true nature of creativity?", 0, "philosophy", "sem operacionalizacao"
    ),
    CuratedQuestion("What is the essence of democracy?", 0, "political philosophy", "normativa"),
    CuratedQuestion(
        "What is the nature of truth?",
        0,
        "epistemology",
        "metafisica; versoes logicas sao trataveis",
    ),
    CuratedQuestion(
        "What is the true nature of matter and energy?",
        0,
        "philosophy",
        "vaga; fisica trata versoes especificas",
    ),
    CuratedQuestion(
        "What is the fundamental nature of life and death?",
        0,
        "philosophy",
        "vaga; biologia trata versoes especificas",
    ),
    CuratedQuestion("What is the true nature of freedom?", 0, "philosophy", "normativa e vaga"),
    CuratedQuestion(
        "What is the essence of science?",
        0,
        "philosophy of science",
        "vaga; filosofia da ciencia trata versoes especificas",
    ),
    CuratedQuestion("What is the true nature of evil?", 0, "ethics", "normativa"),
    CuratedQuestion(
        "What is the fundamental nature of the mind?",
        0,
        "philosophy",
        "vaga; cognitive science trata versoes especificas",
    ),
    CuratedQuestion(
        "What is the nature of being?", 0, "ontology", "ontologia fundamental sem empiria"
    ),
    CuratedQuestion(
        "What is the essence of culture?", 0, "anthropology", "vaga demais para operacionalizar"
    ),
    CuratedQuestion(
        "What is the ultimate nature of reality and existence?",
        0,
        "metaphysics",
        "conjuncao de questoes vaga",
    ),
]


def get_curated() -> list[CuratedQuestion]:
    assert len(CURATED) == 100, f"Esperado 100 questoes, encontrado {len(CURATED)}"
    n_pos = sum(1 for q in CURATED if q.label == 1)
    n_neg = sum(1 for q in CURATED if q.label == 0)
    assert n_pos == 50 and n_neg == 50, f"Esperado 50/50, encontrado {n_pos}/{n_neg}"
    return CURATED
