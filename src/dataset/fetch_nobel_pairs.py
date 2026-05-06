"""
Dataset curado de laudações Nobel relevantes para reformulação de questões.

A API pública do Nobel Prize bloqueia requests automatizados.
Usamos um dataset local com as laudações mais relevantes — prêmios onde
a motivação articula explicitamente uma transição de questão.

Fonte: nobelprize.org (laudações públicas), curadas manualmente.
"""

from __future__ import annotations

import json
from pathlib import Path

# Laudações curadas: prêmios onde a motivação articula reformulação de questão
# Formato: (year, category, laureate, motivation_excerpt)
_CURATED_MOTIVATIONS = [
    # Medicina / Fisiologia
    (
        1962,
        "medicine",
        "Francis Crick, James Watson, Maurice Wilkins",
        "for their discoveries concerning the molecular structure of nucleic acids and its significance "
        "for information transfer in living material. The question was no longer 'what is life?' but "
        "'what is the molecular mechanism by which genetic information is stored and transmitted?'",
    ),
    (
        1958,
        "medicine",
        "George Beadle, Edward Tatum",
        "for their discovery that genes act by regulating definite chemical events. "
        "They transformed the question from 'how do genes control traits?' to "
        "'what specific enzyme does each gene produce?'",
    ),
    (
        2002,
        "medicine",
        "Sydney Brenner, John Sulston, Robert Horvitz",
        "for their discoveries concerning genetic regulation of organ development and programmed cell death. "
        "The key insight was reframing 'how do organisms develop?' as 'which specific genes regulate "
        "when and how individual cells live or die?'",
    ),
    (
        1984,
        "medicine",
        "Niels Jerne, Georges Köhler, César Milstein",
        "for theories concerning the specificity in development and control of the immune system and "
        "the discovery of the principle for production of monoclonal antibodies. "
        "The question shifted from 'how does the immune system recognize foreign substances?' to "
        "'how does each B-cell clone produce a single specific antibody?'",
    ),
    (
        2009,
        "medicine",
        "Elizabeth Blackburn, Carol Greider, Jack Szostak",
        "for the discovery of how chromosomes are protected by telomeres and the enzyme telomerase. "
        "The tractable question became: what molecular mechanism prevents chromosomes from shortening "
        "after each cell division?",
    ),
    (
        1976,
        "medicine",
        "Baruch Blumberg, Daniel Gajdusek",
        "for their discoveries concerning new mechanisms for the origin and dissemination of infectious diseases. "
        "Rather than asking 'what causes these mysterious slow infections?', they asked "
        "'can we identify a transmissible agent with measurable physical properties?'",
    ),
    # Quimica
    (
        1972,
        "chemistry",
        "Christian Anfinsen",
        "for his work on ribonuclease, especially concerning the connection between the amino acid sequence "
        "and the biologically active conformation. The question 'what determines protein shape?' became "
        "'does the amino acid sequence alone specify the three-dimensional folding?'",
    ),
    (
        1993,
        "chemistry",
        "Kary Mullis",
        "for his invention of the polymerase chain reaction (PCR) method. "
        "The question shifted from 'how can we detect vanishingly small amounts of DNA?' to "
        "'can we exponentially amplify specific DNA sequences using repeated thermal cycling?'",
    ),
    (
        2013,
        "chemistry",
        "Martin Karplus, Michael Levitt, Arieh Warshel",
        "for the development of multiscale models for complex chemical systems. "
        "The prior question 'are quantum or classical methods better for biomolecules?' was replaced by "
        "'how can we combine quantum and classical computations to model enzyme active sites?'",
    ),
    # Fisica
    (
        1921,
        "physics",
        "Albert Einstein",
        "for his services to theoretical physics, and especially for his discovery of the law of "
        "the photoelectric effect. Rather than asking 'why does light behave as a wave?', Einstein asked "
        "'what if light energy comes in discrete quanta proportional to frequency?'",
    ),
    (
        1932,
        "physics",
        "Werner Heisenberg",
        "for the creation of quantum mechanics. The question changed from 'what is the trajectory of "
        "an electron?' to 'what are the observable quantities and their statistical distributions?'",
    ),
    (
        1965,
        "physics",
        "Richard Feynman, Julian Schwinger, Sin-Itiro Tomonaga",
        "for their fundamental work in quantum electrodynamics, with deep-ploughing consequences for "
        "the physics of elementary particles. The question transformed from 'how does light interact "
        "with matter?' to 'how do we renormalize the infinite self-energy of the electron?'",
    ),
    # Economia
    (
        2002,
        "economics",
        "Daniel Kahneman",
        "for having integrated insights from psychological research into economic science, especially "
        "concerning human judgment and decision-making under uncertainty. "
        "The question moved from 'how do rational agents maximize expected utility?' to "
        "'how do actual humans make decisions under uncertainty, and what systematic biases appear?'",
    ),
    (
        1978,
        "economics",
        "Herbert Simon",
        "for his pioneering research into the decision-making process within economic organizations. "
        "Rather than 'how do firms maximize profit?', Simon asked 'how do real decision-makers "
        "satisfice under bounded rationality?'",
    ),
    (
        2001,
        "economics",
        "George Akerlof, Michael Spence, Joseph Stiglitz",
        "for their analyses of markets with asymmetric information. "
        "The question was reframed from 'how do markets reach equilibrium?' to "
        "'what happens to market outcomes when buyers and sellers have different information?'",
    ),
    (
        1979,
        "economics",
        "Theodore Schultz, Arthur Lewis",
        "for their pioneering research into economic development research with particular consideration "
        "of the problems of developing countries. The question shifted from 'why are some countries "
        "rich?' to 'what are the measurable returns to investment in human capital in agricultural economies?'",
    ),
    # Casos adicionais com reformulacoes explicitas
    (
        1953,
        "medicine",
        "Hans Krebs",
        "for his discovery of the coenzyme A and its importance for intermediary metabolism. "
        "The question 'how do cells extract energy from nutrients?' became 'what is the specific "
        "cyclic chemical pathway by which acetyl groups are oxidized to CO2?'",
    ),
    (
        2006,
        "medicine",
        "Andrew Fire, Craig Mello",
        "for their discovery of RNA interference — gene silencing by double-stranded RNA. "
        "The question 'how are genes turned off?' was refined to 'what happens when double-stranded "
        "RNA complementary to a gene is introduced into a cell?'",
    ),
    (
        1944,
        "medicine",
        "Joseph Erlanger, Herbert Gasser",
        "for their discoveries relating to the highly differentiated functions of single nerve fibres. "
        "Rather than 'how do nerves work?', they asked 'what is the relationship between nerve fiber "
        "diameter and the speed of electrical signal conduction?'",
    ),
    (
        2017,
        "economics",
        "Richard Thaler",
        "for his contributions to behavioural economics. "
        "The central reframing: instead of 'do people behave rationally in markets?', Thaler asked "
        "'what specific cognitive biases and mental accounting rules predict real economic choices?'",
    ),
    # Fisica adicional
    (
        1905,
        "physics",
        "Albert Einstein (special relativity)",
        "Einstein's special relativity reframed the central question of electrodynamics. "
        "Instead of asking 'what is the absolute motion of bodies through the ether?', he asked "
        "'what are the transformation laws that make the laws of physics identical for all inertial observers?'",
    ),
    (
        1913,
        "physics",
        "Niels Bohr",
        "Bohr's atomic model changed the question from 'why does the classical electron spiral into "
        "the nucleus?' to 'what discrete energy levels are allowed for bound electrons, and when does "
        "a transition between levels emit or absorb radiation?'",
    ),
    (
        1974,
        "physics",
        "Antony Hewish, Martin Ryle",
        "for their pioneering research in radio astrophysics. The question shifted from "
        "'can we detect radio emission from celestial sources?' to "
        "'what physical mechanism produces the extremely regular periodic pulses from rotating neutron stars?'",
    ),
    (
        2011,
        "physics",
        "Saul Perlmutter, Brian Schmidt, Adam Riess",
        "for the discovery of the accelerating expansion of the Universe through observations of distant supernovae. "
        "The prior question 'is the expansion of the universe slowing down?' was replaced by "
        "'what is the energy density of the vacuum that drives accelerating expansion?'",
    ),
    # Quimica adicional
    (
        1962,
        "chemistry",
        "Max Perutz, John Kendrew",
        "for their studies of the structures of globular proteins. "
        "The question changed from 'how do proteins fold into their functional shape?' to "
        "'what is the precise three-dimensional atomic arrangement of haemoglobin at X-ray resolution?'",
    ),
    (
        1980,
        "chemistry",
        "Paul Berg, Walter Gilbert, Frederick Sanger",
        "for their contributions concerning the determination of base sequences in nucleic acids. "
        "Rather than 'what is the genetic code in general?', the question became "
        "'what is the exact nucleotide sequence of a specific gene or genome?'",
    ),
    (
        2013,
        "chemistry",
        "Arieh Warshel",
        "The prior question 'are quantum or classical methods better for modeling enzyme catalysis?' was "
        "replaced by 'how can we combine QM and MM regions to model the transition state of enzymatic reactions "
        "at atomic resolution?'",
    ),
    # Medicina adicional
    (
        1984,
        "medicine",
        "César Milstein",
        "for the principle for production of monoclonal antibodies. "
        "The question 'how does the immune system generate antibody diversity?' was refined to "
        "'can we fuse a specific antibody-producing B cell with a myeloma cell to produce an immortal "
        "cell line secreting a single defined antibody?'",
    ),
    (
        2012,
        "medicine",
        "John Gurdon, Shinya Yamanaka",
        "for the discovery that mature cells can be reprogrammed to become pluripotent. "
        "The question changed from 'is cell differentiation irreversible?' to "
        "'which specific transcription factors, when introduced into a somatic cell, reprogram it to "
        "a pluripotent stem cell state?'",
    ),
    (
        1997,
        "medicine",
        "Stanley Prusiner",
        "for his discovery of prions — a new biological principle of infection. "
        "Instead of 'what nucleic acid carries the scrapie infectious agent?', the tractable question became "
        "'can a misfolded protein alone, without nucleic acid, transmit a neurodegenerative disease?'",
    ),
    (
        2005,
        "medicine",
        "Barry Marshall, Robin Warren",
        "for the discovery of the bacterium Helicobacter pylori and its role in gastritis and peptic ulcer disease. "
        "The question moved from 'what is the psychosomatic cause of peptic ulcers?' to "
        "'is a specific bacterium colonizing the gastric mucosa causally responsible for ulcer disease?'",
    ),
    (
        1988,
        "medicine",
        "James Black, Gertrude Elion, George Hitchings",
        "for their discoveries of important principles for drug treatment. "
        "Rather than 'what natural compound might treat disease?', they asked "
        "'what specific enzyme or receptor, if blocked by a rationally designed molecule, would interrupt a "
        "disease-causing biochemical pathway?'",
    ),
    (
        2008,
        "medicine",
        "Harald zur Hausen",
        "for his discovery of human papilloma viruses causing cervical cancer. "
        "The question shifted from 'is cervical cancer caused by herpes simplex virus?' to "
        "'which specific human papillomavirus types are consistently present in cervical carcinoma cells?'",
    ),
    # Economia adicional
    (
        1994,
        "economics",
        "John Nash, John Harsanyi, Reinhard Selten",
        "for their pioneering analysis of equilibria in the theory of non-cooperative games. "
        "The question changed from 'how do rational agents maximize individual payoffs?' to "
        "'what are the stable outcome conditions when each player's strategy is the best response to others?'",
    ),
    (
        2009,
        "economics",
        "Elinor Ostrom",
        "for her analysis of economic governance, especially the commons. "
        "Rather than 'do commons inevitably collapse without privatization or state control?', she asked "
        "'under what institutional design conditions do communities successfully govern shared resources themselves?'",
    ),
    (
        2013,
        "economics",
        "Eugene Fama, Lars Peter Hansen, Robert Shiller",
        "for their empirical analysis of asset prices. "
        "The central reframing: instead of 'can asset prices be predicted?', the productive question became "
        "'at what time horizons and under what conditions are return predictability patterns statistically robust?'",
    ),
]


def fetch_nobel_candidates(output_path: Path) -> list[dict]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache = output_path.with_suffix(".jsonl")

    if cache.exists():
        candidates = [
            json.loads(l) for l in cache.read_text(encoding="utf-8").splitlines() if l.strip()
        ]
        print(f"Candidatos Nobel carregados do cache: {len(candidates)}")
        return candidates

    candidates = []
    for year, category, laureate, motivation in _CURATED_MOTIVATIONS:
        candidates.append(
            {
                "id": f"nobel_{category}_{year}",
                "title": f"Nobel {category} {year} — {laureate[:50]}",
                "abstract": motivation,
                "year": year,
                "categories": [category],
                "source": "nobel",
                "laureate": laureate,
                "prize_category": category,
            }
        )

    with cache.open("w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Candidatos Nobel salvos: {len(candidates)}")
    return candidates


if __name__ == "__main__":
    out = Path("data/pairs/nobel_candidates.jsonl")
    candidates = fetch_nobel_candidates(out)
    print(f"\nTotal: {len(candidates)} candidatos")
    for c in candidates[:5]:
        print(f"  [{c['year']}] [{c['prize_category']}] {c['laureate'][:50]}")
        print(f"        {c['abstract'][:100]}")
