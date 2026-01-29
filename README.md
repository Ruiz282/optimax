# Case 2: ChatGPT Launch 2022

**Week 3 | Entropy Concept: Sample Space Expansion**
**Scaffolding Level: HEAVY**

## Overview

This case examines the ChatGPT launch (November 30, 2022) as an example of **sample space expansion** — when the investment universe itself changes, not just the probabilities within it.

### The Central Question
> "When does a new game begin?"

### Key Contrast with Week 1
- **Week 1 (Tariff Shock):** P changed (transition probabilities shifted)
- **Week 3 (ChatGPT):** X changed (the sample space itself expanded)

## Learning Objectives

By the end of this session, students will be able to:

1. **Identify "sample space expansion" events** — When new asset classes emerge
2. **Map disruption cascades** — Trace second and third-order effects
3. **Distinguish "big news" from "new game" signals** — Paradigm shift vs. noise
4. **Apply entropy to sector analysis** — Measure creative destruction

## The Creative Destruction

| Metric | Nov 2022 | Nov 2024 | Change |
|--------|----------|----------|--------|
| Nvidia market cap | $345B | $4.4T | +1,175% |
| Nvidia revenue | $27B | $130B | +381% |
| Chegg stock price | $25 | $0.25 | -99% |
| Mag 7 S&P weight | ~20% | ~32% | +60% |

## Folder Structure

```
Case2_ChatGPT/
├── README.md                    # This file
├── slides/
│   └── Case2_Slides.md          # Marp presentation (15 slides)
├── teaching_notes/
│   └── instructor_guide.md      # 90-minute session guide
├── src/
│   ├── __init__.py
│   ├── starter_template.py      # DRIVER-structured starter code
│   └── validate_thesis.py       # Working thesis validation
├── tests/
│   ├── __init__.py
│   └── test_sample_space.py     # Unit tests for entropy/cascade
└── data/
    └── README.md                # Data sources documentation
```

## Quick Start

### Prerequisites
```bash
pip install numpy pandas matplotlib yfinance pytest
```

### Run the Starter Code
```bash
cd src
python starter_template.py
```

### Validate the Thesis
```bash
cd src
python validate_thesis.py
```

### Run Unit Tests
```bash
cd tests
pytest test_sample_space.py -v
```

## Key Thesis Claims

The validation code proves these claims:

1. **Massive Divergence (>800%)**: NVDA +700% vs CHGG -99%
2. **Concentration Increased**: Mag 7 weight 20% → 32%
3. **Sample Space Expanded**: "AI infrastructure" became new asset class
4. **Creative Destruction Measurable**: Quantifiable winner/loser divergence

## The Paradox

**Observed:**
- Sample space EXPANDED (new AI asset class)
- But entropy DECREASED (Mag 7 concentration increased)

**Explanation:**
- New category entered (expansion)
- But few players dominate that category (concentration)
- Net: Bigger universe, but more concentrated

This is the "first mover advantage" in a new investment category.

## Student Deliverables

1. **Disruption Cascade Map**: Network visualization showing winners, losers, and causal pathways
2. **Entropy Analysis**: Sector entropy before/after ChatGPT
3. **3-minute Video**: Explaining the cascade logic
4. **One-page Summary**: Executive summary of findings

## Evaluation Criteria

| Criterion | Weight |
|-----------|--------|
| Completeness of cascade mapping | 25% |
| Valid entropy quantification | 25% |
| Insight on second/third-order effects | 25% |
| Professional presentation | 25% |

## Key Dates

| Date | Event |
|------|-------|
| 2022-11-30 | ChatGPT Launch |
| 2023-05-02 | Chegg admits ChatGPT impact (-49% single day) |
| 2023-05-24 | Nvidia "blowout" AI earnings |

## Connection to Course Arc

| Week | Case | Builds On |
|------|------|-----------|
| 1 | Tariff Shock | Foundation: Regime shift detection |
| **3** | **ChatGPT Launch** | **Sample space expansion vs. regime shift** |
| 4 | GENIUS Act | Both expand sample space (tech vs. regulation) |
| 5 | Europe Energy | Both show irreversibility |

## Sources

1. [Yahoo Finance: Three Years of AI Mania](https://finance.yahoo.com/news/three-years-ai-mania-chatgpt-113000269.html)
2. [Visual Capitalist: Top Sectors Since ChatGPT](https://www.visualcapitalist.com/ranked-the-top-performing-sectors-since-chatgpt-launched/)
3. [CNBC: Chegg Says ChatGPT Killing Business](https://www.cnbc.com/2023/05/02/chegg-drops-more-than-40percent-after-saying-chatgpt-is-killing-its-business.html)
4. [Motley Fool: Magnificent Seven's Market Cap](https://www.fool.com/research/magnificent-seven-sp-500/)

---

*MGMT 69000: Mastering AI for Finance | Purdue MSF | DRIVER Framework*
*"Probability measures outcomes within a fixed game. Entropy measures when the game itself is changing."*
