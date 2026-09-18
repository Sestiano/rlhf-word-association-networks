# RLHF Word Association Networks

Exam project for the course *Cognitive Data Science* (Prof. Massimo Stella), MSc in Cognitive Science, CIMeC, University of Trento.

Analysis of RLHF's impact on the associative structure of LLMs using Text-Free Mental Networks (TFMN).

30 cue words (10 positive, 10 negative, 10 neutral) → LLM generates free associations → TFMN construction → Base vs RLHF (chat) comparison.

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

## Structure

```
rlhf-word-association-networks/
├── src/
│   ├── config.py              # Parameters and cue words
│   ├── 01_generate.py         # Association generation via LM Studio
│   ├── 02_build_network.py    # TFMN graph construction
│   ├── 03_metrics.py          # Metrics: degree, closeness, clustering
│   ├── 04_analysis.py         # Statistics + plots
│   └── 05_fmn_multiplex.py    # FMN ego-networks
├── data/
│   └── associations.json      # Generated associations (50 samples × 30 cues × 2 models)
├── results/                   # Networks, metrics, plots
├── run.py                     # Full pipeline (overnight)
└── requirements.txt
```

## Usage

**Individual steps:**
```bash
python src/01_generate.py         # requires LM Studio running
python src/02_build_network.py
python src/03_metrics.py
python src/04_analysis.py
python src/05_fmn_multiplex.py
```

**Full pipeline (overnight, requires LM Studio running):**
```bash
nohup python run.py > log.txt 2>&1 &
```

## Output

All files are generated in `results/` by the pipeline; the plots are not committed.

| File | Description |
|------|-------------|
| `network_comparison.png` | Degree distribution + CCDF — Base vs RLHF |
| `network_visual.png` | Full topology with community detection |
| `plutchik_comparison.png` | Plutchik emotional wheel |
| `metrics_comparison.png` | Degree, closeness, S/D ratio by valence |
| `fmn_ego_neutral.png` | FMN ego-networks — neutral words (anchor, mirror) |
| `fmn_ego_valenced.png` | FMN ego-networks — valenced words (treasure, ruin) |
| `metrics.csv` | Detailed per-cue metrics |

## License

MIT
