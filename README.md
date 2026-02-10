# Multiplex RLHF — Word Association Analysis

Analisi dell'impatto di RLHF sulla struttura associativa degli LLM usando Text-Free Mental Network (TFMN).

30 cue words (10 positive, 10 negative, 10 neutral) → LLM genera associazioni libere → costruzione TFMN → confronto Base vs RLHF (chat).

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

> **Google Colab:** apri [`multiplex_rlhf_colab.ipynb`](multiplex_rlhf_colab.ipynb) e segui le istruzioni — tutte le dipendenze vengono installate automaticamente.

## Struttura

```
multiplex/
├── src/
│   ├── config.py              # Parametri e cue words
│   ├── 01_generate.py         # Generazione associazioni via LM Studio
│   ├── 02_build_network.py    # Costruzione grafi TFMN
│   ├── 03_metrics.py          # Metriche: degree, closeness, clustering
│   ├── 04_analysis.py         # Statistica + plot
│   └── 05_fmn_multiplex.py    # FMN ego-networks
├── data/
│   └── associations.json      # Associazioni generate (50 samples × 30 cues × 2 modelli)
├── results/                   # Networks, metriche, plot
├── run.py                     # Pipeline completa (overnight)
├── multiplex_rlhf_colab.ipynb # Notebook per Google Colab
└── requirements.txt
```

## Uso

**Step singoli:**
```bash
python src/01_generate.py         # richiede LM Studio attivo
python src/02_build_network.py
python src/03_metrics.py
python src/04_analysis.py
python src/05_fmn_multiplex.py
```

**Pipeline completa (overnight, richiede LM Studio attivo):**
```bash
nohup python run.py > log.txt 2>&1 &
```

## Output

| File | Descrizione |
|------|-------------|
| `network_comparison.png` | Degree distribution + CCDF — Base vs RLHF |
| `network_visual.png` | Topologia completa con community detection |
| `plutchik_comparison.png` | Plutchik emotional wheel |
| `metrics_comparison.png` | Degree, closeness, S/D ratio per valenza |
| `fmn_ego_neutral.png` | FMN ego-networks — parole neutre (anchor, mirror) |
| `fmn_ego_valenced.png` | FMN ego-networks — parole valenzate (treasure, ruin) |
| `metrics.csv` | Metriche per-cue dettagliate |

## License

MIT
