# 03_metrics.py - Network Metrics
import json, pickle
import pandas as pd
import networkx as nx
from config import CUES

def get_cue_metrics(G, cue, closeness_dict):
    """Metriche per singolo cue word"""
    if cue not in G:
        return {"degree": 0, "strength": 0, "clustering": 0, "closeness": 0}
    return {
        "degree": G.degree(cue),
        "strength": sum(d["weight"] for _, _, d in G.edges(cue, data=True)),
        "clustering": nx.clustering(G, cue, weight="weight"),
        "closeness": closeness_dict.get(cue, 0)
    }

def global_metrics(G):
    """Metriche globali della rete"""
    gcc = max(nx.connected_components(G), key=len)
    Gc = G.subgraph(gcc)
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "components": nx.number_connected_components(G),
        "gcc_nodes": len(gcc),
        "avg_path_length": nx.average_shortest_path_length(Gc),
        "avg_clustering": nx.average_clustering(G, weight="weight"),
    }

def type_token_ratio(data):
    """TTR: parole uniche / parole totali nelle associazioni"""
    all_words = []
    for samples in data.values():
        for s in samples:
            all_words.extend(s["words"])
    if not all_words:
        return 0
    return len(set(all_words)) / len(all_words)

if __name__ == "__main__":
    with open("results/networks.pkl", "rb") as f:
        networks = pickle.load(f)
    with open("data/associations.json") as f:
        assoc = json.load(f)

    # Metriche globali
    print("\nGLOBAL METRICS")
    print("-" * 50)
    for model, G in networks.items():
        gm = global_metrics(G)
        ttr = type_token_ratio(assoc[model])
        print(f"  {model.upper()}: nodes={gm['nodes']}, edges={gm['edges']}, "
              f"density={gm['density']:.4f}, components={gm['components']}, "
              f"avg_path={gm['avg_path_length']:.2f}, TTR={ttr:.3f}")

    # Metriche per cue
    rows = []
    for model, G in networks.items():
        closeness = nx.closeness_centrality(G)
        for valence, cues in CUES.items():
            for cue in cues:
                m = get_cue_metrics(G, cue, closeness)
                rows.append({"model": model, "valence": valence, "cue": cue, **m})

    df = pd.DataFrame(rows)
    df.to_csv("results/metrics.csv", index=False)
    print("\nPER-CUE METRICS (mean by valence)")
    print(df.groupby(["model", "valence"])[["degree", "strength", "clustering", "closeness"]].mean())
    print("Salvato results/metrics.csv")
