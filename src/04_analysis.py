# 04_analysis.py - Statistical Analysis and Plots
# Generates: network_comparison.png, plutchik_comparison.png, metrics_comparison.png
import pickle, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import networkx as nx
from scipy.stats import mannwhitneyu
from emoatlas import EmoScores
from config import CUES

PLUTCHIK_COLORS = {
    'joy': '#FFE600', 'trust': '#8BC34A', 'fear': '#4CAF50', 'surprise': '#00BCD4',
    'sadness': '#2196F3', 'disgust': '#9C27B0', 'anger': '#F44336', 'anticipation': '#FF9800'
}
VALENCE_COLORS = {'positive': '#3498DB', 'negative': '#E74C3C', 'neutral': '#95A5A6'}
POSITIVE_EMOTIONS = {'joy', 'trust', 'anticipation'}
NEGATIVE_EMOTIONS = {'anger', 'fear', 'sadness', 'disgust'}

# --- Utility ---

def cohens_d(x, y):
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2: return 0
    pooled = np.sqrt(((n1-1)*np.var(x,ddof=1) + (n2-1)*np.var(y,ddof=1)) / (n1+n2-2))
    return (np.mean(x) - np.mean(y)) / pooled if pooled else 0

def get_valence(word, es):
    try:
        scores = es.zscores(word)
        pos = sum(scores.get(e, 0) for e in POSITIVE_EMOTIONS)
        neg = sum(scores.get(e, 0) for e in NEGATIVE_EMOTIONS)
        return 'positive' if pos > neg else ('negative' if neg > pos else 'neutral')
    except:
        return 'neutral'

def analyze_emotions(G, top_n=100):
    """Plutchik scores for top-N nodes"""
    es = EmoScores()
    top_nodes = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)[:top_n]
    counts = {e: 0 for e in PLUTCHIK_COLORS}
    for word in top_nodes:
        try:
            scores = es.zscores(word)
            for emo, s in scores.items():
                if s > 0: counts[emo] += s
        except: pass
    return counts

# --- Plot: network_comparison.png ---

def plot_network_comparison(networks, output="results/network_comparison.png"):
    """Degree distribution comparison: histogram + CCDF"""
    colors = {'base': '#3498DB', 'chat': '#E74C3C'}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor='white')

    # Panel 1: Degree histogram
    ax = axes[0]
    for model, G in networks.items():
        degs = sorted([G.degree(n) for n in G.nodes()])
        bins = np.arange(0, max(degs)+10, 5)
        ax.hist(degs, bins=bins, alpha=0.55, color=colors[model],
                label=f"{model.upper()} (N={G.number_of_nodes()}, "
                      f"E={G.number_of_edges()})",
                edgecolor='white', linewidth=0.5)
        avg_deg = np.mean(degs)
        ax.axvline(avg_deg, color=colors[model], linestyle='--', linewidth=1.5,
                   alpha=0.8)
        ax.text(avg_deg + 2, ax.get_ylim()[1]*0.9 if model == 'base' else ax.get_ylim()[1]*0.8,
                f'μ={avg_deg:.1f}', color=colors[model], fontsize=9, fontweight='bold')

    ax.set_xlabel('Degree', fontsize=11)
    ax.set_ylabel('Number of nodes', fontsize=11)
    ax.set_title('Degree Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, frameon=True, fancybox=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.2)

    # Panel 2: CCDF (complementary cumulative distribution)
    ax2 = axes[1]
    for model, G in networks.items():
        degs = sorted([G.degree(n) for n in G.nodes()])
        ccdf = 1 - np.arange(1, len(degs)+1) / len(degs)
        ax2.plot(degs, ccdf, color=colors[model], linewidth=2,
                 label=f'{model.upper()}', alpha=0.85)
        ax2.fill_between(degs, ccdf, alpha=0.08, color=colors[model])

    ax2.set_xlabel('Degree (k)', fontsize=11)
    ax2.set_ylabel('P(degree ≥ k)', fontsize=11)
    ax2.set_title('Complementary Cumulative Distribution', fontsize=12, fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend(fontsize=9, frameon=True, fancybox=True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.2)

    plt.suptitle('Network Structure Comparison: Base vs RLHF',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved {output}")

# --- Plot: plutchik_comparison.png ---

def plot_plutchik_comparison(networks, output="results/plutchik_comparison.png"):
    """Radar/flower chart of Plutchik emotions per model"""
    emo_base = analyze_emotions(networks['base'])
    emo_chat = analyze_emotions(networks['chat'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), subplot_kw=dict(polar=True))

    for ax, (title, emo) in zip(axes, [('BASE', emo_base), ('CHAT (RLHF)', emo_chat)]):
        emotions = list(PLUTCHIK_COLORS.keys())
        values = [emo[e] for e in emotions]
        colors = [PLUTCHIK_COLORS[e] for e in emotions]
        mx = max(values) if max(values) > 0 else 1
        norm = [v / mx for v in values]
        angles = np.linspace(0, 2*np.pi, len(emotions), endpoint=False)

        for i, (em, col) in enumerate(zip(emotions, colors)):
            theta = np.linspace(angles[i]-np.pi/8, angles[i]+np.pi/8, 30)
            r = norm[i] * np.cos(np.linspace(-np.pi/2, np.pi/2, 30))**0.5
            ax.fill(theta, r, color=col, alpha=0.7, edgecolor='black', linewidth=1)
            ax.text(angles[i], 1.2, f"{em[:3]}\n({values[i]:.0f})", ha='center', fontsize=8, fontweight='bold')

        ax.set_ylim(0, 1.3)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.grid(True, alpha=0.3)

    plt.suptitle('Plutchik Emotional Wheel - Base vs RLHF', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved {output}")

# --- Plot: metrics_comparison.png ---

def plot_metrics_comparison(df, assoc, output="results/metrics_comparison.png"):
    """Bar chart: degree, closeness, S/D ratio by valence and model"""
    valences = ['positive', 'neutral', 'negative']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(valences))
    w = 0.35
    colors = {'base': '#3498DB', 'chat': '#E74C3C'}

    # Panel 1: Degree
    for i, model in enumerate(['base', 'chat']):
        means = [df[(df['model']==model) & (df['valence']==v)]['degree'].mean() for v in valences]
        stds = [df[(df['model']==model) & (df['valence']==v)]['degree'].std() for v in valences]
        axes[0].bar(x + i*w, means, w, yerr=stds, label=model.upper(),
               color=colors[model], alpha=0.8, capsize=3, edgecolor='white')

    # Panel 2: Closeness
    for i, model in enumerate(['base', 'chat']):
        means = [df[(df['model']==model) & (df['valence']==v)]['closeness'].mean() for v in valences]
        stds = [df[(df['model']==model) & (df['valence']==v)]['closeness'].std() for v in valences]
        axes[1].bar(x + i*w, means, w, yerr=stds, label=model.upper(),
               color=colors[model], alpha=0.8, capsize=3, edgecolor='white')

    # Panel 3: S/D ratio
    for i, model in enumerate(['base', 'chat']):
        means_sd = []
        stds_sd = []
        for v in valences:
            sub = df[(df['model']==model) & (df['valence']==v)]
            sd_vals = sub['strength'] / sub['degree'].replace(0, np.nan)
            means_sd.append(sd_vals.mean())
            stds_sd.append(sd_vals.std())
        axes[2].bar(x + i*w, means_sd, w, yerr=stds_sd, label=model.upper(),
               color=colors[model], alpha=0.8, capsize=3, edgecolor='white')

    labels = ['Degree', 'Closeness Centrality', 'Strength / Degree']
    for ax, label in zip(axes, labels):
        ax.set_xticks(x + w/2)
        ax.set_xticklabels([v.capitalize() for v in valences])
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Network Metrics Comparison: Base vs RLHF', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved {output}")

# --- Plot: network_visual.png (appendix) ---

def plot_network_visual(networks, output="results/network_visual.png"):
    """Full network topology — community layout with inter-community bridges in red."""
    from networkx.algorithms.community import greedy_modularity_communities

    fig, axes = plt.subplots(1, 2, figsize=(30, 15))
    cmap = plt.cm.tab20

    for ax, (model, G) in zip(axes, networks.items()):
        # Community detection
        communities = sorted(greedy_modularity_communities(G),
                             key=len, reverse=True)
        node_to_comm = {}
        for i, comm in enumerate(communities):
            for n in comm:
                node_to_comm[n] = i
        colors = [cmap(node_to_comm[n] % 20) for n in G.nodes()]

        # Degree-based node sizing
        degrees = np.array([G.degree(n) for n in G.nodes()], dtype=float)
        sizes = 10 + 90 * (degrees / max(degrees.max(), 1))

        # ---- Community-aware layout ----
        n_comm = len(communities)
        comm_centres = {}
        radius = 10.0
        for i in range(n_comm):
            angle = 2 * np.pi * i / n_comm
            comm_centres[i] = np.array([radius * np.cos(angle),
                                        radius * np.sin(angle)])
        pos = {}
        for i, comm in enumerate(communities):
            sub = G.subgraph(comm)
            if len(comm) == 1:
                local = {list(comm)[0]: np.array([0.0, 0.0])}
            else:
                spread = 1.0 + 1.5 * np.sqrt(len(comm) / len(G.nodes()))
                local = nx.spring_layout(sub, k=0.5, iterations=50,
                                         seed=42 + i, scale=spread)
            for n, p in local.items():
                pos[n] = p + comm_centres[i]

        # Separate intra vs inter-community edges
        intra_edges = [(u, v) for u, v in G.edges()
                       if node_to_comm[u] == node_to_comm[v]]
        inter_edges = [(u, v) for u, v in G.edges()
                       if node_to_comm[u] != node_to_comm[v]]

        # Draw intra-community edges (subtle grey)
        nx.draw_networkx_edges(G, pos, edgelist=intra_edges, ax=ax,
                               edge_color='#cccccc', alpha=0.25, width=0.3)
        # Draw inter-community edges (red bridges)
        nx.draw_networkx_edges(G, pos, edgelist=inter_edges, ax=ax,
                               edge_color='#dd3333', alpha=0.4, width=0.5)
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, ax=ax,
                                node_color=colors, node_size=sizes,
                                alpha=0.9, linewidths=0.3,
                                edgecolors='white')

        # Label top-12 hub nodes
        top_hubs = sorted(G.nodes(), key=lambda n: G.degree(n),
                          reverse=True)[:12]
        nx.draw_networkx_labels(G, pos,
                                labels={n: n for n in top_hubs}, ax=ax,
                                font_size=7, font_weight='bold',
                                font_color='#111111')

        n_n = G.number_of_nodes()
        n_e = G.number_of_edges()
        n_inter = len(inter_edges)
        pct_inter = 100 * n_inter / max(n_e, 1)
        avg_deg = 2 * n_e / n_n
        clust = nx.average_clustering(G)
        ax.set_title(
            f"{model.upper()} Model\n"
            f"{n_n} nodes · {n_e} edges · {n_comm} communities\n"
            f"avg degree {avg_deg:.1f} · clustering {clust:.3f}\n"
            f"inter-community edges: {n_inter} ({pct_inter:.0f}%) "
            f"— shown in red",
            fontsize=14, fontweight='bold')
        ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved {output}")


# --- MAIN ---

if __name__ == "__main__":
    print("=" * 50)
    print("ANALYSIS")
    print("=" * 50)

    with open("results/networks.pkl", "rb") as f:
        networks = pickle.load(f)
    df = pd.read_csv("results/metrics.csv")

    # 1. Statistics (degree)
    print("\n1. DEGREE COMPARISON")
    for valence in ['positive', 'negative', 'neutral']:
        base = df[(df['model']=='base') & (df['valence']==valence)]['degree'].values
        chat = df[(df['model']=='chat') & (df['valence']==valence)]['degree'].values
        _, p = mannwhitneyu(base, chat)
        d = cohens_d(base, chat)
        print(f"  {valence.upper()}: Base={np.mean(base):.1f}, Chat={np.mean(chat):.1f}, p={p:.4f}, d={d:.2f}")

    # 2. Strength & concentration
    print("\n2. STRENGTH & CONCENTRATION")
    for valence in ['positive', 'negative', 'neutral']:
        for model in ['base', 'chat']:
            sub = df[(df['model']==model) & (df['valence']==valence)]
            s = sub['strength'].mean()
            deg = sub['degree'].mean()
            ratio = s / deg if deg > 0 else 0
            print(f"  {model.upper()} {valence}: strength={s:.1f}, degree={deg:.1f}, "
                  f"strength/degree={ratio:.2f}")

    # 3. Emotions
    print("\n3. EMOTIONAL ANALYSIS")
    for model, G in networks.items():
        emo = analyze_emotions(G)
        print(f"  {model.upper()}: joy={emo['joy']:.0f}, trust={emo['trust']:.0f}, "
              f"fear={emo['fear']:.0f}, anger={emo['anger']:.0f}")

    # 4. Plots
    print("\n4. PLOTS")
    with open("data/associations.json") as f:
        assoc = json.load(f)
    plot_network_comparison(networks)
    plot_plutchik_comparison(networks)
    plot_metrics_comparison(df, assoc)
    plot_network_visual(networks)

    print("\nDone!")


