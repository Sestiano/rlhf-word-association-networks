# 05_fmn_multiplex.py - FMN Ego-Networks
# Generates: fmn_ego_neutral.png, fmn_ego_valenced.png
import os, pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import networkx as nx
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

try:
    from community import community_louvain
except ImportError:
    community_louvain = None

from emoatlas import EmoScores

# --- Palette (EmoAtlas-compliant) ---

COLZ = {
    'positive':     (26/256, 133/256, 255/256),
    'negative':     (255/256, 25/256, 25/256),
    'neutral':      (0.012, 0.012, 0.012),
    'semipositive': (117/256, 152/256, 191/256),
    'seminegative': (200/256, 50/256, 50/256),
    'mixed':        (0.5, 0.0, 0.5),
    'neutral_edge': (0.82, 0.82, 0.82),
}

POSITIVE_EMOTIONS = {'joy', 'trust', 'anticipation'}
NEGATIVE_EMOTIONS = {'anger', 'fear', 'sadness', 'disgust'}
# Words chosen to highlight RLHF effects:
# - anchor: neutral→NEG(base) vs neutral→POS(chat) — positivity bias
_es_cache = {}

def get_word_valence(word):
    """Emotional valence (cached)"""
    if word in _es_cache:
        return _es_cache[word]
    try:
        es = EmoScores()
        scores = es.zscores(word)
        pos = sum(scores.get(e, 0) for e in POSITIVE_EMOTIONS)
        neg = sum(scores.get(e, 0) for e in NEGATIVE_EMOTIONS)
        val = 'positive' if pos > neg else ('negative' if neg > pos else 'neutral')
        _es_cache[word] = val
        return val
    except:
        _es_cache[word] = 'neutral'
        return 'neutral'

def _node_color(word):
    v = get_word_valence(word)
    return COLZ.get(v, COLZ['neutral'])

def _edge_color_and_width(w1, w2, base_lw=1.0):
    """Edge color and width according to FMN conventions"""
    v1, v2 = get_word_valence(w1), get_word_valence(w2)

    if v1 == 'positive' and v2 == 'positive':
        return COLZ['positive'], 3 * base_lw, 1
    if v1 == 'negative' and v2 == 'negative':
        return COLZ['negative'], 3 * base_lw, 1
    if (v1 == 'positive' and v2 == 'negative') or (v1 == 'negative' and v2 == 'positive'):
        return COLZ['mixed'], 3 * base_lw, 1
    if v1 == 'positive' or v2 == 'positive':
        return COLZ['semipositive'], 2 * base_lw, -1
    if v1 == 'negative' or v2 == 'negative':
        return COLZ['seminegative'], 2 * base_lw, -1
    return COLZ['neutral_edge'], base_lw, -1

# --- Bezier edge bundling ---

def _bezier_curve(p0, p1, control, n_points=30):
    t = np.linspace(0, 1, n_points)
    curve = np.outer((1 - t)**2, p0) + np.outer(2 * (1 - t) * t, control) + np.outer(t**2, p1)
    return curve

def _get_community_positions(G, radius=1.0):
    """Circular layout grouped by community (Louvain)"""
    if community_louvain is not None and len(G.nodes()) > 2:
        try:
            partition = community_louvain.best_partition(G)
        except:
            partition = {n: 0 for n in G.nodes()}
    else:
        partition = {n: 0 for n in G.nodes()}

    communities = defaultdict(list)
    for node, comm in partition.items():
        communities[comm].append(node)

    sorted_comms = sorted(communities.keys(), key=lambda c: -len(communities[c]))

    pos = {}
    n_nodes = len(G.nodes())
    if n_nodes == 0:
        return pos, partition, {}

    gap_angle = 2 * np.pi * 0.02
    n_comms = len(sorted_comms)
    total_gap = gap_angle * n_comms
    available_angle = 2 * np.pi - total_gap

    current_angle = 0
    comm_centers = {}

    for comm_id in sorted_comms:
        nodes = communities[comm_id]
        nodes_sorted = sorted(nodes, key=lambda n: G.degree(n), reverse=True)
        comm_angle = available_angle * (len(nodes) / n_nodes)
        angles = np.linspace(current_angle, current_angle + comm_angle, len(nodes), endpoint=False)
        center_angle = current_angle + comm_angle / 2
        comm_centers[comm_id] = np.array([radius * 0.3 * np.cos(center_angle),
                                          radius * 0.3 * np.sin(center_angle)])

        for node, angle in zip(nodes_sorted, angles):
            pos[node] = np.array([radius * np.cos(angle), radius * np.sin(angle)])

        current_angle += comm_angle + gap_angle

    return pos, partition, comm_centers

# --- FMN Ego-Network ---

def plot_ego_fmn(G, center_word, ax, depth=1):
    """FMN ego-network: circular layout, edge bundling, valence"""
    if center_word not in G:
        ax.text(0.5, 0.5, f"'{center_word}' not found\nin the network",
                ha='center', va='center', fontsize=11, transform=ax.transAxes,
                style='italic', color='gray')
        ax.set_title(f"'{center_word}'", fontsize=12, fontweight='bold')
        ax.axis('off')
        return

    ego = nx.ego_graph(G, center_word, radius=depth)

    # Pre-calculate valences
    for n in ego.nodes():
        get_word_valence(n)

    # Layout circolare con edge bundling
    pos, partition, comm_centers = _get_community_positions(ego, radius=1.0)
    pos[center_word] = np.array([0.0, 0.0])

    weights = [ego[u][v].get('weight', 1) for u, v in ego.edges()]
    max_w = max(weights) if weights else 1
    base_lw = 15 / (len(ego.nodes()) ** 0.6)

    # ---- Draw edges with Bezier curves ----
    edges_bg = []
    edges_fg = []

    for u, v in ego.edges():
        w = ego[u][v].get('weight', 1)
        color, width_mult, zorder = _edge_color_and_width(u, v, base_lw)
        lw = width_mult * (0.3 + 0.7 * w / max_w)

        p0 = pos[u]
        p1 = pos[v]
        control = np.array([0.0, 0.0])

        if u == center_word or v == center_word:
            curve_x = [p0[0], p1[0]]
            curve_y = [p0[1], p1[1]]
        else:
            curve = _bezier_curve(p0, p1, control)
            curve_x = curve[:, 0]
            curve_y = curve[:, 1]

        entry = (curve_x, curve_y, color, lw, zorder)
        if zorder < 0:
            edges_bg.append(entry)
        else:
            edges_fg.append(entry)

    for cx, cy, color, lw, zo in edges_bg:
        ax.plot(cx, cy, color=color, linewidth=lw, alpha=0.3, zorder=0, solid_capstyle='round')
    for cx, cy, color, lw, zo in edges_fg:
        ax.plot(cx, cy, color=color, linewidth=lw, alpha=0.6, zorder=1, solid_capstyle='round')

    # ---- Draw node labels ----
    N = len(ego.nodes())
    for node in ego.nodes():
        x, y = pos[node]
        color = _node_color(node)

        if node == center_word:
            fontsize = max(14, 18 / (N ** 0.3))
            ax.text(x, y, node.upper(), ha='center', va='center',
                    fontsize=fontsize, fontweight='bold', color=color,
                    path_effects=[pe.withStroke(linewidth=3, foreground='white')],
                    zorder=10)
        else:
            deg = ego.degree(node)
            fontsize = max(6, 16 / (N ** 0.5) * 1.5 + deg * 0.3)
            angle_rad = np.arctan2(y, x)
            angle_deg = np.degrees(angle_rad)
            ha = 'left' if -90 <= angle_deg <= 90 else 'right'
            rotation = angle_deg if -90 <= angle_deg <= 90 else angle_deg + 180

            ax.text(x * 1.08, y * 1.08, node, ha=ha, va='center',
                    fontsize=fontsize, fontweight='bold', color=color,
                    rotation=rotation, rotation_mode='anchor',
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')],
                    zorder=5)

    # Dots per i nodi
    for node in ego.nodes():
        x, y = pos[node]
        color = _node_color(node)
        size = 60 if node == center_word else 15
        ax.scatter(x, y, s=size, c=[color], edgecolors='white', linewidths=0.5, zorder=4)

    # Valence aura
    neighbors = list(ego.neighbors(center_word))
    n_pos = sum(1 for n in neighbors if get_word_valence(n) == 'positive')
    n_neg = sum(1 for n in neighbors if get_word_valence(n) == 'negative')
    aura = 'positive' if n_pos > n_neg else ('negative' if n_neg > n_pos else 'neutral')
    aura_color = COLZ.get(aura, (0.5, 0.5, 0.5))

    ax.set_title(f"'{center_word}'\n({ego.number_of_nodes()} nodes, aura: {aura})",
                 fontsize=11, fontweight='bold', color=aura_color)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')


def _ego_figure(networks, words, title, output_path):
    """Helper: generate a 2×N ego-network figure (BASE top, CHAT bottom)."""
    n_words = len(words)
    fig, axes = plt.subplots(2, n_words, figsize=(7 * n_words, 14), facecolor='white')
    if n_words == 1:
        axes = axes.reshape(2, 1)

    for i, word in enumerate(words):
        plot_ego_fmn(networks['base'], word, axes[0, i])
        plot_ego_fmn(networks['chat'], word, axes[1, i])

    axes[0, 0].annotate('BASE', xy=(-0.25, 0.5), xycoords='axes fraction',
                        fontsize=16, fontweight='bold', rotation=90, va='center',
                        color='#333333')
    axes[1, 0].annotate('CHAT\n(RLHF)', xy=(-0.25, 0.5), xycoords='axes fraction',
                        fontsize=16, fontweight='bold', rotation=90, va='center',
                        color='#333333')

    legend_elements = [
        mpatches.Patch(facecolor=COLZ['positive'], label='Positive node'),
        mpatches.Patch(facecolor=COLZ['negative'], label='Negative node'),
        mpatches.Patch(facecolor=COLZ['neutral'], label='Neutral node'),
        plt.Line2D([0], [0], color=COLZ['positive'], lw=3, label='Pos-Pos edge'),
        plt.Line2D([0], [0], color=COLZ['negative'], lw=3, label='Neg-Neg edge'),
        plt.Line2D([0], [0], color=COLZ['mixed'], lw=3, label='Pos-Neg edge'),
        plt.Line2D([0], [0], color=COLZ['semipositive'], lw=2, label='Pos-Neutral'),
        plt.Line2D([0], [0], color=COLZ['seminegative'], lw=2, label='Neg-Neutral'),
        plt.Line2D([0], [0], color=COLZ['neutral_edge'], lw=1, label='Neutral-Neutral'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=9,
               frameon=True, fancybox=True, shadow=False,
               edgecolor='lightgray', facecolor='white')

    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0.03, 0.06, 1, 0.95])
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved {output_path}")


def plot_ego_neutral(networks, output_path="results/fmn_ego_neutral.png"):
    """Ego-networks for neutral cue words: anchor, mirror (positivity bias)."""
    _ego_figure(networks, ['anchor', 'mirror'],
                'FMN Ego-Networks: Neutral Words — Positivity Bias',
                output_path)


def plot_ego_valenced(networks, output_path="results/fmn_ego_valenced.png"):
    """Ego-networks for valenced cue words: treasure, ruin (semantic impoverishment)."""
    _ego_figure(networks, ['treasure', 'ruin'],
                'FMN Ego-Networks: Valenced Words — Semantic Impoverishment',
                output_path)






# --- MAIN ---

if __name__ == "__main__":
    print("=" * 50)
    print("FMN + MULTIPLEX")
    print("=" * 50)

    with open(os.path.join(PROJECT_ROOT, "results/networks.pkl"), "rb") as f:
        networks = pickle.load(f)

    # 1. FMN Ego-networks
    print("\n1. FMN EGO-NETWORKS")
    print("-" * 40)
    plot_ego_neutral(networks,
                     output_path=os.path.join(PROJECT_ROOT, "results/fmn_ego_neutral.png"))
    plot_ego_valenced(networks,
                      output_path=os.path.join(PROJECT_ROOT, "results/fmn_ego_valenced.png"))

    print("\nDone!")
