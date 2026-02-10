# 02_build_network.py - Build TFMN from Word Associations
import json, pickle
import networkx as nx

def build_association_network(data):
    """Build graph from cue -> associations"""
    G = nx.Graph()
    for key, samples in data.items():
        for sample in samples:
            cue = sample["cue"]
            for word in sample["words"]:
                if cue != word:
                    if G.has_edge(cue, word):
                        G[cue][word]["weight"] += 1
                    else:
                        G.add_edge(cue, word, weight=1)
    return G

if __name__ == "__main__":
    with open("data/associations.json") as f:
        data = json.load(f)
    
    networks = {}
    for model in ["base", "chat"]:
        G = build_association_network(data[model])
        networks[model] = G
        print(f"{model}: nodes={G.number_of_nodes()} edges={G.number_of_edges()}")
    
    with open("results/networks.pkl", "wb") as f:
        pickle.dump(networks, f)
    print("Saved results/networks.pkl")
