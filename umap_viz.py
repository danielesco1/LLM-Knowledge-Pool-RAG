import json
import numpy as np
import matplotlib.pyplot as plt
import umap
from collections import defaultdict

def plot_embeddings_umap(embeddings_json="knowledge_pool/merged.json", save_path="embeddings_plot.png"):
    """Plot embeddings in 2D using UMAP dimensionality reduction"""
    
    # Load embeddings
    with open(embeddings_json, 'r', encoding='utf8') as f:
        data = json.load(f)
    
    # Extract vectors and source files
    vectors = np.array([item['vector'] for item in data])
    sources = [item.get('source_file', 'unknown') for item in data]
    
    print(f"Loaded {len(vectors)} embeddings with {vectors.shape[1]} dimensions")
    
    # Apply UMAP dimensionality reduction
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding_2d = reducer.fit_transform(vectors)
    
    # Create color map for different source files  
    unique_sources = list(set(sources))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_sources)))
    source_colors = {source: colors[i] for i, source in enumerate(unique_sources)}
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    for source in unique_sources:
        mask = np.array([s == source for s in sources])
        plt.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1], 
                   c=[source_colors[source]], label=source.replace('.json', ''), 
                   alpha=0.6, s=20)
    
    plt.title('UMAP Visualization of Document Embeddings')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print clustering info
    source_counts = defaultdict(int)
    for source in sources:
        source_counts[source] += 1
    
    print("\nDocument distribution:")
    for source, count in sorted(source_counts.items()):
        print(f"{source}: {count} chunks")
    
    return embedding_2d, sources

def analyze_clusters(embedding_2d, sources, data):
    """Analyze what types of content cluster together"""
    
    # Find content that's far from others (outliers)
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=5).fit(embedding_2d)
    distances, indices = nbrs.kneighbors(embedding_2d)
    avg_distances = np.mean(distances, axis=1)
    
    # Get outliers (top 10 most isolated points)
    outlier_indices = np.argsort(avg_distances)[-10:]
    
    print("\nMost isolated content (potential outliers):")
    for i, idx in enumerate(outlier_indices[-5:]):  # Show top 5
        content_preview = data[idx]['content'][:100] + "..."
        print(f"{i+1}. Source: {sources[idx]}")
        print(f"   Content: {content_preview}\n")

# Usage
if __name__ == "__main__":
    # Create UMAP plot
    embedding_2d, sources = plot_embeddings_umap()
    
    # Load data for analysis
    with open("knowledge_pool/merged.json", 'r', encoding='utf8') as f:
        data = json.load(f)
    
    # Analyze clusters
    analyze_clusters(embedding_2d, sources, data)