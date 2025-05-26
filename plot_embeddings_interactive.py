import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import umap
from collections import defaultdict

def plot_embeddings_umap_interactive(embeddings_json="knowledge_pool/merged.json", save_path="embeddings_plot.html"):
    """Interactive UMAP plot with hover content preview"""
    
    # Load embeddings
    with open(embeddings_json, 'r', encoding='utf8') as f:
        data = json.load(f)
    
    vectors = np.array([item['vector'] for item in data])
    sources = [item.get('source_file', 'unknown').replace('.json', '') for item in data]
    contents = [item['content'][:200] + "..." if len(item['content']) > 200 else item['content'] for item in data]
    
    print(f"Loaded {len(vectors)} embeddings with {vectors.shape[1]} dimensions")
    
    # UMAP reduction - optimized for text embeddings
    reducer = umap.UMAP(
        n_neighbors=30,      # Higher for better global structure
        min_dist=0.05,       # Tighter clusters 
        n_components=2,
        metric='cosine',     # Better for high-dim embeddings
        random_state=42
    )
    embedding_2d = reducer.fit_transform(vectors)
    
    # Create interactive plot
    fig = px.scatter(
        x=embedding_2d[:, 0], y=embedding_2d[:, 1],
        color=sources,
        hover_data={'Content': contents},
        title='Interactive UMAP Visualization of Document Embeddings',
        labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2', 'color': 'Document'},
        width=1000, height=700,
        color_discrete_sequence=px.colors.qualitative.Set1  # Better visibility
    )
    
    # Customize hover template
    fig.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>" +
                      "X: %{x:.2f}<br>" +
                      "Y: %{y:.2f}<br>" +
                      "<b>Content:</b> %{customdata[0]}<br>" +
                      "<extra></extra>",
        hovertext=sources,
        customdata=[[content] for content in contents]
    )
    
    fig.update_layout(
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01)
    )
    
    # Save and show
    fig.write_html(save_path)
    
    # Force browser opening
    import webbrowser
    import os
    webbrowser.open('file://' + os.path.realpath(save_path))
    
    print(f"Opening {save_path} in browser...")
    
    # Print stats
    source_counts = defaultdict(int)
    for source in sources:
        source_counts[source] += 1
    
    print(f"\nSaved interactive plot to: {save_path}")
    print("Document distribution:")
    for source, count in sorted(source_counts.items()):
        print(f"{source}: {count} chunks")
    
    return embedding_2d, sources, data

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
    # Create interactive UMAP plot
    embedding_2d, sources, data = plot_embeddings_umap_interactive()
    
    # Analyze clusters
    analyze_clusters(embedding_2d, sources, data)