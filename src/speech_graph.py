#!/usr/bin/python
# -*- coding: utf8 -*-

# This code has been adapted from https://github.com/guillermodoghel/speechgraph/blob/master/speechgraph/speechgraph.py
# None of that belongs to me, I just added some comments and made it more readable and up to date

# I've modified the codebase to add more visualizations and statistics, everything else is the same. 

from collections import Counter
import re
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import nltk
from nltk.stem.snowball import SnowballStemmer

class GraphStatistics:
    def __init__(self, graph):
        self.graph = graph
    
    def statistics(self):
        """Calculate various graph statistics"""
        res = {}
        graph = self.graph
        
        # Basic graph properties
        res['number_of_nodes'] = graph.number_of_nodes()
        res['number_of_edges'] = graph.number_of_edges()
        
        # Parallel edges (PE)
        edge_counts = Counter(graph.edges())
        res['PE'] = sum(1 for count in edge_counts.values() if count > 1)
        
        # Connected components
        res['LCC'] = nx.number_weakly_connected_components(graph)
        res['LSC'] = nx.number_strongly_connected_components(graph)
        
        # Degree statistics
        degrees = [d for _, d in graph.degree()]
        res['degree_average'] = np.mean(degrees) if degrees else 0
        res['degree_std'] = np.std(degrees) if degrees else 0
        
        # Loop statistics (L1, L2, L3)
        adj_matrix = nx.adjacency_matrix(graph).toarray()
        adj_matrix2 = np.dot(adj_matrix, adj_matrix)
        adj_matrix3 = np.dot(adj_matrix2, adj_matrix)
        
        res['L1'] = np.trace(adj_matrix)
        res['L2'] = np.trace(adj_matrix2)
        res['L3'] = np.trace(adj_matrix3)
        
        return res

class NaiveGraph:
    def __init__(self):
        pass
    
    def text2graph(self, text, word_tokenizer=None):
        """Convert text to a graph where nodes are words and edges represent adjacency"""
        if word_tokenizer is None:
            word_tokenizer = lambda x: x.split()
            
        # Clean and tokenize text
        cleaned_text = re.sub(r'[^\w ]+', ' ', text.lower().strip())
        words = [w for w in word_tokenizer(cleaned_text) if len(w) > 0]
        
        # Create directed multigraph
        gr = nx.MultiDiGraph()
        gr.add_edges_from(zip(words[:-1], words[1:]))
        
        return gr
    
    def analyze_text(self, text, word_tokenizer=None):
        """Analyze text and return graph statistics"""
        dgr = self.text2graph(text, word_tokenizer)
        return GraphStatistics(dgr).statistics(), dgr
    
    def visualize_graph(self, graph, title="Word Graph", max_nodes=50, layout='kamada_kawai'):
        """
        Visualize the graph with improved layout and readability
        
        Parameters:
        -----------
        graph : networkx.MultiDiGraph
            The graph to visualize
        title : str
            Title of the graph
        max_nodes : int
            Maximum number of nodes to display
        layout : str
            Layout algorithm to use ('spring', 'kamada_kawai', 'circular', 'shell')
            
        Returns:
        --------
        plt : matplotlib.pyplot
            The plot object
        """
        # If graph is too large, take a subgraph of the most connected nodes
        if graph.number_of_nodes() > max_nodes:
            # Get nodes with highest degree
            degrees = dict(graph.degree())
            top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
            graph = graph.subgraph(top_nodes)
        
        plt.figure(figsize=(14, 12))
        
        # Choose layout based on parameter
        if layout == 'spring':
            pos = nx.spring_layout(graph, seed=42, k=0.5)  # Increased k for more spacing
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(graph)  # Often better for word graphs
        elif layout == 'circular':
            pos = nx.circular_layout(graph)
        elif layout == 'shell':
            pos = nx.shell_layout(graph)
        else:
            pos = nx.kamada_kawai_layout(graph)  # Default to kamada_kawai
        
        # Get edge weights for thickness
        edge_counts = Counter(graph.edges())
        max_edge_count = max(edge_counts.values()) if edge_counts else 1
        
        # Normalize edge weights for better visualization
        edge_weights = [0.5 + (edge_counts[edge] / max_edge_count) * 2 for edge in graph.edges()]
        
        # Calculate node sizes based on degree centrality
        centrality = nx.degree_centrality(graph)
        node_sizes = [300 + (centrality[node] * 2000) for node in graph.nodes()]
        
        # Draw edges with alpha transparency for better visibility
        nx.draw_networkx_edges(
            graph, 
            pos=pos,
            width=edge_weights,
            edge_color='gray',
            alpha=0.6,
            arrows=True,
            arrowsize=15,
            connectionstyle='arc3,rad=0.15'  # More curved edges
        )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            graph,
            pos=pos,
            node_size=node_sizes,
            node_color='skyblue',
            alpha=0.8,
            edgecolors='white'
        )
        
        # Draw labels with smaller font and slight offset for better readability
        label_pos = {node: (coords[0], coords[1] + 0.02) for node, coords in pos.items()}
        nx.draw_networkx_labels(
            graph,
            pos=label_pos,
            font_size=8,  # Smaller font
            font_family='sans-serif',
            font_weight='bold',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
        )
        
        plt.title(title, fontsize=18)
        plt.axis('off')
        plt.tight_layout()
        
        return plt

class StemGraph(NaiveGraph):
    def __init__(self, language='english'):
        super().__init__()
        self.stemmer = SnowballStemmer(language).stem
    
    def text2graph(self, text, word_tokenizer=None):
        """Convert text to a graph where nodes are stemmed words"""
        if word_tokenizer is None:
            word_tokenizer = lambda x: x.split()
            
        # Clean and tokenize text
        cleaned_text = re.sub(r'[^\w ]+', ' ', text.lower().strip())
        words = [w for w in word_tokenizer(cleaned_text) if len(w) > 0]
        
        # Stem words
        stemmed_words = [self.stemmer(w) for w in words]
        
        # Create directed multigraph
        gr = nx.MultiDiGraph()
        gr.add_edges_from(zip(stemmed_words[:-1], stemmed_words[1:]))
        
        return gr

class PosGraph(NaiveGraph):
    def __init__(self):
        super().__init__()
        # Make sure NLTK resources are available
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
    
    def text2graph(self, text, word_tokenizer=None, sentence_tokenizer=None):
        """Convert text to a graph where nodes are POS tags"""
        if word_tokenizer is None:
            word_tokenizer = nltk.word_tokenize
        if sentence_tokenizer is None:
            sentence_tokenizer = lambda x: x.split('.')
            
        # Process sentences
        sentences = sentence_tokenizer(text)
        tags = []
        
        for s in sentences:
            cleaned_text = re.sub(r'[^\w ]+', ' ', s.lower().strip())
            if cleaned_text.strip():  # Skip empty sentences
                words = word_tokenizer(cleaned_text)
                if words:
                    pos_tags = nltk.pos_tag(words)
                    tags.extend([tag for _, tag in pos_tags])
        
        # Create directed multigraph
        gr = nx.MultiDiGraph()
        if len(tags) > 1:
            gr.add_edges_from(zip(tags[:-1], tags[1:]))
        
        return gr 