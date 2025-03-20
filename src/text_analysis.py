import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import networkx as nx
from collections import defaultdict
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import imageio
from tqdm import tqdm
import tempfile
from io import BytesIO
import imageio.v2 as imageio

print("All imports successful!")
# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('spanish')

def preprocess_text(text):
    """Clean and tokenize Spanish text"""
    # First standardize Chilean slang
    text = standardize_chilean_slang(text)
    # Convert to lowercase
    text = text.lower()
    # Remove special characters but keep Spanish characters
    text = re.sub(r'[^a-záéíóúñü\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text, language='spanish')
    # Remove Spanish stopwords
    # Get Spanish stopwords and add custom ones
    custom_stopwords = {
        "bueno", "voy", "ir", "dijo", "ahí", "vez", "oye", "así","gusta","dije","encima","creo","ser",
        "puede", "hacer", "tan", "solo", "aún", "aquí", "mismo", "acuerdo", "acá", "cómo",
        "entonces", "quizá", "casi", "todavía", "dentro", "fuera", 
        "luego", "verdad", "seguro", "hacia", "cuanto", "alguna", "menos",
        "aunque", "mientras", "pronto", "claro", "gran", "pues", "cuánto", "cuán", "cuánta", "cuántos", "cuántas",
        "ahora", "siempre", "tanto", "tanta", "tanto", "tantas", "tanto", "tantas", "tanto", "tantas", "tanto", "tantas",
    }

    spanish_stopwords = set(stopwords.words('spanish')) | custom_stopwords

    tokens = [word for word in tokens if word not in spanish_stopwords and len(word) > 2]
    return tokens

# 1. Basic Text Statistics
def basic_stats(text):
    tokens = preprocess_text(text)
    word_count = len(tokens)
    unique_words = len(set(tokens))
    return {
        'Total de Palabras': word_count,
        'Palabras Únicas': unique_words,
        'Diversidad Léxica': unique_words / word_count
    }

# 2. Word Frequency Analysis
def word_frequency(tokens, top_n=20, remove_stopwords=True):
    """Analyze word frequency with option to remove stopwords"""
    if remove_stopwords:
        # Get Spanish stopwords
        spanish_stops = set(stopwords.words('spanish'))
        # Filter out stopwords
        filtered_tokens = [word for word in tokens if word.lower() not in spanish_stops]
    else:
        filtered_tokens = tokens
    
    word_freq = Counter(filtered_tokens)
    return pd.DataFrame(word_freq.most_common(top_n), 
                       columns=['Palabra', 'Frecuencia'])

# 3. Zipf's Law Plot
def plot_zipf(tokens):
    word_freq = Counter(tokens)
    freq_df = pd.DataFrame(word_freq.most_common(), columns=['Palabra', 'Frecuencia'])
    freq_df['Rango'] = range(1, len(freq_df) + 1)
    
    plt.figure(figsize=(12, 6))
    plt.loglog(freq_df['Rango'], freq_df['Frecuencia'], 'b.')
    plt.title("Ley de Zipf")
    plt.xlabel('Rango (log)')
    plt.ylabel('Frecuencia (log)')
    plt.grid(True)
    return plt

# 4. Word Cloud
def generate_wordcloud(text):
    """Generate word cloud with Spanish support"""
    wordcloud = WordCloud(width=800, 
                         height=400,
                         background_color='white',
                         # Add Spanish characters support
                         regexp=r"[a-záéíóúñü]+[']*[a-z]+").generate(text)
    
    plt.figure(figsize=(15,8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    return plt

# 5. Word Length Distribution
def word_length_dist(tokens):
    word_lengths = [len(word) for word in tokens]
    plt.figure(figsize=(10, 6))
    sns.histplot(word_lengths, bins=20)
    plt.title('Distribución de Longitud de Palabras')
    plt.xlabel('Longitud de Palabra')
    plt.ylabel('Frecuencia')
    return plt

# 6. Chilean Slang Standardization
def standardize_chilean_slang(text):
    """Replace all variations of Chilean slang with their standard form"""
    # Dictionary of slang variations
    slang_dict = {
        'weón': ['weón', 'weon', 'huevón', 'huevon'],
        'weones': ['weones', 'weones', 'huevones', 'huevones'],
        'weá': ['wea', 'weá', 'huevá', 'huevada','hue'],
        'pene': ['pene', 'pico', 'pichula', 'corneta', 'callampa','tula']
    }
    
    # Convert text to lowercase for better matching
    text_lower = text.lower()
    
    # Replace each variation with its standard form
    for standard, variations in slang_dict.items():
        for variant in variations:
            # Using word boundaries (\b) to match whole words only
            text_lower = re.sub(fr'\b{variant}\b', standard, text_lower)
    
    return text_lower

# 7. Count Specific Words
def count_specific_words(text, words_to_count):
    """
    Count occurrences of specific words in the text
    
    Parameters:
    -----------
    text : str
        The text to analyze
    words_to_count : list
        List of words to count occurrences of
        
    Returns:
    --------
    dict
        Dictionary with words as keys and their counts as values
    """
    # Preprocess the text to get tokens
    tokens = preprocess_text(text)
    
    # Convert all words to lowercase for case-insensitive matching
    tokens_lower = [token.lower() for token in tokens]
    words_to_count_lower = [word.lower() for word in words_to_count]
    
    # Count occurrences
    word_counts = {}
    for word in words_to_count_lower:
        word_counts[word] = tokens_lower.count(word)
    
    # Sort by frequency (descending)
    sorted_counts = {k: v for k, v in sorted(word_counts.items(), 
                                            key=lambda item: item[1], 
                                            reverse=True)}
    
    return sorted_counts

# 8. Word Ego Network
def create_word_ego_network(text, target_word, vicinity_size=5, top_connections=20):
    """
    Create and visualize an ego-network for a specific word with both counting methods
    
    Parameters:
    -----------
    text : str
        The text to analyze
    target_word : str
        The central word for the ego-network
    vicinity_size : int
        Number of words to consider before and after each occurrence of the target word
    top_connections : int
        Number of top connections to display in the network
        
    Returns:
    --------
    G : networkx.Graph
        The ego-network graph
    """
    # Preprocess the text
    tokens = preprocess_text(text)
    tokens_lower = [t.lower() for t in tokens]
    
    # Get total frequency of each word in the entire text
    word_frequencies = Counter(tokens_lower)
    
    # Standardize the target word
    target_word = target_word.lower()
    
    # Find all occurrences of the target word
    target_indices = [i for i, word in enumerate(tokens_lower) if word == target_word]
    
    if not target_indices:
        print(f"La palabra '{target_word}' no se encontró en el texto.")
        return None
    
    # Create dictionaries to store both counting methods
    co_occurrences = defaultdict(int)  # Counts unique occurrences per vicinity
    raw_occurrences = defaultdict(int)  # Counts every occurrence (allows double-counting)
    
    # For each occurrence of the target word
    for idx in target_indices:
        # Define the vicinity range (handle text boundaries)
        start = max(0, idx - vicinity_size)
        end = min(len(tokens), idx + vicinity_size + 1)
        
        # Get unique words in this vicinity (for co-occurrence)
        current_vicinity = set(tokens_lower[start:idx] + tokens_lower[idx+1:end])
        
        # Count each word once per vicinity
        for word in current_vicinity:
            co_occurrences[word] += 1
        
        # Count every occurrence (with potential double-counting)
        for i in range(start, end):
            if i != idx:  # Skip the target word itself
                raw_occurrences[tokens_lower[i]] += 1
    
    # Create a graph
    G = nx.Graph()
    
    # Add the target word as the central node with its total frequency
    target_freq = word_frequencies[target_word]
    G.add_node(target_word, 
              size=1500, 
              color='#FF6666',
              frequency=target_freq)
    
    # Get the top co-occurring words
    top_words = sorted(co_occurrences.items(), key=lambda x: x[1], reverse=True)[:top_connections]
    
    # Calculate the maximum co-occurrence for scaling
    max_count = max([count for _, count in top_words]) if top_words else 1
    
    # Add nodes and edges for top co-occurring words
    for word, co_occur_count in top_words:
        # Get total frequency of this word in the entire text
        total_freq = word_frequencies[word]
        
        # Get raw occurrence count (with double-counting)
        raw_count = raw_occurrences[word]
        
        # Calculate percentage of word's occurrences that are near target word
        vicinity_percentage = (co_occur_count / total_freq) * 100 if total_freq > 0 else 0
        
        # Scale node size based on co-occurrence frequency
        node_size = 200 + (co_occur_count / max_count) * 600
        
        G.add_node(word, 
                  size=node_size, 
                  color='#6699CC',
                  total_frequency=total_freq,
                  vicinity_frequency=co_occur_count,
                  raw_frequency=raw_count,
                  vicinity_percentage=vicinity_percentage)
        
        # Edge weight based on co-occurrence count
        G.add_edge(target_word, word, 
                  weight=co_occur_count,
                  co_occurrence=co_occur_count,
                  raw_occurrence=raw_count)
    
    # Visualize the network
    plt.figure(figsize=(12, 12))
    
    # Position nodes using a spring layout with more space
    pos = nx.spring_layout(G, seed=42, k=0.3)
    
    # Get node sizes and colors
    node_sizes = [G.nodes[node]['size'] for node in G.nodes]
    node_colors = [G.nodes[node]['color'] for node in G.nodes]
    
    # Get edge weights for width
    edge_weights = [G[u][v]['weight'] for u, v in G.edges]
    max_edge_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [2 + (w / max_edge_weight) * 8 for w in edge_weights]
    
    # Draw the network
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, edge_color='#999999')
    
    # Create node labels with word and vicinity frequency
    node_labels = {}
    for node in G.nodes:
        if node == target_word:
            node_labels[node] = f"{node}\n(freq: {G.nodes[node]['frequency']})"
        else:
            node_labels[node] = f"{node}\n(vec: {G.nodes[node]['vicinity_frequency']})"
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, 
                           labels=node_labels,
                           font_size=14, 
                           font_family='sans-serif', 
                           font_weight='bold',
                           verticalalignment='center')
    
    # Add edge labels with raw occurrence counts (allowing double-counting)
    edge_labels = {(target_word, word): f"oc:{raw_occurrences[word]}" 
                  for word, _ in top_words}
    
    nx.draw_networkx_edge_labels(
        G, 
        pos, 
        edge_labels=edge_labels, 
        font_size=8,  # Smaller font size
        font_color='#444444',
        font_weight='bold',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2)
    )
    
    plt.title(f"Red Ego para '{target_word}' (Vecindad: {vicinity_size} palabras)", fontsize=18)
    plt.axis('off')
    plt.tight_layout()
    
    return G

# 9. Sentiment Analysis
def analyze_sentiment(text, sentiment_lexicon=None):
    """
    Analyze sentiment in Spanish text using a lexicon-based approach
    
    Parameters:
    -----------
    text : str
        The text to analyze
    sentiment_lexicon : dict, optional
        Dictionary mapping words to sentiment scores. If None, a basic Spanish
        sentiment lexicon will be used.
        
    Returns:
    --------
    dict
        Dictionary with sentiment analysis results
    """
    # Download Spanish sentiment lexicon if needed
    if not sentiment_lexicon:
        try:
            nltk.download('vader_lexicon')
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            # This is a basic approach - VADER is for English but can be a starting point
            sia = SentimentIntensityAnalyzer()
            
            # Basic Spanish sentiment lexicon (can be expanded)
            sentiment_lexicon = {
                # Positive words
                'bueno': 1.0, 'excelente': 1.8, 'feliz': 1.5, 'alegre': 1.3, 
                'maravilloso': 1.9, 'genial': 1.7, 'fantástico': 1.8, 'increíble': 1.6,
                'agradable': 0.8, 'positivo': 0.7, 'bonito': 0.6, 'hermoso': 1.0,
                
                # Negative words
                'malo': -1.0, 'terrible': -1.8, 'triste': -1.5, 'enojado': -1.3,
                'horrible': -1.9, 'pésimo': -1.7, 'desagradable': -1.4, 'negativo': -0.7,
                'feo': -0.6, 'peor': -1.2, 'difícil': -0.5, 'complicado': -0.4,
                
                # Chilean slang
                'weón': -0.2,  # Context dependent, slightly negative default
                'weá': -0.3,   # Context dependent, slightly negative default
            }
        except:
            print("Error loading sentiment lexicon. Using empty lexicon.")
            sentiment_lexicon = {}
    
    # Preprocess the text
    tokens = preprocess_text(text)
    
    # Calculate sentiment scores
    positive_score = 0
    negative_score = 0
    neutral_count = 0
    
    for token in tokens:
        token = token.lower()
        if token in sentiment_lexicon:
            score = sentiment_lexicon[token]
            if score > 0:
                positive_score += score
            elif score < 0:
                negative_score += score
            else:
                neutral_count += 1
        else:
            neutral_count += 1
    
    # Calculate compound score (normalized between -1 and 1)
    total_words = len(tokens)
    if total_words > 0:
        compound_score = (positive_score + negative_score) / total_words
    else:
        compound_score = 0
    
    # Determine overall sentiment
    if compound_score >= 0.05:
        sentiment = "Positivo"
    elif compound_score <= -0.05:
        sentiment = "Negativo"
    else:
        sentiment = "Neutral"
    
    return {
        'sentiment': sentiment,
        'compound_score': compound_score,
        'positive_score': positive_score,
        'negative_score': negative_score,
        'neutral_count': neutral_count,
        'analyzed_words': total_words
    }

def plot_sentiment_distribution(text, chunk_size=200):
    """
    Plot sentiment distribution across text chunks
    
    Parameters:
    -----------
    text : str
        The text to analyze
    chunk_size : int
        Number of words per chunk for analysis
        
    Returns:
    --------
    plt
        Matplotlib plot object
    """
    # Tokenize the text
    tokens = word_tokenize(text, language='spanish')
    
    # Create chunks of text
    chunks = []
    current_chunk = []
    
    for token in tokens:
        current_chunk.append(token)
        if len(current_chunk) >= chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    # Analyze sentiment for each chunk
    sentiment_scores = []
    for i, chunk in enumerate(chunks):
        sentiment = analyze_sentiment(chunk)
        sentiment_scores.append({
            'chunk': i+1,
            'score': sentiment['compound_score']
        })
    
    # Create DataFrame
    df = pd.DataFrame(sentiment_scores)
    
    # Plot sentiment distribution
    plt.figure(figsize=(12, 6))
    plt.plot(df['chunk'], df['score'], marker='o', linestyle='-', color='#3366CC')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    
    # Add colored background based on sentiment
    plt.fill_between(df['chunk'], df['score'], 0, 
                    where=(df['score'] >= 0), 
                    color='green', alpha=0.1)
    plt.fill_between(df['chunk'], df['score'], 0, 
                    where=(df['score'] < 0), 
                    color='red', alpha=0.1)
    
    plt.title('Distribución de Sentimiento a lo largo del Texto')
    plt.xlabel('Segmento de Texto')
    plt.ylabel('Puntuación de Sentimiento')
    plt.grid(True, alpha=0.3)
    
    return plt 

# 10. Evolving Word Network
from collections import defaultdict, Counter
import imageio.v2 as imageio
from tqdm import tqdm
import os
import tempfile

def create_word_evolution_gif(text, target_word, chunk_size=1000, overlap=500, 
                             vicinity_size=5, top_connections=10, 
                             output_gif="word_evolution.gif", fps=0.5):
    """
    Create a GIF showing how a word's context network evolves throughout a text
    
    Parameters:
    -----------
    text : str
        The full text to analyze
    target_word : str
        The central word to track
    chunk_size : int
        Size of each text chunk in words
    overlap : int
        Number of words to overlap between chunks
    vicinity_size : int
        Number of words to consider before and after each occurrence of the target word
    top_connections : int
        Number of top connections to display in each network
    output_gif : str
        Filename for the output GIF
    fps : float
        Frames per second for the GIF
    """
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    
    # Preprocess the text
    tokens = preprocess_text(text)
    tokens_lower = [t.lower() for t in tokens]
    
    # Standardize the target word
    target_word = target_word.lower()
    
    # Create chunks with overlap
    chunks = []
    chunk_positions = []
    
    i = 0
    while i < len(tokens):
        end = min(i + chunk_size, len(tokens))
        chunks.append(tokens[i:end])
        chunk_positions.append(i)
        i += chunk_size - overlap
    
    # Store frames in memory
    frames = []
    
    # Process each chunk and create a frame
    for chunk_idx, (chunk, position) in enumerate(tqdm(zip(chunks, chunk_positions), 
                                                      total=len(chunks), 
                                                      desc="Creating frames")):
        chunk_lower = [t.lower() for t in chunk]
        
        # Skip chunks that don't contain the target word
        if target_word not in chunk_lower:
            continue
            
        # Find all occurrences of the target word in this chunk
        target_indices = [i for i, word in enumerate(chunk_lower) if word == target_word]
        
        # Create co-occurrence counts for this chunk
        co_occurrences = defaultdict(int)
        raw_occurrences = defaultdict(int)
        
        # For each occurrence of the target word in this chunk
        for idx in target_indices:
            # Define the vicinity range (handle chunk boundaries)
            start = max(0, idx - vicinity_size)
            end = min(len(chunk), idx + vicinity_size + 1)
            
            # Get unique words in this vicinity
            current_vicinity = set(chunk_lower[start:idx] + chunk_lower[idx+1:end])
            
            # Count each word once per vicinity
            for word in current_vicinity:
                co_occurrences[word] += 1
            
            # Count every occurrence (with potential double-counting)
            for i in range(start, end):
                if i != idx:  # Skip the target word itself
                    raw_occurrences[chunk_lower[i]] += 1
        
        # Create a graph for this chunk
        G = nx.Graph()
        
        # Add the target word as the central node
        target_freq = chunk_lower.count(target_word)
        G.add_node(target_word, 
                  size=1500, 
                  color='#FF6666',
                  frequency=target_freq)
        
        # Get the top co-occurring words for this chunk
        top_words = sorted(co_occurrences.items(), key=lambda x: x[1], reverse=True)[:top_connections]
        
        # Calculate the maximum co-occurrence for scaling
        max_count = max([count for _, count in top_words]) if top_words else 1
        
        # Add nodes and edges for top co-occurring words
        for word, co_occur_count in top_words:
            # Skip if it's the target word
            if word == target_word:
                continue
                
            # Get total frequency of this word in the chunk
            total_freq = chunk_lower.count(word)
            
            # Get raw occurrence count
            raw_count = raw_occurrences[word]
            
            # Scale node size based on co-occurrence frequency
            node_size = 200 + (co_occur_count / max_count) * 600
            
            G.add_node(word, 
                      size=node_size, 
                      color='#6699CC',
                      total_frequency=total_freq,
                      vicinity_frequency=co_occur_count,
                      raw_frequency=raw_count)
            
            # Edge weight based on co-occurrence count
            G.add_edge(target_word, word, 
                      weight=co_occur_count,
                      co_occurrence=co_occur_count,
                      raw_occurrence=raw_count)
        
        # Create figure without displaying it
        fig = plt.figure(figsize=(12, 12))
        
        # Position nodes using a spring layout with more space
        pos = nx.spring_layout(G, seed=42, k=0.3)
        
        # Get node sizes and colors
        node_sizes = [G.nodes[node]['size'] for node in G.nodes]
        node_colors = [G.nodes[node]['color'] for node in G.nodes]
        
        # Get edge weights for width
        edge_weights = [G[u][v]['weight'] for u, v in G.edges]
        max_edge_weight = max(edge_weights) if edge_weights else 1
        edge_widths = [2 + (w / max_edge_weight) * 8 for w in edge_weights]
        
        # Draw the network
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, edge_color='#999999')
        
        # Create node labels with word and vicinity frequency
        node_labels = {}
        for node in G.nodes:
            if node == target_word:
                node_labels[node] = f"{node}\n(freq: {G.nodes[node]['frequency']})"
            else:
                node_labels[node] = f"{node}\n(vec: {G.nodes[node]['vicinity_frequency']})"
        
        # Draw node labels
        nx.draw_networkx_labels(G, pos, 
                               labels=node_labels,
                               font_size=14, 
                               font_family='sans-serif', 
                               font_weight='bold',
                               verticalalignment='center')
        
        # Add a title showing chunk position and progress
        progress_percent = (position / len(tokens)) * 100
        plt.title(f"'{target_word}' network - Position: {progress_percent:.1f}%", 
                 fontsize=18)
        
        plt.axis('off')
        plt.tight_layout()
        
        # Save the frame to memory
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        frame = imageio.imread(buf)
        frames.append(frame)
        plt.close(fig)
    
    # Create the GIF if we have frames
    if frames:
        print(f"Creating GIF with {len(frames)} frames...")
        imageio.mimsave(output_gif, frames, fps=fps)
        print(f"GIF saved as {output_gif}")
        return output_gif
    else:
        print(f"No occurrences of '{target_word}' found in the text chunks.")
        return None