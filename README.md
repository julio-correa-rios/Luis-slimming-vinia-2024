# Text and Speech Graph Analysis

A -very- simple Python toolkit for analysing text and speech patterns through graph-based approaches and traditional NLP techniques. In this particular case, I'm using it to analyse **Luis Slimming's Viña del Mar standup show**. 

## Overview

This project combines graph theory with natural language processing to analyse text and speech patterns. It provides tools for:

- Creating and analysing word networks from text
- Visualising word relationships as graphs
- Performing basic text analysis (word frequency, sentiment, etc.)
- Tracking the evolution of word contexts throughout a text

## Features

### Speech Graph Analysis
- **NaiveGraph**: Creates graphs where nodes are words and edges represent adjacency
- **StemGraph**: Creates graphs using stemmed words as nodes. To do this, it uses the SnowballStemmer from NLTK.
- **PosGraph**: Creates graphs using part-of-speech tags as nodes. To do this, it uses the averaged_perceptron_tagger from NLTK.
- **GraphStatistics**: Calculates various graph metrics (nodes, edges, loops, etc.)

### Text Analysis
- Word frequency analysis
- Sentiment analysis
- Word cloud generation
- Zipf's law verification
- Word length distribution
- Word ego networks (showing word contexts)
- Word evolution tracking (showing how contexts change throughout a text)

## Installation

# Clone the repository
git clone https://github.com/julio-correa-rios/Luis-slimming-vinia-2024
cd text-speech-analysis

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"
```

## Usage

### Basic Text Analysis

```python
from src.text_analysis import preprocess_text, basic_stats, word_frequency

# Analyze text
text = "Your text here..."
tokens = preprocess_text(text)
stats = basic_stats(text)

# Get word frequencies
freq_df = word_frequency(tokens, top_n=20)
```

### Speech Graph Analysis

```python
from src.speech_graph import NaiveGraph, GraphStatistics

# Create a graph from text
ng = NaiveGraph()
graph = ng.text2graph("Your text here...")

# Calculate graph statistics
stats = GraphStatistics(graph).statistics()

# Visualize the graph
ng.visualize_graph(graph, title="My Word Graph")
```

### Word Context Analysis

```python
from src.text_analysis import create_word_ego_network

# Create an ego network for a specific word
create_word_ego_network(
    text="Your text here...",
    target_word="example",
    vicinity_size=5,
    top_connections=15
)fsdfds 
```

## Examples

See the `notebooks/` directory for example Jupyter notebooks demonstrating various analyses.

## License

[MIT License](LICENSE)

## Acknowledgments

This project includes code adapted from [guillermodoghel/speechgraph](https://github.com/guillermodoghel/speechgraph).

---

# Análisis de Texto y grafos de discurso

Una herramienta Python para analizar patrones de texto y discurso mediante enfoques basados en grafos y técnicas tradicionales de PLN. Especialmente, se ha utilizado para analizar la transcripción del show de Luis Slimming en Viña del Mar 2024.

## Descripción General

Este proyecto combina teoría de grafos con procesamiento de lenguaje natural para analizar patrones de texto y habla. Proporciona herramientas para:

- Crear y analizar redes de palabras a partir de texto
- Visualizar relaciones entre palabras como grafos
- Realizar análisis básico de texto (frecuencia de palabras, sentimiento, etc.)
- Seguir la evolución de contextos de palabras a lo largo de un texto

## Características

### Análisis de Grafos de Habla
- **NaiveGraph**: Crea grafos donde los nodos son palabras y las aristas representan adyacencia
- **StemGraph**: Crea grafos usando palabras radicalizadas como nodos. Para esto, usa el SnowballStemmer de NLTK.
- **PosGraph**: Crea grafos usando etiquetas de partes del habla como nodos. Para esto, usa el averaged_perceptron_tagger de NLTK.
- **GraphStatistics**: Calcula varias métricas de grafos (nodos, aristas, bucles, etc.)

### Análisis de Texto
- Análisis de frecuencia de palabras
- Análisis de sentimiento
- Generación de nubes de palabras
- Verificación de la ley de Zipf
- Distribución de longitud de palabras
- Redes ego de palabras (mostrando contextos de palabras)
- Seguimiento de evolución de palabras (mostrando cómo cambian los contextos a lo largo de un texto)

## Instalación

# Clonar el repositorio
git clone https://github.com/julio-correa-rios/Luis-slimming-vinia-2024
cd text-speech-analysis

# Crear y activar un entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Descargar datos de NLTK
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"
```

## Uso

### Análisis Básico de Texto

```python
from src.text_analysis import preprocess_text, basic_stats, word_frequency

# Analizar texto
texto = "Tu texto aquí..."
tokens = preprocess_text(texto)
estadisticas = basic_stats(texto)

# Obtener frecuencias de palabras
df_freq = word_frequency(tokens, top_n=20)
```

### Análisis de Grafos de Habla

```python
from src.speech_graph import NaiveGraph, GraphStatistics

# Crear un grafo a partir de texto
ng = NaiveGraph()
grafo = ng.text2graph("Tu texto aquí...")

# Calcular estadísticas del grafo
estadisticas = GraphStatistics(grafo).statistics()

# Visualizar el grafo
ng.visualize_graph(grafo, title="Mi Grafo de Palabras")
```

### Análisis de Contexto de Palabras

```python
from src.text_analysis import create_word_ego_network

# Crear una red ego para una palabra específica
create_word_ego_network(
    text="Tu texto aquí...",
    target_word="ejemplo",
    vicinity_size=5,
    top_connections=15
)
```

## Ejemplos

Consulta el directorio `notebooks/` para ver un notebook que contiene todos estos análisis. 

## Licencia

[Licencia MIT](LICENSE)

## Agradecimientos

Este proyecto incluye código adaptado de [guillermodoghel/speechgraph](https://github.com/guillermodoghel/speechgraph).
Gracias a Luchito Slimming por las risas, el Sentido Del Humor y la inspiración para esta tallita. 
