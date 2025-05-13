# Textual Product Mapping using RAG and LLM

This is an intelligent, automated system to match external products (varying product descriptions from suppliers) with internal items (inventory SKU product descriptions). The internal and external product description texts are acquired from separate CSV files.

A key criterion for matching is that it should be exact, meaning the product manufacturer, name, and size must be identical.

## Solution Architecture Flowchart

```mermaid
graph TD
    subgraph "Configuration"
        A[JSON Configuration Files] --> B[Load Unit of Measure Mappings]
        A --> C[Load Abbreviation Mappings]
    end

    subgraph "Vector Database Creation"
        D[Load Internal Product Data] --> E[Text Preprocessing]
        E --> F[Generate Vector Embeddings]
        F --> G{Save Checkpoints}
        G -->|Checkpoint| H[(Checkpoint Files)]
        G --> I[Create FAISS Index]
        I --> J[Save Vector Database]
        J --> K[(Vector Database Files)]
    end

    subgraph "Vector Database Update"
        L[Load Existing Vector Database] --> M[Load New Product Data]
        M --> N[Text Preprocessing]
        N --> O[Identify New Products]
        O --> P{Any New Products?}
        P -->|Yes| Q[Generate Vector Embeddings]
        P -->|No| R[Skip Update]
        Q --> S{Save Checkpoints}
        S -->|Checkpoint| T[(Checkpoint Files)]
        S --> U[Update FAISS Index]
        U --> V[Save Updated Vector Database]
        V --> K
    end

    subgraph "Product Matching"
        W[Load Vector Database] --> X[Load External Products]
        X --> Y[Text Preprocessing]
        Y --> Z[Generate Query Embeddings]
        Z --> AA[Perform Vector Similarity Search]
        AA --> AB[Filter Results by Size]
        AB --> AC[Sort by Similarity Score]
        AC --> AD[Return Top Matches]
    end

    subgraph "Utility Functions"
        AF[Resource Management] -.-> F
        AF -.-> Q
        AF -.-> Z
        AG[Error Handling & Recovery] -.-> G
        AG -.-> S
    end

    H -.-> G
    T -.-> S
    K --> L
    K --> W
```

## Core Features of the Solution

### 1. Vector Embedding + LLM Matching

This solution uses a two-step process for better accuracy:

- **Step 1:** Vector similarity search to find the most semantically similar product candidates
- **Step 2:** LLM verification to determine exact matches based on precise criteria

### 2. Memory-Efficient Implementation

- Data processing is done in small batches to reduce memory usage for vector embeddings and for product matching
- Garbage collection is forced between processing steps
- Checkpoint-based processing makes it safer to resume processing in case of failure saving computational resources and time.

### 3. Size-Based Filtering

To improve matching accuracy from the LLM, candidates retrieved from vector search are filtered based on size similarity. In other words, only size-filtered candidates are sent to the LLM for final verification.

## Project Directory Structure:

```bash
project_root/
│
├── config/ # Configuration data in JSON format
│ ├── __init__.py
│ ├── uom_mappings.json
│ └── abbr_mappings.json
│
├── core/ # Core functionality modules
│ ├── __init__.py
│ ├── vector_db_module.py
│ ├── product_matcher.py
│ └── vector_db_updater.py
│
├── data/ # Data files
│ ├── external.csv
│ ├── internal.csv
│ └── internal_new.csv
│
├── db/ # Vector database
│ ├── vector.index
│ └── vector.pkl
│
├── out/ # Output mapping file
│ └── output.csv
│
├── utils/ # Shared utility functions
│ ├── __init__.py
│ ├── config_loader.py
│ └── text_processing.py
│
├── main.py # Script for product mapping
├── create_vector_db.py # Script for creating vector database
├── update_vector_db.py # Script for updating vector database
├── requirements.txt # Package dependencies
└── .env # Environment file containing `OPENAI_API_KEY`
```
