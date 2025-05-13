import os
import gc
import time
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from utils.config_loader import get_uom_dict, get_abbr_dict
from utils.text_processing import standardize_text, extract_size_info

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class ProductMatcher:
    def __init__(
        self, 
        vector_db_filepath, 
        llm_model="gpt-4.1", 
        top_k=10, 
        batch_size=10, 
        max_retries=3
    ):
        """Initialize the Product Matcher with necessary parameters."""
        self.vector_db_filepath = vector_db_filepath
        self.llm_model = llm_model
        self.top_k = top_k
        self.batch_size = batch_size
        self.max_retries = max_retries
        
        # Initialize LLM Client
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.llm_client = OpenAI()
        
        # Load Vector Database
        self.internal_df, self.internal_embeddings, self.index, self.embed_model = self._load_vector_database()
        
        # Set up dictionaries for text processing
        self.uom_dict = get_uom_dict()
        self.abbr_dict = get_abbr_dict()
    
    def _setup_embedding_model(self, model_name):
        """Set up the sentence transformer embedding model for vector embedding generation."""
        embedding_model = SentenceTransformer(model_name, device="cpu")
        print(f"Loaded embedding model: {model_name} on CPU")
        return embedding_model
        
    def _load_vector_database(self):
        """Load a previously saved vector database."""
        # Force garbage collection before loading the vector db and model
        gc.collect()
        
        # Load the pickle data
        with open(f"{self.vector_db_filepath}.pkl", 'rb') as f:
            vector_db_data = pickle.load(f)
        
        # Load data and embeddings
        df = vector_db_data['internal_df']
        embeddings = vector_db_data['internal_embeddings']
        
        # Load FAISS index
        index = faiss.read_index(f"{self.vector_db_filepath}.index")
        
        # Load Embedding Model
        model_name = vector_db_data['embedding_model_name']
        embed_model = self._setup_embedding_model(model_name)
            
        print(f"Vector database loaded from {self.vector_db_filepath}.pkl and {self.vector_db_filepath}.index")

        return df, embeddings, index, embed_model
    
    def load_external_data(self, filepath):
        """Load and preprocess external product data."""
        df = pd.read_csv(filepath)
        df.drop_duplicates(ignore_index=True, inplace=True)
        return df
    
    def preprocess_external_data(self, df, name_column='PRODUCT_NAME'):
        """Preprocess external product data for matching."""
        df['VECTOR_NAME'] = df[name_column].apply(
            standardize_text, 
            args=(self.uom_dict, self.abbr_dict)
        )
        return df
    
    def _vector_retrieve_candidates(self, query_text):
        """Retrieve top-k similar internal products using vector similarity search."""
        # Generate embedding for the query
        query_embedding = self.embed_model.encode([query_text])[0].reshape(1, -1)
        
        # Normalize query embedding using numpy
        norm = np.linalg.norm(query_embedding)
        query_embedding = query_embedding / np.maximum(norm, 1e-10)  # Avoid division by zero
        
        # Convert to float32 for FAISS
        query_embedding = query_embedding.astype(np.float32)
        
        # Search in the index
        scores, indices = self.index.search(query_embedding, self.top_k)
        
        # Get the corresponding internal products
        candidates = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.internal_df):  # Valid index check
                candidates.append({
                    'product': self.internal_df.iloc[idx]['VECTOR_NAME'],
                    'score': float(scores[0][i])
                })
        
        return candidates
    
    def _size_filtered_candidates(self, external_product, candidates):
        """
        Filter candidates based on size matching.
        Handles both cases:
        1. internal product name contains size info
        2. internal product name doesn't contain size info
        """

        size_filtered_candidates = []
    
        ext_size = extract_size_info(external_product, self.uom_dict)
        '''
        # CASE 1: More sensitive (less flexibile) to size matching        
        for c in candidates:
            int_size = extract_size_info(c['product'], self.uom_dict)
            if int_size: # internal product name has size info
                if ext_size and int_size[0] == ext_size[0] and int_size[1] == ext_size[1]: # external product name has same size
                    size_filtered_candidates.append(c)
                    
            else: # internal product name has no size info
                if not ext_size: # external product name also has no size info
                    size_filtered_candidates.append(c)
        '''
        # CASE 2: Less sensitive (more flexible) to size matching
        for c in candidates:
            int_size = extract_size_info(c['product'], self.uom_dict)
            if int_size: # internal product name has size info
                if not ext_size: # external product name has no size info
                    size_filtered_candidates.append(c)
                elif ext_size and int_size[0] == ext_size[0] and int_size[1] == ext_size[1]: # external product name has same size
                    size_filtered_candidates.append(c)
                    
            else: # internal product name has no size info
                if not ext_size: # external product name also has no size info
                    size_filtered_candidates.append(c)
        
        return size_filtered_candidates
    
    def match_products(self, external_df):
        """
        Match external products to internal products using vector retrieval and LLM.
        """
        # Initialize result dataframe with external products
        result_df = external_df[['VECTOR_NAME']].copy()
        result_df.rename(columns={'VECTOR_NAME':'ext_prod_name'}, inplace=True)
        result_df['int_prod_name'] = 'NULL'
        result_df['vector_similarity'] = 0.0

        # Process each batch
        for batch_start in range(0, len(external_df), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(external_df))
            batch_indices = range(batch_start, batch_end)
            
            # Process each external product
            for idx in tqdm(batch_indices, desc=f"LLM matching batch {batch_start//self.batch_size + 1}"):
                ext_product = external_df.iloc[idx]['VECTOR_NAME']
        
                # Top-K candidates retrieved from Vector DB
                candidates = self._vector_retrieve_candidates(ext_product)

                #print('External Product: \n', ext_product)
                #print('Candidates: \n', candidates)

                # Skip if no good candidates found in Vector DB
                if not candidates:
                    continue
        
                # Candidates with matching size
                candidate_matches = self._size_filtered_candidates(ext_product, candidates)
        
                # Skip if no candidates left that match size
                if not candidate_matches:     
                    continue
                    
                prompt_products = "\n".join([f"{i+1}. {match_dict['product']}" for i, match_dict in enumerate(candidate_matches)])

                #print('Candidate Matches: \n', candidate_matches)
                #print('Prompt Products: \n', prompt_products)
                
                # Prepare prompt for LLM
                prompt = f""" Task: Match an external product name to the most appropriate internal product name based on close matching of product manufacturer and name.

External Product: {ext_product}

Candidate Internal Products:
{prompt_products}

Instructions:
1. The match must be almost exact - product manufacturer and product description must be almost identical.
2. If there's no exact match among the candidates, respond with "No match found".
3. If there is a match, respond with the number of the matching product only (e.g., "3").

Response:
"""
                # Try API call with retries
                for attempt in range(self.max_retries):
                    try:
                        response = self.llm_client.chat.completions.create(
                            model = self.llm_model,
                            messages=[
                                {"role": "system", "content": "You are a precise product matching assistant."},
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=10,
                            temperature=0
                        )
                        break
                        
                    except Exception as e:
                        print(f"Error on attempt {attempt+1}: {e}")
                        if attempt < self.max_retries - 1:
                            time.sleep(2 ** attempt)  # Exponential backoff
                        else:
                            print(f"Failed to process {ext_product} after {self.max_retries} attempts")
                            response = None
                
                if response:
                    llm_response = response.choices[0].message.content.strip()
                    
                    # Parse response
                    try:
                        if llm_response.isdigit() and 1 <= int(llm_response) <= len(candidate_matches):
                            match_idx = int(llm_response) - 1
                            result_df.at[idx, 'int_prod_name'] = candidate_matches[match_idx]['product']
                            result_df.at[idx, 'vector_similarity'] = candidate_matches[match_idx]['score']
                    except:
                        pass  # No valid match found
           
            # Pause between batches to respect rate limits
            if batch_end < len(external_df):
                time.sleep(2)

        return result_df
    
    def export_results(self, result_df, external_df, output_filepath):
        """Export matched results to CSV."""
        output_df = result_df.copy()
        output_df = pd.merge(
            output_df, 
            self.internal_df[['VECTOR_NAME','LONG_NAME']], 
            left_on='int_prod_name', 
            right_on='VECTOR_NAME', 
            how='left'
        )
        output_df = pd.merge(
            output_df, 
            external_df[['VECTOR_NAME','PRODUCT_NAME']], 
            left_on='ext_prod_name', 
            right_on='VECTOR_NAME', 
            how='left'
        )
        output_df.rename(columns={'PRODUCT_NAME':'external', 'LONG_NAME':'internal'}, inplace=True)
        
        output_df = output_df[['external','internal']]
        output_df['internal'] = output_df['internal'].fillna('NULL')

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        
        # Save results
        output_df.to_csv(output_filepath, index=False)

        # Print summary
        match_count = output_df['internal'].ne('NULL').sum()
        print(f"Found {match_count} matches out of {len(output_df)} records using LLM matching")
        print(f"Results saved to {output_filepath}")
        
        return output_df
        
    def run_match_pipeline(self, external_data_filepath, output_filepath):
        """Execute the complete product matching pipeline."""
        # Load and preprocess external data
        external_df = self.load_external_data(external_data_filepath)
        external_df = self.preprocess_external_data(external_df)
        
        # Match products
        result_df = self.match_products(external_df)
        
        # Export results
        output_df = self.export_results(result_df, external_df, output_filepath)
        
        return output_df