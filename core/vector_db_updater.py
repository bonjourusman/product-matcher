# core/vector_db_updater.py
import os
import gc
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import faiss
from sentence_transformers import SentenceTransformer
from utils.config_loader import get_uom_dict, get_abbr_dict
from utils.text_processing import standardize_text

class VectorDBUpdater:
    def __init__(self, vector_db_filepath):
        """Initialize the Vector DB Updater with an existing vector database."""
        self.vector_db_filepath = vector_db_filepath
        self.load_vector_database()
        self.uom_dict = get_uom_dict()
        self.abbr_dict = get_abbr_dict()
        
    def _setup_embedding_model(self, model_name):
        """Set up the sentence transformer embedding model with multiprocessing disabled."""
        # Create the model
        embedding_model = SentenceTransformer(model_name, device="cpu")
        
        # Disable multiprocessing features
        embedding_model.parallel_tokenization = False
        
        # Set a safer default for encoding parameters
        embedding_model._target_devices = [torch.device('cpu')]
        
        # Disable multiprocessing pool if it exists
        if hasattr(embedding_model, 'pool'):
            embedding_model.pool = None
        
        print(f"Loaded embedding model: {model_name} on CPU (multiprocessing disabled)")
        return embedding_model
        
    def __del__(self):
        """Destructor to ensure proper cleanup of resources."""
        if hasattr(self, 'embed_model'):
            del self.embed_model
        gc.collect()
    
    def load_vector_database(self):
        """Load the existing vector database."""
        # Check if files exist
        pkl_file = f"{self.vector_db_filepath}.pkl"
        index_file = f"{self.vector_db_filepath}.index"
        
        if not os.path.exists(pkl_file) or not os.path.exists(index_file):
            raise FileNotFoundError(f"Vector database files not found at {self.vector_db_filepath}")
        
        # Load pickled data
        try:
            with open(pkl_file, 'rb') as f:
                vector_db_data = pickle.load(f)
            
            self.internal_df = vector_db_data.get('internal_df')
            self.internal_embeddings = vector_db_data.get('internal_embeddings')
            self.embedding_model_name = vector_db_data.get('embedding_model_name', "all-MiniLM-L6-v2")
            
            # Initialize embedding model
            self.embed_model = self._setup_embedding_model(self.embedding_model_name)
            
            # Load FAISS index
            self.index = faiss.read_index(index_file)
            
            print(f"Loaded vector database with {len(self.internal_df)} products")
            print(f"Using embedding model: {self.embedding_model_name}")
            
        except Exception as e:
            raise RuntimeError(f"Error loading vector database: {e}")
    
    def load_data(self, filepath):
        """Load new product data."""
        df = pd.read_csv(filepath)
        df.drop_duplicates(ignore_index=True, inplace=True)
        return df
    
    def preprocess_data(self, df, name_column='LONG_NAME'):
        """Clean and standardize product text for vector embedding."""
        df['VECTOR_NAME'] = df[name_column].apply(
            standardize_text, 
            args=(self.uom_dict, self.abbr_dict)
        )
        return df
    
    def identify_new_products(self, new_df):
        """Identify products that are not in the internal database."""
        # Create sets of existing and new product names for faster comparison
        existing_products = set(self.internal_df['VECTOR_NAME'].values)
        all_new_products = set(new_df['VECTOR_NAME'].values)
        
        # Find products that are in the new set but not in the existing set
        truly_new_products = all_new_products - existing_products
        
        # Filter the new_df to keep only the truly new products
        filtered_new_df = new_df[new_df['VECTOR_NAME'].isin(truly_new_products)].copy()
        
        print(f"Found {len(filtered_new_df)} truly new products out of {len(new_df)} total new products")
        return filtered_new_df
    
    def generate_embeddings(self, new_df, batch_size=16):
        """Generate embeddings for new products."""
        print("Generating embeddings for new products...")
        
        # Generate embeddings for new products
        texts = new_df['VECTOR_NAME'].tolist()
            
        # Initialize empty array for embeddings
        total_texts = len(texts)
        embedding_dim = self.embed_model.get_sentence_embedding_dimension()
        new_embeddings = np.zeros((total_texts, embedding_dim), dtype=np.float32)
        
        try:
            # Process in batches
            for i in tqdm(range(0, total_texts, batch_size), desc="Generating embeddings"):
                batch_texts = texts[i:min(i+batch_size, total_texts)]
                
                # Use encode with safer parameters
                batch_embeddings = self.embed_model.encode(
                    batch_texts, 
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    batch_size=len(batch_texts),
                    normalize_embeddings=True,
                    # Explicitly disable parallel encoding
                    num_workers=0
                )
                
                new_embeddings[i:i+len(batch_texts)] = batch_embeddings
                
                # Force garbage collection after each batch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error during embedding generation: {e}")
            # Clean up any remaining resources
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
        
        # Normalize embeddings using numpy
        norms = np.linalg.norm(new_embeddings, axis=1, keepdims=True)
        new_embeddings = new_embeddings / np.maximum(norms, 1e-10)  # Avoid division by zero
        
        # Convert to float32 for better compatibility with FAISS
        new_embeddings = new_embeddings.astype(np.float32)
        
        print(f"Generated embeddings for {len(texts)} new products")
        
        return new_embeddings
    
    def update_vector_database(self, new_df, new_embeddings):
        """Update the vector database with new products."""
        print("Updating vector database...")
        
        # Combine DataFrames
        updated_df = pd.concat([self.internal_df, new_df], ignore_index=True)
        
        # Combine embeddings
        if len(new_embeddings) > 0:
            updated_embeddings = np.vstack([self.internal_embeddings, new_embeddings])
        else:
            updated_embeddings = self.internal_embeddings
        
        # Update FAISS index
        if len(new_embeddings) > 0:
            self.index.add(new_embeddings)
        
        print(f"Vector database updated with {len(new_df)} new products")
        print(f"Total products in updated database: {len(updated_df)}")
        
        return updated_df, updated_embeddings, self.index
    
    def save_vector_database(self, updated_df, updated_embeddings, updated_index, filepath=None):
        """Save the updated vector database safely."""
        if filepath is None:
            filepath = self.vector_db_filepath
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        print("Saving updated data to pickle file...")
        # Save the dictionary with data
        temp_data_path = f"{filepath}.pkl.tmp"
        
        # Create a dictionary with all necessary data
        vector_db_data = {
            'internal_df': updated_df.copy(),
            'internal_embeddings': updated_embeddings.copy(),
            'embedding_model_name': self.embedding_model_name
        }
        
        # Save data to pickle file
        try:
            with open(temp_data_path, 'wb') as f:
                pickle.dump(vector_db_data, f, protocol=4)
                f.flush()
            
            # Rename temp file to final file
            if os.path.exists(f"{filepath}.pkl"):
                os.remove(f"{filepath}.pkl")
            os.rename(temp_data_path, f"{filepath}.pkl")
            print(f"Successfully saved updated data to {filepath}.pkl")
        except Exception as e:
            print(f"Error saving vector database data: {e}")
            if os.path.exists(temp_data_path):
                try:
                    os.remove(temp_data_path)
                except:
                    pass
            return False
        
        print("Saving updated FAISS index...")
        # Save the FAISS index separately
        temp_index_path = f"{filepath}.index.tmp"
        try:
            faiss.write_index(updated_index, temp_index_path)
            if os.path.exists(f"{filepath}.index"):
                os.remove(f"{filepath}.index")
            os.rename(temp_index_path, f"{filepath}.index")
            print(f"Successfully saved updated index to {filepath}.index")
        except Exception as e:
            print(f"Error saving FAISS index: {e}")
            if os.path.exists(temp_index_path):
                try:
                    os.remove(temp_index_path)
                except:
                    pass
            return False
        
        # Clean up any leftover temporary files
        cleanup_files = [
            f"{filepath}.pkl.tmp",
            f"{filepath}.index.tmp",
            f"{filepath}.data.tmp"
        ]
        for tmp_file in cleanup_files:
            if os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                    print(f"Removed temporary file: {tmp_file}")
                except Exception as e:
                    print(f"Warning: Could not remove temporary file {tmp_file}: {e}")
        
        print(f"Updated vector database saved successfully to {filepath}.pkl and {filepath}.index")
        return True
    
    def run_update_pipeline(self, new_data_path, name_column='LONG_NAME', batch_size=16):
        """Execute the complete pipeline for updating a vector database."""
        # Load new data
        print(f"Loading new data from {new_data_path}...")
        new_df = self.load_data(new_data_path)
        
        # Preprocess new data
        print("Preprocessing new data...")
        new_df = self.preprocess_data(new_df, name_column)
        
        # Identify new products
        print("Identifying new products...")
        new_products = self.identify_new_products(new_df)
        
        # If no new products, return existing data
        if len(new_products) == 0:
            print("No new products found")
            return self.internal_df, self.internal_embeddings, self.index
        
        # Generate embeddings for new products
        print("Generating embeddings for new products...")
        new_embeddings = self.generate_embeddings(new_products, batch_size)
        
        # Update vector database
        print("Updating vector database...")
        updated_df, updated_embeddings, updated_index = self.update_vector_database(new_products, new_embeddings)
        
        # Save updated vector database
        print("Saving updated vector database...")
        success = self.save_vector_database(updated_df, updated_embeddings, updated_index)
        
        if success:
            return updated_df, updated_embeddings, updated_index
        else:
            print("Failed to save updated vector database")
            return None, None, None