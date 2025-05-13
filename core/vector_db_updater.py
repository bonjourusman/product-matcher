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
        """Identify products that don't already exist in the vector database."""
        # Create sets of existing and new product names for faster comparison
        existing_products = set(self.internal_df['VECTOR_NAME'].values)
        all_new_products = set(new_df['VECTOR_NAME'].values)
        
        # Find products that are in the new set but not in the existing set
        truly_new_products = all_new_products - existing_products
        
        # Filter the new_df to keep only the truly new products
        filtered_new_df = new_df[new_df['VECTOR_NAME'].isin(truly_new_products)].copy()
        
        print(f"Found {len(filtered_new_df)} truly new products out of {len(new_df)} total new products")
        return filtered_new_df
    
    def save_checkpoint(self, new_df, new_embeddings, checkpoint_path, checkpoint_idx):
        """Save an intermediate checkpoint during processing."""
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        checkpoint_data = {
            'new_df': new_df,
            'new_embeddings': new_embeddings,
            'embedding_model_name': self.embedding_model_name,
            'checkpoint_idx': checkpoint_idx
        }
        
        # Use a consistent naming pattern for checkpoints
        checkpoint_filepath = f"{checkpoint_path}.{checkpoint_idx}.pkl"
        temp_path = f"{checkpoint_filepath}.tmp"
        
        try:
            with open(temp_path, 'wb') as f:
                pickle.dump(checkpoint_data, f, protocol=4)
                f.flush()
            
            # Rename for atomic replacement
            if os.path.exists(checkpoint_filepath):
                os.remove(checkpoint_filepath)
            os.rename(temp_path, checkpoint_filepath)
            
            print(f"Saved checkpoint {checkpoint_idx} to {checkpoint_filepath}")
            return True
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            return False

    def load_latest_checkpoint(self, checkpoint_path):
        """Load the latest checkpoint if it exists."""
        # Find all checkpoint files
        checkpoint_dir = os.path.dirname(checkpoint_path)
        base_name = os.path.basename(checkpoint_path)
        
        if not os.path.exists(checkpoint_dir):
            return None, None, 0
        
        checkpoint_files = []
        try:
            # The checkpoint path pattern
            checkpoint_base = f"{base_name}.checkpoint"
            
            # List all files in the directory
            all_files = os.listdir(checkpoint_dir)
            
            # Find checkpoint files with pattern like "base.checkpoint.123.pkl"
            for f in all_files:
                if f.startswith(checkpoint_base) and f.endswith('.pkl') and not f.endswith('.tmp'):
                    checkpoint_files.append(f)
        except Exception as e:
            print(f"Error listing checkpoint files: {e}")
            return None, None, 0
        
        if not checkpoint_files:
            return None, None, 0
        
        # Sort by checkpoint index
        try:
            checkpoint_files.sort(key=lambda f: int(f.split('.')[-2]))
            latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
        except Exception as e:
            print(f"Error sorting checkpoint files: {e}")
            return None, None, 0
        
        print(f"Loading latest checkpoint: {latest_checkpoint}")
        
        try:
            with open(latest_checkpoint, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            new_df = checkpoint_data.get('new_df')
            new_embeddings = checkpoint_data.get('new_embeddings')
            checkpoint_idx = checkpoint_data.get('checkpoint_idx', 0)
            
            return new_df, new_embeddings, checkpoint_idx
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None, None, 0

    def _cleanup_checkpoints(self, filepath):
        """Clean up checkpoint files after successful save."""
        checkpoint_dir = os.path.dirname(filepath)
        base_name = os.path.basename(filepath)
        
        try:
            # Only proceed if the directory exists
            if not os.path.exists(checkpoint_dir):
                return
                
            print(f"Looking for checkpoint files in {checkpoint_dir}...")
            
            # The checkpoint path pattern
            checkpoint_base = f"{base_name}.checkpoint"
            
            # List all files in the directory
            all_files = os.listdir(checkpoint_dir)
            
            # Find checkpoint files with pattern like "base.checkpoint.123.pkl"
            checkpoint_files = []
            for f in all_files:
                if f.startswith(checkpoint_base) and f.endswith('.pkl') and not f.endswith('.tmp'):
                    checkpoint_files.append(os.path.join(checkpoint_dir, f))
            
            if checkpoint_files:
                print(f"Found {len(checkpoint_files)} checkpoint files to clean up")
                
                for f in checkpoint_files:
                    try:
                        os.remove(f)
                        print(f"Removed checkpoint file: {f}")
                    except Exception as e:
                        print(f"Warning: Could not remove checkpoint file {f}: {e}")
            else:
                print("No checkpoint files found to clean up")
                
            # Also check for any temporary files that might have been left behind
            temp_files = [
                os.path.join(checkpoint_dir, f) for f in all_files
                if f.endswith('.tmp')
            ]
            
            for f in temp_files:
                try:
                    os.remove(f)
                    print(f"Removed temporary file: {f}")
                except Exception as e:
                    print(f"Warning: Could not remove temporary file {f}: {e}")
                    
        except Exception as e:
            print(f"Warning: Failed to clean up checkpoints: {e}")
    
    def generate_embeddings(self, new_df, batch_size=32):
        """Generate embeddings with checkpointing for resilience."""
        if len(new_df) == 0:
            print("No new products to embed")
            return np.array([])
            
        print("Generating embeddings for new products...")
        
        # Generate embeddings for new products
        texts = new_df['VECTOR_NAME'].tolist()
            
        # Initialize empty array for embeddings
        total_texts = len(texts)
        embedding_dim = self.embed_model.get_sentence_embedding_dimension()
        new_embeddings = np.zeros((total_texts, embedding_dim), dtype=np.float32)
        
        # Use a smaller batch size for stability
        for i in tqdm(range(0, total_texts, batch_size), desc="Generating embeddings"):
            batch_end = min(i+batch_size, total_texts)
            batch_texts = texts[i:batch_end]
            
            # Use a more direct approach to encoding
            batch_embeddings = self.embed_model.encode(
                batch_texts, 
                show_progress_bar=False,
                convert_to_numpy=True,
                batch_size=len(batch_texts),
                normalize_embeddings=True,
                # Disable multiprocessing explicitly
                num_workers=0
            )
            
            new_embeddings[i:batch_end] = batch_embeddings
            
            # Explicitly force garbage collection
            gc.collect()
            
            # Check if we've processed at least 25% of the data, yield intermediate results
            if i > 0 and i % max(1, total_texts // 4) == 0:
                # Create a copy of current embeddings for checkpoint
                current_embeddings = new_embeddings[:batch_end].copy()
                
                # Normalize the current embeddings
                norms = np.linalg.norm(current_embeddings, axis=1, keepdims=True)
                current_embeddings = current_embeddings / np.maximum(norms, 1e-10)  # Avoid division by zero
                
                # Convert to float32 for better compatibility with FAISS
                current_embeddings = current_embeddings.astype(np.float32)
                
                print(f"Processed {batch_end}/{total_texts} items. Saving checkpoint...")
                yield current_embeddings, batch_end
        
        # Normalize final embeddings using numpy
        norms = np.linalg.norm(new_embeddings, axis=1, keepdims=True)
        new_embeddings = new_embeddings / np.maximum(norms, 1e-10)  # Avoid division by zero
        
        # Convert to float32 for better compatibility with FAISS
        new_embeddings = new_embeddings.astype(np.float32)
        
        # Return the final normalized embeddings
        yield new_embeddings, total_texts
    
    def update_vector_database(self, new_df, new_embeddings):
        """Update the vector database with new products."""
        if len(new_df) == 0:
            print("No new products to update")
            return self.internal_df, self.internal_embeddings, self.index
        
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
            f"{filepath}.index.tmp"
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
        """Execute the complete pipeline for updating a vector database with checkpointing."""
        checkpoint_path = f"{self.vector_db_filepath}.checkpoint"
        
        print(f"Starting vector database update with checkpoint path: {checkpoint_path}")
        
        # Try to load from checkpoint first
        checkpoint_new_df, checkpoint_new_embeddings, checkpoint_idx = self.load_latest_checkpoint(checkpoint_path)
        
        # If no checkpoint, start from scratch
        if checkpoint_new_df is None:
            print("No checkpoint found. Starting from scratch...")
            # Load and preprocess new data
            new_df = self.load_data(new_data_path)
            new_df = self.preprocess_data(new_df, name_column)
            
            # Identify new products
            new_products = self.identify_new_products(new_df)
            
            if len(new_products) == 0:
                print("No new products to add.")
                return self.internal_df, self.internal_embeddings, self.index
                
            checkpoint_idx = 0
        else:
            print(f"Resuming from checkpoint with {checkpoint_idx} processed items.")
            new_products = checkpoint_new_df
        
        # Skip already processed embeddings if we have a checkpoint
        if checkpoint_new_embeddings is not None and checkpoint_idx > 0:
            # Start where we left off
            embedding_dim = self.embed_model.get_sentence_embedding_dimension()
            new_embeddings = np.zeros((len(new_products), embedding_dim), dtype=np.float32)
            new_embeddings[:checkpoint_idx] = checkpoint_new_embeddings[:checkpoint_idx]
        else:
            # Start from the beginning
            checkpoint_new_embeddings = None
        
        # Generate embeddings safely, potentially in chunks
        for embeddings_chunk, processed_count in self.generate_embeddings(
            new_products, batch_size=batch_size
        ):
            # If we have a checkpoint, merge with new data
            if checkpoint_new_embeddings is not None:
                # Update only the new part
                total_processed = checkpoint_idx + processed_count
                # Save a checkpoint with the combined data
                self.save_checkpoint(new_products, embeddings_chunk, checkpoint_path, total_processed)
            else:
                # No checkpoint, just save the new data
                self.save_checkpoint(new_products, embeddings_chunk, checkpoint_path, processed_count)
                checkpoint_new_embeddings = embeddings_chunk
        
        # At this point, checkpoint_new_embeddings should contain all our embeddings
        # Update the vector database
        updated_df, updated_embeddings, updated_index = self.update_vector_database(
            new_products, checkpoint_new_embeddings
        )
        
        # Save the updated database
        success = self.save_vector_database(
            updated_df, updated_embeddings, updated_index
        )
        
        if success:
            print("Vector database update completed successfully.")
            # Clean up checkpoint files
            self._cleanup_checkpoints(self.vector_db_filepath)
            return updated_df, updated_embeddings, updated_index
        else:
            print("Failed to save updated vector database. Check the checkpoint files to recover.")
            return None, None, None