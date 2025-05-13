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

class VectorDBCreator:
    def __init__(self, embed_model_name="all-MiniLM-L6-v2"):
        """Initialize the Vector DB Creator with an embedding model."""
        self.embed_model_name = embed_model_name
        self.embed_model = self._setup_embedding_model()
        self.uom_dict = get_uom_dict()
        self.abbr_dict = get_abbr_dict()
        
    def _setup_embedding_model(self):
        """Set up the sentence transformer embedding model with multiprocessing disabled."""
        # Create the model
        embedding_model = SentenceTransformer(self.embed_model_name, device="cpu")
        
        # Disable multiprocessing features
        embedding_model.parallel_tokenization = False
        
        # Set a safer default for encoding parameters
        embedding_model._target_devices = [torch.device('cpu')]
        
        # Disable multiprocessing pool if it exists
        if hasattr(embedding_model, 'pool'):
            embedding_model.pool = None
        
        print(f"Loaded embedding model: {self.embed_model_name} on CPU (multiprocessing disabled)")
        return embedding_model
    
    def __del__(self):
        """Destructor to ensure proper cleanup of resources."""
        if hasattr(self, 'embed_model'):
            del self.embed_model
        gc.collect()
    
    def load_data(self, filepath):
        """Load and preprocess internal product data."""
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
    
    def load_and_preprocess_data(self, filepath):
        """Load and preprocess data as a separate step."""
        df = self.load_data(filepath)
        df = self.preprocess_data(df)
        return df
    
    def _safe_save_to_file(self, data, filepath, is_binary=True):
        """Safely save data to a file with proper error handling."""
        temp_filepath = f"{filepath}.tmp"
        mode = 'wb' if is_binary else 'w'
        
        try:
            with open(temp_filepath, mode) as f:
                if is_binary:
                    pickle.dump(data, f, protocol=4)
                else:
                    f.write(data)
                f.flush()  # Flush within context
            
            # Rename for atomic replacement
            os.rename(temp_filepath, filepath)
            return True
        except Exception as e:
            print(f"Error saving to {filepath}: {e}")
            if os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except:
                    pass
            return False
    
    def generate_embeddings(self, texts, batch_size=32):
        """Generate embeddings in a safe manner, handling process isolation."""
        total_texts = len(texts)
        embedding_dim = self.embed_model.get_sentence_embedding_dimension()
        embeddings = np.zeros((total_texts, embedding_dim), dtype=np.float32)
        
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
                # Disable multiprocessing explicitly
                num_workers=0
            )
            
            embeddings[i:batch_end] = batch_embeddings
            
            # Explicitly force garbage collection
            gc.collect()
            
            # Check if we've processed at least 25% of the data, save intermediate results
            if i > 0 and i % max(1, total_texts // 4) == 0:
                print(f"Processed {i}/{total_texts} items. Saving checkpoint...")
                yield embeddings[:batch_end].copy(), i
        
        # Return the final embeddings
        yield embeddings, total_texts
    
    """
    def create_vector_database(self, df, batch_size=32):
        '''Create a FAISS vector database with internal product embeddings.'''
        print("Creating vector database for internal products...")
        
        # Generate embeddings for internal products
        texts = df['VECTOR_NAME'].tolist()
            
        # Initialize empty array for embeddings
        total_texts = len(texts)
        embedding_dim = self.embed_model.get_sentence_embedding_dimension()
        embeddings = np.zeros((total_texts, embedding_dim), dtype=np.float32)
        
        try:
            # Process in batches
            for i in tqdm(range(0, total_texts, batch_size), desc="Generating embeddings"):
                batch_texts = texts[i:min(i+batch_size, total_texts)]
                
                # Use encode with show_progress_bar=False and convert_to_numpy=True
                # and explicitly set batch_size and normalize_embeddings
                batch_embeddings = self.embed_model.encode(
                    batch_texts, 
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    batch_size=len(batch_texts),
                    normalize_embeddings=True,
                    # Explicitly disable parallel encoding
                    num_workers=0
                )
                
                embeddings[i:i+len(batch_texts)] = batch_embeddings
                
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
        
        # Ensure embeddings are in the right format for FAISS
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
        
        # Create FAISS index
        vector_dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(vector_dimension)  # Inner product for cosine similarity
        
        # Add embeddings to index
        index.add(embeddings)
        
        print(f"Vector database created with {len(texts)} internal products")
        
        return embeddings, index
    """

    def create_index_from_embeddings(self, embeddings):
        """Create a FAISS index from embeddings."""
        # Ensure embeddings are properly formatted for FAISS
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
        
        # Create FAISS index
        vector_dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(vector_dimension)
        
        # Add embeddings to index
        index.add(embeddings)
        
        return index
    
    def save_checkpoint(self, df, embeddings, checkpoint_path, checkpoint_idx):
        """Save an intermediate checkpoint during processing."""
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        checkpoint_data = {
            'df': df,
            'embeddings': embeddings,
            'embedding_model_name': self.embed_model_name,
            'checkpoint_idx': checkpoint_idx
        }
        
        checkpoint_filepath = f"{checkpoint_path}.{checkpoint_idx}.pkl"
        success = self._safe_save_to_file(checkpoint_data, checkpoint_filepath)
        
        if success:
            print(f"Saved checkpoint {checkpoint_idx} to {checkpoint_filepath}")
        else:
            print(f"Failed to save checkpoint {checkpoint_idx}")
    
    def load_latest_checkpoint(self, checkpoint_path):
        """Load the latest checkpoint if it exists."""
        # Find all checkpoint files
        checkpoint_dir = os.path.dirname(checkpoint_path)
        base_name = os.path.basename(checkpoint_path)
        
        if not os.path.exists(checkpoint_dir):
            return None, None, 0
        
        checkpoint_files = []
        try:
            checkpoint_files = [
                f for f in os.listdir(checkpoint_dir) 
                if f.startswith(base_name) and f.endswith('.pkl') and '.tmp' not in f
            ]
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
            
            df = checkpoint_data.get('df')
            embeddings = checkpoint_data.get('embeddings')
            checkpoint_idx = checkpoint_data.get('checkpoint_idx', 0)
            
            return df, embeddings, checkpoint_idx
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None, None, 0
    
    def save_vector_database(self, df, embeddings, filepath):
        """Save the vector database safely, handling potential errors."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        print("Creating FAISS index...")
        try:
            index = self.create_index_from_embeddings(embeddings)
        except Exception as e:
            print(f"Error creating FAISS index: {e}")
            return False
        
        print("Saving data to pickle file...")
        # Save the dictionary with data
        temp_data_path = f"{filepath}.pkl.tmp"
        
        # Create a dictionary with all necessary data
        vector_db_data = {
            'internal_df': df.copy(),
            'internal_embeddings': embeddings.copy(),
            'embedding_model_name': self.embed_model_name
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
            print(f"Successfully saved data to {filepath}.pkl")
        except Exception as e:
            print(f"Error saving vector database data: {e}")
            if os.path.exists(temp_data_path):
                try:
                    os.remove(temp_data_path)
                except:
                    pass
            return False
        
        print("Saving FAISS index...")
        # Save the FAISS index separately
        temp_index_path = f"{filepath}.index.tmp"
        try:
            faiss.write_index(index, temp_index_path)
            if os.path.exists(f"{filepath}.index"):
                os.remove(f"{filepath}.index")
            os.rename(temp_index_path, f"{filepath}.index")
            print(f"Successfully saved index to {filepath}.index")
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
        
        # Clean up checkpoints
        self._cleanup_checkpoints(filepath)
        
        print(f"Vector database saved successfully to {filepath}.pkl and {filepath}.index")
        return True
    
    def _cleanup_checkpoints(self, filepath):
        """Clean up checkpoint files after successful save."""
        checkpoint_dir = os.path.dirname(filepath)
        base_name = os.path.basename(filepath)
        
        try:
            # Only proceed if the directory exists
            if not os.path.exists(checkpoint_dir):
                return
                
            print(f"Looking for checkpoint files in {checkpoint_dir}...")
            
            # Create a pattern to match checkpoint files
            checkpoint_pattern = f"{base_name}.checkpoint"
            
            # List all files in the directory and filter for checkpoints
            all_files = os.listdir(checkpoint_dir)
            checkpoint_files = [
                os.path.join(checkpoint_dir, f) for f in all_files
                if checkpoint_pattern in f and f.endswith('.pkl')
            ]
            
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
    
    def run_create_pipeline(self, input_filepath, output_filepath, batch_size=32):
        """Execute the complete pipeline with safety measures."""
        checkpoint_path = f"{output_filepath}.checkpoint"
        
        print(f"Starting vector database creation with checkpoint path: {checkpoint_path}")
        
        # Try to load from checkpoint first
        df, checkpoint_embeddings, checkpoint_idx = self.load_latest_checkpoint(checkpoint_path)
        
        # If no checkpoint, start from scratch
        if df is None:
            print("No checkpoint found. Starting from scratch...")
            df = self.load_and_preprocess_data(input_filepath)
            checkpoint_idx = 0
        
        # Get the texts to process
        texts = df['VECTOR_NAME'].tolist()
        print(f"Processing {len(texts)} product names")
        
        # Skip already processed texts if we have a checkpoint
        if checkpoint_embeddings is not None and checkpoint_idx > 0:
            print(f"Resuming from checkpoint with {checkpoint_idx} processed items.")
            # Start where we left off
            start_idx = checkpoint_idx
            # Initialize embeddings with the checkpoint data
            embedding_dim = self.embed_model.get_sentence_embedding_dimension()
            embeddings = np.zeros((len(texts), embedding_dim), dtype=np.float32)
            embeddings[:checkpoint_idx] = checkpoint_embeddings[:checkpoint_idx]
        else:
            # Start from the beginning
            start_idx = 0
            checkpoint_embeddings = None
        
        # Generate embeddings safely, potentially in chunks
        for embeddings_chunk, processed_count in self.generate_embeddings(
            texts[start_idx:], batch_size=batch_size
        ):
            # If we have a checkpoint, merge with new data
            if checkpoint_embeddings is not None:
                # Update only the new part
                checkpoint_embeddings[start_idx:start_idx+processed_count] = embeddings_chunk
                total_processed = start_idx + processed_count
                # Save a checkpoint with the combined data
                self.save_checkpoint(df, checkpoint_embeddings, checkpoint_path, total_processed)
            else:
                # No checkpoint, just save the new data
                self.save_checkpoint(df, embeddings_chunk, checkpoint_path, processed_count)
                checkpoint_embeddings = embeddings_chunk
        
        print(f"Embedding generation complete. Generated embeddings for {len(texts)} products.")
        print(f"Embeddings shape: {checkpoint_embeddings.shape}")
        
        # At this point, checkpoint_embeddings should contain all our embeddings
        # Save the final database safely
        print("Saving final vector database...")
        success = self.save_vector_database(df, checkpoint_embeddings, output_filepath)
        
        if success:
            print(f"Vector database creation completed successfully.")
            print(f"Final files created: {output_filepath}.pkl and {output_filepath}.index")
            return df, checkpoint_embeddings
        else:
            print("Failed to save vector database. Check the checkpoint files to recover data.")
            return None, None