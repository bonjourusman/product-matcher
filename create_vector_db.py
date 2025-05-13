import gc
import os
import sys
import json
import signal
import traceback
from dotenv import load_dotenv, find_dotenv
from core import VectorDBCreator
import multiprocessing
import torch

# Add signal handler for graceful shutdown
def signal_handler(sig, frame):
    """Handle termination signals gracefully."""
    print("\nReceived termination signal. Cleaning up...")
    # Force garbage collection
    gc.collect()
    
    # Clean up torch resources if available
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass
    
    print("Resources cleaned up. Exiting.")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def ensure_config_files():
    """Ensure configuration files exist."""
    # Create directories if they don't exist
    os.makedirs('config', exist_ok=True)
    
    # Default UOM mappings
    uom_dict = {
        "oz": "ounce",
        "lbs": "pound",
        "lb": "pound",
        "pk": "pack",
        "ct": "count",
        "ltr": "liter",
        "gal": "gallon",
        "g": "gram",
        "gram": "gram",
        "pt": "pint",
        "qt": "quart",
        "pc": "piece",
        "piec": "piece",
        "ml": "milliliter",
        "stk": "stack",
        "kg": "kilogram"
    }
    
    # Default abbreviation mappings
    abbr_dict = {
        "w/": " with ",
        " & ": " and ",
        "%": " percent "
    }
    
    # Create UOM mappings file if it doesn't exist
    uom_path = 'config/uom_mappings.json'
    if not os.path.exists(uom_path):
        with open(uom_path, 'w') as f:
            json.dump(uom_dict, f, indent=2)
    
    # Create abbreviation mappings file if it doesn't exist
    abbr_path = 'config/abbr_mappings.json'
    if not os.path.exists(abbr_path):
        with open(abbr_path, 'w') as f:
            json.dump(abbr_dict, f, indent=2)

def create_vector_db():
    # Set multiprocessing start method to 'spawn' for better stability
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, ignore
        pass
    
    # Load environment variables
    _ = load_dotenv(find_dotenv())
    
    # Ensure configuration files exist
    ensure_config_files()
    
    # Check command line arguments
    if len(sys.argv) < 3:
        print("Usage: python create_vector_db.py <internal_data_path> <vector_db_path> [embed_model_name] [batch_size]")
        sys.exit(1)
    
    # Get paths from command line
    internal_data_path = sys.argv[1]
    vector_db_path = sys.argv[2]
    
    # Get optional embedding model name and batch size
    embed_model_name = "all-MiniLM-L6-v2"
    batch_size = 32  # Use a smaller default batch size for stability
    
    if len(sys.argv) > 3:
        embed_model_name = sys.argv[3]
    if len(sys.argv) > 4:
        try:
            batch_size = int(sys.argv[4])
        except ValueError:
            print(f"Warning: Invalid batch size '{sys.argv[4]}', using default of {batch_size}")
    
    # Verify file exists
    if not os.path.exists(internal_data_path):
        print(f"Error: Internal data file not found at {internal_data_path}")
        sys.exit(1)
    
    # Create directory for vector database if it doesn't exist
    os.makedirs(os.path.dirname(vector_db_path), exist_ok=True)
    
    try:
        # Initialize and run creator with safer pipeline
        print(f"Creating vector database at {vector_db_path} with data from {internal_data_path}...")
        print(f"Using embedding model: {embed_model_name}, batch size: {batch_size}")
        
        creator = VectorDBCreator(embed_model_name=embed_model_name)
        df, embeddings = creator.run_create_pipeline(
            internal_data_path, 
            vector_db_path,
            batch_size=batch_size
        )
        
        if df is not None:
            print(f"Vector database created successfully! Contains {len(df)} products.")
        else:
            print("Vector database creation failed. Check logs for details.")
            sys.exit(1)
    
    except Exception as e:
        print(f"Error creating vector database: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Force cleanup
        gc.collect()
        
        # Clean up torch resources if available
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

if __name__ == "__main__":
    create_vector_db()