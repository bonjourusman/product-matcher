import os
import sys
import json
import signal
import traceback
import gc
import multiprocessing
from dotenv import load_dotenv, find_dotenv
from core.vector_db_updater import VectorDBUpdater

# Add signal handler for graceful shutdown
def signal_handler(sig, frame):
    """Handle termination signals gracefully."""
    print("\nReceived termination signal. Cleaning up...")
    # Force garbage collection
    gc.collect()
    
    # Clean up torch resources if available
    try:
        import torch
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

def update_vector_db():
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
        print("Usage: python update_vector_db.py <vector_db_path> <new_data_path> [name_column] [batch_size]")
        sys.exit(1)
    
    # Get paths from command line
    vector_db_path = sys.argv[1]
    new_data_path = sys.argv[2]
    
    # Get optional parameters
    name_column = 'LONG_NAME'
    batch_size = 16  # Use a smaller default batch size for stability
    
    if len(sys.argv) > 3:
        name_column = sys.argv[3]
    if len(sys.argv) > 4:
        try:
            batch_size = int(sys.argv[4])
        except ValueError:
            print(f"Warning: Invalid batch size '{sys.argv[4]}', using default of {batch_size}")
    
    # Verify files exist
    if not os.path.exists(f"{vector_db_path}.pkl") or not os.path.exists(f"{vector_db_path}.index"):
        print(f"Error: Vector database files not found at {vector_db_path}")
        sys.exit(1)
        
    if not os.path.exists(new_data_path):
        print(f"Error: New data file not found at {new_data_path}")
        sys.exit(1)
    
    try:
        # Initialize and run updater
        print(f"Updating vector database at {vector_db_path} with new data from {new_data_path}...")
        print(f"Using name column: {name_column}, batch size: {batch_size}")
        
        updater = VectorDBUpdater(vector_db_path)
        updated_df, updated_embeddings, updated_index = updater.run_update_pipeline(
            new_data_path, 
            name_column=name_column, 
            batch_size=batch_size
        )
        
        if updated_df is not None:
            print(f"Update completed successfully! Vector database now contains {len(updated_df)} products.")
        else:
            print("Vector database update failed. Check logs for details.")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error updating vector database: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Force cleanup
        gc.collect()
        
        # Clean up torch resources if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

if __name__ == "__main__":
    update_vector_db()