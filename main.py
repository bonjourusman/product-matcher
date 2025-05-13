import os
import sys
import json
from dotenv import load_dotenv, find_dotenv
from core import ProductMatcher

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

def main():
    # Load environment variables
    _ = load_dotenv(find_dotenv())
    
    # Ensure configuration files exist
    ensure_config_files()
    
    # Paths
    external_data_path = 'data/external.csv'
    vector_db_path = 'db/vector'
    output_path = 'out/output.csv'
    
    # Parameters - Vector DB
    top_k = 10

    # Parameters - LLM
    llm_model = "gpt-4.1"
    llm_batch_size = 10
    max_retries = 3
    
    # Match products with external data
    print(f"Matching products using vector database at {vector_db_path} with external data from {external_data_path}...")

    # Verify files exist
    if not os.path.exists(f"{vector_db_path}.pkl") or not os.path.exists(f"{vector_db_path}.index"):
        print(f"Error: Vector database files not found at {vector_db_path}")
        sys.exit(1)
        
    if not os.path.exists(external_data_path):
        print(f"Error: External data file not found at {external_data_path}")
        sys.exit(1)
    
    # Create directory for output if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Initialize and run matcher
    matcher = ProductMatcher(
        vector_db_filepath=vector_db_path,
        llm_model=llm_model,
        top_k=top_k,
        batch_size=llm_batch_size,
        max_retries=max_retries
    )
    matcher.run_match_pipeline(external_data_path, output_path)

    print(f"Product matching completed successfully!")
    
if __name__ == "__main__":
    main()