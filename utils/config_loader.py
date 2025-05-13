import os
import json

def load_json_config(filepath):
    """
    Load a JSON configuration file.
    
    Args:
        filepath (str): Path to the JSON file
        
    Returns:
        dict: Dictionary containing the configuration
    """
    try:
        with open(filepath, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Warning: Configuration file {filepath} not found. Using empty dictionary.")
        return {}
    except json.JSONDecodeError:
        print(f"Warning: Error parsing {filepath}. Using empty dictionary.")
        return {}


def get_config_path(filename):
    """
    Get the absolute path to a configuration file.
    
    Args:
        filename (str): Name of the configuration file
        
    Returns:
        str: Absolute path to the configuration file
    """
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate to the config directory
    config_dir = os.path.join(os.path.dirname(current_dir), 'config')
    
    # Return the full path to the config file
    return os.path.join(config_dir, filename)


def get_uom_dict():
    """
    Get the unit of measure mapping dictionary.
    
    Returns:
        dict: Unit of measure mapping dictionary
    """
    return load_json_config(get_config_path('uom_mappings.json'))


def get_abbr_dict():
    """
    Get the abbreviation mapping dictionary.
    
    Returns:
        dict: Abbreviation mapping dictionary
    """
    return load_json_config(get_config_path('abbr_mappings.json'))