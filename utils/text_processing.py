import re
import pandas as pd

def normalize_units(text, map_dict):
    """
    Replace all units of measure in the text with their long form.
    
    Args:
        text (str): Text to normalize
        map_dict (dict): Dictionary mapping unit abbreviations to their full forms
        
    Returns:
        str: Normalized text
    """
    # Sort units by length (longest first) to avoid substring matches
    units_by_length = sorted(map_dict.keys(), key=len, reverse=True)
    
    # First pass: Find all numeric values
    value_pattern = r'(\.?\d+\.?\d*)\.?'
    
    # Keep track of positions and replacements
    replacements = []
    
    # Find all numeric values in the text
    for value_match in re.finditer(value_pattern, text):
        value = value_match.group(1)
        start_pos = value_match.end()
        remaining_text = text[start_pos:]
        
        # Check for optional space followed by a unit
        space_match = re.match(r'^(\s*)', remaining_text)
        space = space_match.group(1) if space_match else ""
        after_space = start_pos + len(space)
        
        # Look for a unit after the number and optional space
        for unit in units_by_length:
            
            # Case 1: Unit with a period, e.g. 'oz.'
            period_unit = unit + "."
            if remaining_text[len(space):].lower().startswith(period_unit.lower()):
                unit_end = after_space + len(period_unit)
                long_form = map_dict[unit]
                
                # Format with a space for readability
                replacements.append((
                    value_match.start(),
                    unit_end,
                    f"{float(value):.2f} {long_form}"
                ))
                break
                
            # Case 2: Unit in plural form, e.g. 'kgs'
            plural_unit = unit + "s"
            if remaining_text[len(space):].lower().startswith(plural_unit.lower()):
                unit_end = after_space + len(plural_unit)
                long_form = map_dict[unit]
                
                # Format with a space for readability
                replacements.append((
                    value_match.start(),
                    unit_end,
                    f"{float(value):.2f} {long_form}"
                ))
                break
            
            # Case 3: Unit without a period or in singular form, e.g. 'oz', 'kg'
            if remaining_text[len(space):].lower().startswith(unit.lower()):
                unit_end = after_space + len(unit)
                long_form = map_dict[unit]
                
                # Format with a space for readability
                replacements.append((
                    value_match.start(),
                    unit_end,
                    f"{float(value):.2f} {long_form}"
                ))
                break

    # Apply replacements in reverse order to avoid position shifts
    result = text
    for start, end, replacement in sorted(replacements, key=lambda x: x[0], reverse=True):
        result = result[:start] + replacement + result[end:]
    
    return result


def standardize_text(text, uom_map_dict, abbr_map_dict):
    """
    Clean and standardize product text.
    
    Args:
        text (str): Text to standardize
        uom_map_dict (dict): Unit of measure mapping dictionary
        abbr_map_dict (dict): Abbreviation mapping dictionary
        
    Returns:
        str: Standardized text
    """
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()

    # Convert units of measure to standard long form
    text = normalize_units(text, uom_map_dict)
    
    for abbr, full in abbr_map_dict.items():
        text = text.replace(abbr, full)
    
    # Remove special characters but keep word characters, spaces, hyphens, period
    text = re.sub(r'[^\w\s\.\'&]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def extract_size_info(product_name, uom_dict):
    """
    Extract size information from product name.
    
    Args:
        product_name (str): Product name
        uom_dict (dict): Unit of measure mapping dictionary
        
    Returns:
        tuple or None: Tuple of (value, unit) if size info found, None otherwise
    """
    if pd.isna(product_name):
        return None

    units = {v for v in uom_dict.values()}
    units_pattern = '|'.join(units)
    
    # Extract size (numeric value + unit)
    size_pattern = rf'(\d+\.?\d*)\s*({units_pattern})'
    size_matches = re.findall(size_pattern, product_name.lower())
    
    if not size_matches:
        return None
    
    # Return as tuple of (value, unit)
    return float(size_matches[0][0]), size_matches[0][1]