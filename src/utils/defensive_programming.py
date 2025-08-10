"""
Defensive Programming Utilities

Utility functions to handle common null-safety patterns and reduce code duplication
throughout the codebase.
"""

from typing import Any, List, Optional, Union
from loguru import logger


def safe_len(obj: Optional[Any]) -> int:
    """
    Calculate the length of an object safely, returning 0 if the object is None.
    
    Args:
        obj: The object to calculate length for (can be None)
        
    Returns:
        int: The length of the object, or 0 if None or if len() raises an exception
        
    Examples:
        >>> safe_len([1, 2, 3])
        3
        >>> safe_len(None)
        0
        >>> safe_len([])
        0
    """
    if obj is None:
        return 0
    
    try:
        return len(obj)
    except (TypeError, AttributeError):
        logger.warning(f"Cannot calculate length of object type {type(obj)}: {obj}")
        return 0


def ensure_list(obj: Optional[Any], default: Optional[List] = None) -> List:
    """
    Ensure that an object is a list, converting or defaulting as needed.
    
    Args:
        obj: The object to ensure is a list
        default: Default list to return if obj is None (defaults to empty list)
        
    Returns:
        List: A valid list object
        
    Examples:
        >>> ensure_list([1, 2, 3])
        [1, 2, 3]
        >>> ensure_list(None)
        []
        >>> ensure_list("not_a_list")
        []
        >>> ensure_list(None, ["default"])
        ['default']
    """
    if default is None:
        default = []
        
    if obj is None:
        return default
        
    if isinstance(obj, list):
        return obj
        
    # Log warning for unexpected types
    logger.warning(f"Expected list but got {type(obj)}: {obj}. Converting to empty list.")
    return default


def safe_dict_len(dict_obj: Optional[dict], key: str) -> int:
    """
    Safely get the length of a value from a dictionary.
    
    Args:
        dict_obj: The dictionary to access (can be None)
        key: The key to access in the dictionary
        
    Returns:
        int: The length of the value at the key, or 0 if None/missing
        
    Examples:
        >>> safe_dict_len({"files": [1, 2, 3]}, "files")
        3
        >>> safe_dict_len(None, "files")
        0
        >>> safe_dict_len({"files": None}, "files")
        0
    """
    if dict_obj is None:
        return 0
        
    value = dict_obj.get(key)
    return safe_len(value)


def validate_initialization(obj: Any, field_name: str, expected_type: type = list) -> bool:
    """
    Validate that a field has been properly initialized to the expected type.
    
    Args:
        obj: The object containing the field
        field_name: The name of the field to validate
        expected_type: The expected type of the field
        
    Returns:
        bool: True if the field is properly initialized, False otherwise
    """
    if not hasattr(obj, field_name):
        logger.error(f"Object {type(obj)} missing required field: {field_name}")
        return False
        
    field_value = getattr(obj, field_name)
    
    if field_value is None:
        logger.error(f"Field {field_name} in {type(obj)} is None after initialization")
        return False
        
    if not isinstance(field_value, expected_type):
        logger.error(f"Field {field_name} in {type(obj)} has incorrect type: {type(field_value)}, expected: {expected_type}")
        return False
        
    logger.debug(f"Field {field_name} in {type(obj)} properly initialized as {expected_type}")
    return True