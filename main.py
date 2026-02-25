#!/usr/bin/env python3
"""
Main entry point for the Modular RAG MCP Server.
"""

import os
import sys
from typing import Optional


def load_settings(path: str = "config/settings.yaml") -> dict:
    """
    Load settings from YAML file.
    
    Args:
        path: Path to settings.yaml file
        
    Returns:
        dict: Loaded settings
    """
    try:
        import yaml
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading settings: {e}")
        return {}


def validate_settings(settings: dict) -> bool:
    """
    Validate that required settings are present.
    
    Args:
        settings: Loaded settings
        
    Returns:
        bool: True if settings are valid, False otherwise
    """
    required_sections = ['llm', 'embedding', 'vector_store']
    for section in required_sections:
        if section not in settings:
            print(f"Missing required section: {section}")
            return False
    return True


def main() -> int:
    """
    Main function.
    
    Returns:
        int: Exit code
    """
    print("Starting Modular RAG MCP Server...")
    
    # Load settings
    settings = load_settings()
    
    # Validate settings
    if not validate_settings(settings):
        print("Invalid settings. Please check config/settings.yaml")
        return 1
    
    # Test imports
    try:
        import src.core
        import src.ingestion
        import src.libs
        import src.observability
        import src.mcp_server
        print("✓ All modules imported successfully!")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return 1
    
    print("\nServer initialized successfully!")
    print("\nAvailable modules:")
    print("- src.core: Core functionality and types")
    print("- src.ingestion: Document ingestion pipeline")
    print("- src.libs: Pluggable backend implementations")
    print("- src.observability: Logging and tracing")
    print("- src.mcp_server: MCP server implementation")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())