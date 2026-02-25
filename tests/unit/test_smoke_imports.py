#!/usr/bin/env python3
"""
Smoke test to verify that all modules can be imported successfully.
"""

import pytest


def test_core_import():
    """Test that core module can be imported."""
    import src.core
    assert src.core is not None


def test_ingestion_import():
    """Test that ingestion module can be imported."""
    import src.ingestion
    assert src.ingestion is not None


def test_libs_import():
    """Test that libs module can be imported."""
    import src.libs
    assert src.libs is not None


def test_observability_import():
    """Test that observability module can be imported."""
    import src.observability
    assert src.observability is not None


def test_mcp_server_import():
    """Test that mcp_server module can be imported."""
    import src.mcp_server
    assert src.mcp_server is not None


def test_main_import():
    """Test that main module can be imported."""
    import main
    assert main is not None