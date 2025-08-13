"""
Project analyzers package for Generic Q&A Agent.

This package contains analyzers that follow the EventFlowAnalyzer pattern
to analyze different aspects of project repositories including architecture,
business capabilities, API endpoints, data models, and operations.
"""

from .architecture_detector import ArchitectureDetector, ArchitecturePattern
from .business_capability_analyzer import BusinessCapabilityAnalyzer
from .api_endpoint_analyzer import APIEndpointAnalyzer
from .data_model_analyzer import DataModelAnalyzer
from .operational_analyzer import OperationalAnalyzer

__all__ = [
    "ArchitectureDetector",
    "ArchitecturePattern", 
    "BusinessCapabilityAnalyzer",
    "APIEndpointAnalyzer",
    "DataModelAnalyzer",
    "OperationalAnalyzer"
]