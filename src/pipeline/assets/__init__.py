"""
Asset-specific implementations for the pipeline.

This module contains asset-specific processors and feature engineers
that inherit from the base classes and implement domain knowledge.
"""

from .watch import WatchProcessor, WatchFeatureEngineer

__all__ = ['WatchProcessor', 'WatchFeatureEngineer']