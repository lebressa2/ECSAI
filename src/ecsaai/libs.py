# ===========================================
# Utility Libraries
# ===========================================

"""
Basic utility libraries for the framework
"""

import logging
import os
from typing import Optional, Dict, Any

# ===========================================
# Logger Utility
# ===========================================

class Logger:
    """Simple logging utility"""

    @staticmethod
    def get_logger(name: str):
        """Get a configured logger"""
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

# ===========================================
# LLM Factory (Mock)
# ===========================================

class LLMFactory:
    """Factory for creating LLM instances"""

    def __init__(self):
        self.providers = {}

    def create_llm(self, provider: str, **kwargs):
        """Create an LLM instance"""
        # Mock implementation - would integrate with actual LLM libraries
        return MockLLM(provider, **kwargs)

class MockLLM:
    """Mock LLM for demonstration"""

    def __init__(self, provider: str, **kwargs):
        self.provider = provider
        self.config = kwargs

    async def generate(self, prompt: str) -> str:
        """Generate a mock response"""
        return f"Mock {self.provider} response to: {prompt[:50]}..."

# ===========================================
# Utils
# ===========================================

class Utils:
    """General utility functions"""

    @staticmethod
    def get_env_var(key: str, default: Optional[str] = None) -> str:
        """Get environment variable with fallback"""
        return os.getenv(key, default or "")

    @staticmethod
    def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple dictionaries"""
        result = {}
        for d in dicts:
            result.update(d)
        return result

# Export utilities
logger = Logger.get_logger("ecsaai")
factory = LLMFactory()

__all__ = ['Logger', 'LLMFactory', 'MockLLM', 'Utils', 'logger', 'factory']
