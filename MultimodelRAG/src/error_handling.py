"""
Error handling utilities for the MultimodalRAG system.
Provides standardized error handling and logging.
"""

import logging
import traceback
import sys
import os
from typing import Dict, Any, Optional, Callable, TypeVar, Union
from functools import wraps
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'multimodal_rag.log')),
        logging.StreamHandler(sys.stdout)
    ]
)

# Create logger
logger = logging.getLogger('multimodal_rag')

# Define error types
class MultimodalRAGError(Exception):
    """Base exception class for MultimodalRAG errors"""
    def __init__(self, message: str, error_code: str = "GENERAL_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

class DocumentProcessingError(MultimodalRAGError):
    """Exception raised for errors during document processing"""
    def __init__(self, message: str, document_id: Optional[str] = None):
        self.document_id = document_id
        error_code = "DOC_PROCESSING_ERROR"
        super().__init__(message, error_code)

class ImageProcessingError(MultimodalRAGError):
    """Exception raised for