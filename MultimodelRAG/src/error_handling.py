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
    """Exception raised for errors during image processing"""
    def __init__(self, message: str, image_id: Optional[str] = None):
        self.image_id = image_id
        error_code = "IMG_PROCESSING_ERROR"
        super().__init__(message, error_code)

class EmbeddingError(MultimodalRAGError):
    """Exception raised for errors during embedding generation"""
    def __init__(self, message: str, model_name: Optional[str] = None):
        self.model_name = model_name
        error_code = "EMBEDDING_ERROR"
        super().__init__(message, error_code)

class VectorDBError(MultimodalRAGError):
    """Exception raised for errors related to vector database operations"""
    def __init__(self, message: str, operation: Optional[str] = None):
        self.operation = operation
        error_code = "VECTOR_DB_ERROR"
        super().__init__(message, error_code)

class LLMError(MultimodalRAGError):
    """Exception raised for errors related to LLM interactions"""
    def __init__(self, message: str, model_name: Optional[str] = None):
        self.model_name = model_name
        error_code = "LLM_ERROR"
        super().__init__(message, error_code)

class APIError(MultimodalRAGError):
    """Exception raised for API-related errors"""
    def __init__(self, message: str, status_code: int = 500):
        self.status_code = status_code
        error_code = f"API_ERROR_{status_code}"
        super().__init__(message, error_code)

# Type variable for function return type
T = TypeVar('T')

# Decorator for error handling
def handle_errors(
    default_return: Optional[Any] = None,
    raise_exception: bool = False,
    log_level: int = logging.ERROR
) -> Callable[[Callable[..., T]], Callable[..., Union[T, Any]]]:
    """
    Decorator for standardized error handling.
    
    Args:
        default_return: Value to return on error if raise_exception is False
        raise_exception: Whether to re-raise the exception
        log_level: Logging level for errors
    
    Returns:
        Decorated function with error handling
    """
    def decorator(func: Callable[..., T]) -> Callable[..., Union[T, Any]]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Union[T, Any]:
            try:
                return func(*args, **kwargs)
            except MultimodalRAGError as e:
                # Log specific error with details
                logger.log(
                    log_level,
                    f"{e.__class__.__name__} in {func.__name__}: {e.message} "
                    f"[{e.error_code}] - {traceback.format_exc()}"
                )
                if raise_exception:
                    raise
                return default_return
            except Exception as e:
                # Log unexpected error
                logger.log(
                    log_level,
                    f"Unexpected error in {func.__name__}: {str(e)} - {traceback.format_exc()}"
                )
                if raise_exception:
                    raise MultimodalRAGError(f"Unexpected error in {func.__name__}: {str(e)}")
                return default_return
        return wrapper
    return decorator

# Retry decorator
def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for retrying operations that may fail temporarily.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Factor by which to increase delay between retries
        exceptions: Tuple of exceptions to catch and retry
    
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            attempt = 1
            current_delay = delay
            
            while attempt <= max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        logger.error(
                            f"Failed after {max_attempts} attempts: {func.__name__} - {str(e)}"
                        )
                        raise
                    
                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed: {func.__name__} - {str(e)}. "
                        f"Retrying in {current_delay:.2f}s"
                    )
                    
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
                    attempt += 1
        
        return wrapper
    return decorator

# Function to format error response for API
def format_error_response(error: Exception) -> Dict[str, Any]:
    """
    Format an exception into a standardized API error response.
    
    Args:
        error: The exception to format
    
    Returns:
        Dictionary with error details for API response
    """
    if isinstance(error, MultimodalRAGError):
        return {
            "success": False,
            "error": {
                "code": error.error_code,
                "message": error.message,
                "type": error.__class__.__name__
            }
        }
    else:
        return {
            "success": False,
            "error": {
                "code": "UNKNOWN_ERROR",
                "message": str(error),
                "type": error.__class__.__name__
            }
        }

# Function to log API requests
def log_api_request(
    endpoint: str,
    method: str,
    request_data: Optional[Dict[str, Any]] = None,
    response_status: Optional[int] = None,
    duration_ms: Optional[float] = None
) -> None:
    """
    Log API request details.
    
    Args:
        endpoint: API endpoint
        method: HTTP method
        request_data: Request data (sanitized)
        response_status: HTTP response status code
        duration_ms: Request duration in milliseconds
    """
    # Sanitize request data to remove sensitive information
    sanitized_data = None
    if request_data:
        sanitized_data = request_data.copy()
        # Remove sensitive fields
        for key in ["api_key", "password", "token"]:
            if key in sanitized_data:
                sanitized_data[key] = "***REDACTED***"
    
    log_data = {
        "endpoint": endpoint,
        "method": method,
        "request": sanitized_data,
        "status": response_status,
        "duration_ms": duration_ms
    }
    
    if response_status and response_status >= 400:
        logger.error(f"API Error: {log_data}")
    else:
        logger.info(f"API Request: {log_data}")
