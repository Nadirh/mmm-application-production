"""
Custom exception classes for MMM application.
"""


class MMMException(Exception):
    """Base exception for MMM application."""
    pass


class DataValidationError(MMMException):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, errors: list = None):
        super().__init__(message)
        self.errors = errors or []


class ModelTrainingError(MMMException):
    """Raised when model training fails."""
    pass


class OptimizationError(MMMException):
    """Raised when optimization fails."""
    pass


class ConfigurationError(MMMException):
    """Raised when configuration is invalid."""
    pass


class FileProcessingError(MMMException):
    """Raised when file processing fails."""
    pass


class InsufficientDataError(MMMException):
    """Raised when dataset doesn't meet minimum requirements."""
    pass


class ModelNotFittedError(MMMException):
    """Raised when trying to use an unfitted model."""
    pass


class ParameterValidationError(MMMException):
    """Raised when model parameters fail validation."""
    pass