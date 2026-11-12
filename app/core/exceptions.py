"""
Custom Exception Classes for Adaptive LLM Routing
"""

class RoutingException(Exception):
    """Base exception for routing errors."""
    pass

class ModelUnavailableException(RoutingException):
    """Raised when an inference model cannot be reached or loaded."""
    pass

class EscalationException(RoutingException):
    """Raised when cascading escalation encounters an unrecoverable state."""
    pass

class EvaluationException(RoutingException):
    """Raised when quality or performance evaluation fails."""
    pass

class ConfigurationException(RoutingException):
    """Raised when configuration file is invalid or missing required keys."""
    pass
