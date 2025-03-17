from typing import Any


class RefException(Exception):
    """Base class for exceptions related to REF operations"""

    pass


class InvalidExecutorException(RefException):
    """Exception raised when an invalid executor is used"""

    def __init__(self, executor: Any, message: str) -> None:
        message = f"Invalid executor: '{executor}'\n {message}"

        super().__init__(message)


class InvalidProviderException(RefException):
    """Exception raised when an invalid metric is registered"""

    def __init__(self, provider: Any, message: str) -> None:
        message = f"Invalid provider: '{provider}'\n {message}"

        super().__init__(message)


class InvalidMetricException(RefException):
    """Exception raised when an invalid metric is registered"""

    def __init__(self, metric: Any, message: str) -> None:
        message = f"Invalid metric: '{metric}'\n {message}"

        super().__init__(message)


class ConstraintNotSatisfied(RefException):
    """Exception raised when a constraint is violated"""

    # TODO: implement when we have agreed on using constraints


class ResultValidationError(RefException):
    """Exception raised when the results from a metric execution are invalid"""
