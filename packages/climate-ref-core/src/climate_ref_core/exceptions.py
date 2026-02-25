"""
Custom exceptions
"""

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
    """Exception raised when an invalid diagnostic is registered"""

    def __init__(self, provider: Any, message: str) -> None:
        message = f"Invalid provider: '{provider}'\n {message}"

        super().__init__(message)


class InvalidDiagnosticException(RefException):
    """Exception raised when an invalid diagnostic is registered"""

    def __init__(self, metric: Any, message: str) -> None:
        message = f"Invalid diagnostic: '{metric}'\n {message}"

        super().__init__(message)


class ConstraintNotSatisfied(RefException):
    """Exception raised when a constraint is violated"""

    # TODO: implement when we have agreed on using constraints


class ResultValidationError(RefException):
    """Exception raised when the executions from a diagnostic are invalid"""


class ExecutionError(RefException):
    """Exception raised when an execution fails"""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class DiagnosticError(RefException):
    """Error from diagnostic computing"""

    def __init__(self, message: str, result: Any):
        super().__init__(message)
        self.message = message
        self.result = result

    # need for serialization of parsl
    def __reduce__(self) -> tuple[type["DiagnosticError"], tuple[str, Any]]:
        # Return a tuple: (callable, args_tuple_for_reconstruction)
        return (self.__class__, (self.message, self.result))


class TestCaseError(RefException):
    """Raised when there is an error with a test case."""

    pass


class TestCaseNotFoundError(TestCaseError):
    """Raised when a test case is not found."""

    pass


class NoTestDataSpecError(TestCaseError):
    """Raised when a diagnostic has no test_data_spec."""

    pass


class DatasetResolutionError(TestCaseError):
    """Raised when datasets cannot be resolved for a test case."""

    pass
