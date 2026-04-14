"""
Common Exceptions for Multi-Agent Literature Review System
通用异常类定义
"""


class LiteratureReviewException(Exception):
    """Base exception for all literature review system errors"""
    pass


class WorkspaceError(LiteratureReviewException):
    """Workspace related errors"""
    pass


class AgentError(LiteratureReviewException):
    """Agent execution errors"""
    pass


class SearchError(AgentError):
    """Search agent specific errors"""
    pass


class ScreeningError(AgentError):
    """Screening agent specific errors"""
    pass


class ClusteringError(AgentError):
    """Clustering agent specific errors"""
    pass


class SummaryError(AgentError):
    """Summary agent specific errors"""
    pass


class LLMError(LiteratureReviewException):
    """LLM invocation errors"""
    pass


class APIError(LiteratureReviewException):
    """External API call errors"""
    pass


class ValidationError(LiteratureReviewException):
    """Input validation errors"""
    pass


class ConfigurationError(LiteratureReviewException):
    """Configuration errors"""
    pass