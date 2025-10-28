# ===========================================
# Components Package
# ===========================================

from .LLMComponent import (
    LLMComponent,
    LLMConfig,
    LoggingMiddleware,
    RetryMiddleware,
    create_fast_llm_config,
    create_creative_llm_config,
    create_analytical_llm_config
)

from .ContextComponent import (
    ContextComponent,
    ContextConfig,
    ContextLoggingMiddleware,
    create_chatbot_config,
    create_teacher_config,
    create_enterprise_config
)

from .OutputComponent import (
    OutputComponent,
    OutputConfig,
    console_output_callback
)

try:
    # Import ComponentTestHarness from main module
    from ..main import ComponentTestHarness
    test_harness_imported = True
except ImportError:
    test_harness_imported = False

__all__ = [
    # LLM
    'LLMComponent', 'LLMConfig',
    'LoggingMiddleware', 'RetryMiddleware',
    'create_fast_llm_config', 'create_creative_llm_config', 'create_analytical_llm_config',

    # Context
    'ContextComponent', 'ContextConfig',
    'ContextLoggingMiddleware',
    'create_chatbot_config', 'create_teacher_config', 'create_enterprise_config',

    # Output
    'OutputComponent', 'OutputConfig',
    'console_output_callback',
]

if test_harness_imported:
    __all__.append('ComponentTestHarness')
