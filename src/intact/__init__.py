"""intact: Qwen3-4B reward-hacking pipeline: prompts → sampling → grading → activations.

The consumer API is `intact.core`, re-exported here: `load_rollouts`, `load_activations`, `list_activations`,
`load_prompts`, `concat`, `Activations`, `IntegrityError`.
"""

from intact.core import (
    Activations,
    IntegrityError,
    concat,
    list_activations,
    load_activations,
    load_prompts,
    load_rollouts,
)

__all__ = ["Activations", "IntegrityError", "concat", "list_activations", "load_activations", "load_prompts", "load_rollouts"]
