"""Core algorithms for trustworthy QA preference alignment."""

__all__ = [
    "CompletionLogProbs",
    "PreferenceLossOutput",
    "completion_logps_from_logits",
    "dpo_loss",
    "orpo_loss",
    "simpo_loss",
]


def __getattr__(name):
    if name in __all__:
        from . import losses

        return getattr(losses, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
