from enum import Enum

class Decision(Enum):
    NO_ACTION = "NO_ACTION"
    AUTO_HEAL = "AUTO_HEAL"
    HITL_REQUIRED = "HITL_REQUIRED"


def decide_action(drift_score: float, avg_confidence: float) -> Decision:
    """
    Determines system action based on risk signals.
    No side effects. No UI. No execution.
    """
    if drift_score > 0.3 and avg_confidence < 0.6:
        return Decision.HITL_REQUIRED
    elif drift_score > 0.25:
        return Decision.AUTO_HEAL
    else:
        return Decision.NO_ACTION
