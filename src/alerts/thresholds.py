from dataclasses import dataclass

# Remaining Useful Life thresholds, in cycles
RUL_WARNING_CYCLES = 60
RUL_CRITICAL_CYCLES = 20

# Failure-window classifier probability thresholds
FAILURE_PROB_WARNING = 0.4
FAILURE_PROB_CRITICAL = 0.7

RISK_ORDER = ["OK", "WATCH", "WARNING", "CRITICAL"]


@dataclass
class RiskAssessment:
    risk_state: str
    reason: str


def assess_risk(predicted_rul: float, failure_probability: float, is_anomaly: bool) -> RiskAssessment:
    if predicted_rul <= RUL_CRITICAL_CYCLES or failure_probability >= FAILURE_PROB_CRITICAL:
        state, reason = "CRITICAL", "RUL or failure probability past critical threshold"
    elif predicted_rul <= RUL_WARNING_CYCLES or failure_probability >= FAILURE_PROB_WARNING:
        state, reason = "WARNING", "RUL or failure probability past warning threshold"
    elif is_anomaly:
        state, reason = "WATCH", "Sensor pattern flagged as anomalous, no threshold breach yet"
    else:
        state, reason = "OK", "Within normal operating range"

    if is_anomaly and state == "OK":
        state, reason = "WATCH", "Sensor pattern flagged as anomalous, no threshold breach yet"

    return RiskAssessment(risk_state=state, reason=reason)
