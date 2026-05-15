"""
Diagnostic translator: SHAP value patterns → operator-readable diagnoses.

The supervised LightGBM classifier produces a failure probability and feature
attributions (SHAP values) for each prediction. SHAP outputs are interpretable
to ML practitioners but not to maintenance technicians on the factory floor.
This module translates the most-significant SHAP patterns into concise
diagnoses with recommended actions.

The translation is rule-based and pattern-driven:
  1. Each pattern checks for a specific configuration of dominant SHAP
     features combined with raw input thresholds
  2. Patterns are tested in order of physical specificity (most specific first)
  3. The first matching pattern produces the diagnosis
  4. A general fallback applies when no specific pattern fits

The pattern definitions are calibrated against the physical meaning of the
features in the AI4I 2020 dataset (heat dissipation, tool wear, power
output, etc.). In a production deployment, these patterns would be refined
against historical maintenance records and operator feedback rather than
inferred purely from feature physics.

Usage:
    from app.diagnostic_translator import translate_shap, Diagnosis

    diagnosis = translate_shap(
        shap_values={"Temp_Delta": 0.4, "Power_W": 0.1, ...},
        raw_inputs={"Temp_Delta": 12.0, "Power_W": 6500, ...},
        probability=0.65,
    )
    print(diagnosis.primary_diagnosis)
    print(diagnosis.recommended_action)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# A SHAP value is "significant" if its magnitude exceeds this fraction of
# typical model outputs. Calibrated empirically against AI4I 2020 model
# behaviour. SHAP values in tree models typically fall in [-1, 1] for this
# kind of binary classifier.
SHAP_SIGNIFICANT = 0.05
SHAP_DOMINANT = 0.15  # roughly 3x significant

# Risk category boundaries on predicted probability.
# Calibrated against the cost-optimal threshold from 06_cost_analysis.ipynb.
# The empirical cost-optimal cut-point is approximately 0.01 (the theoretical
# Bayes-optimal is 0.059; empirical lands lower due to under-prediction in
# the model's low-probability tail).
#
# HEALTHY  (<0.01):    Below the cost-optimal cut-point. No action needed.
# ADVISORY (0.01-0.10): Above the cut-point but absolute risk modest.
#                       Log for engineer review; schedule inspection
#                       at next convenient window.
# WARNING  (0.10-0.50): Meaningful elevation above the cut-point.
#                       Address proactively; do not defer past current shift.
# CRITICAL (>=0.50):   Strong evidence of imminent failure. Stop operation,
#                      investigate immediately.
PROB_ADVISORY = 0.01     # below this is Healthy
PROB_WARNING = 0.10      # at/above this is Warning
PROB_CRITICAL = 0.50     # at/above this is Critical

# Stall zone definition (from EDA in 03_data_visualisation.ipynb)
STALL_SPEED_THRESHOLD = 1400    # RPM below this with high torque = stall
STALL_TORQUE_THRESHOLD = 60     # Nm above this at low speed = stall

# Tool wear concern threshold (minutes). AI4I synthetic data has Tool Wear
# Failures clustering above 200 minutes; we set the watch level lower to
# enable preventive scheduling rather than reactive replacement.
TOOL_WEAR_CONCERN = 180


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class RiskCategory(str, Enum):
    """Risk severity buckets for the operator dashboard.

    Naming follows process-control conventions (advisory / warning / critical)
    rather than colloquial English. Operators trained on DCS or SCADA systems
    will recognise this hierarchy.
    """

    HEALTHY = "Healthy"
    ADVISORY = "Advisory"
    WARNING = "Warning"
    CRITICAL = "Critical"


@dataclass
class Diagnosis:
    """Operator-readable diagnosis for a single machine prediction.

    Attributes:
        risk_category: One of HEALTHY/WATCH/HIGH/CRITICAL.
        risk_probability: Predicted failure probability (0.0 to 1.0).
        primary_diagnosis: Single-sentence summary for the operator UI.
        contributing_factors: List of specific factors driving the diagnosis,
            each one human-readable.
        recommended_action: Suggested next step for the operator.
        pattern_id: Identifier of the matched pattern (for debugging /
            traceability). Useful when refining patterns against feedback.
    """

    risk_category: RiskCategory
    risk_probability: float
    primary_diagnosis: str
    contributing_factors: list[str] = field(default_factory=list)
    recommended_action: str = ""
    pattern_id: str = ""

    def __str__(self) -> str:
        """Human-readable rendering for terminal/log output."""
        lines = [f"[{self.risk_category.value}] {self.primary_diagnosis}"]
        if self.contributing_factors:
            lines.append("Contributing factors:")
            for factor in self.contributing_factors:
                lines.append(f"  - {factor}")
        if self.recommended_action:
            lines.append(f"Action: {self.recommended_action}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core translation logic
# ---------------------------------------------------------------------------

def risk_category_from_probability(prob: float) -> RiskCategory:
    """Map a failure probability to a risk category."""
    if prob >= PROB_CRITICAL:
        return RiskCategory.CRITICAL
    if prob >= PROB_WARNING:
        return RiskCategory.WARNING
    if prob >= PROB_ADVISORY:
        return RiskCategory.ADVISORY
    return RiskCategory.HEALTHY


def translate_shap(
    shap_values: dict[str, float],
    raw_inputs: dict[str, float],
    probability: float,
) -> Diagnosis:
    """Translate SHAP feature attributions to an operator-readable diagnosis.

    Args:
        shap_values: Dict mapping feature name to SHAP value. Positive values
            push the prediction toward failure; negative toward healthy.
        raw_inputs: Dict mapping feature name to the raw input value (e.g.
            Temp_Delta in Kelvin, Tool_Wear in minutes). Used to provide
            concrete numbers in the diagnosis text.
        probability: The model's predicted failure probability (0.0 to 1.0).

    Returns:
        Diagnosis object summarising the prediction for the operator.
    """
    risk_category = risk_category_from_probability(probability)

    # Healthy state: short-circuit with monitoring guidance
    if risk_category == RiskCategory.HEALTHY:
        return Diagnosis(
            risk_category=risk_category,
            risk_probability=probability,
            primary_diagnosis="Machine operating within normal parameters.",
            contributing_factors=[],
            recommended_action="Continue routine monitoring.",
            pattern_id="healthy_default",
        )

    # Pattern matching proceeds in order of physical specificity.
    # First match wins; ordering matters.

    # Pattern 1: Heat dissipation failure
    # SHAP dominated by Temp_Delta contribution; raw Temp_Delta also elevated
    temp_delta_shap = shap_values.get("Temp_Delta", 0.0)
    temp_delta_raw = raw_inputs.get("Temp_Delta", 0.0)
    if temp_delta_shap > SHAP_DOMINANT and temp_delta_raw > 10.0:
        return Diagnosis(
            risk_category=risk_category,
            risk_probability=probability,
            primary_diagnosis=(
                "Heat dissipation pattern detected — cooling system check recommended."
            ),
            contributing_factors=[
                f"Temperature differential elevated: Process − Air = {temp_delta_raw:.1f} K",
                "Heat is not being removed from the cutting zone effectively.",
            ],
            recommended_action=(
                "Inspect spindle cooling system and coolant circulation. "
                "Check coolant flow rate, coolant temperature at the nozzle, "
                "and enclosure ventilation. Verify that the spindle motor "
                "cooling fan is unobstructed."
            ),
            pattern_id="heat_dissipation",
        )

    # Pattern 2: Tool wear failure
    # SHAP dominated by Tool_Wear; raw Tool_Wear in concern range
    tool_wear_shap = shap_values.get("Tool_Wear", 0.0)
    tool_wear_raw = raw_inputs.get("Tool_Wear", 0.0)
    if tool_wear_shap > SHAP_DOMINANT and tool_wear_raw > TOOL_WEAR_CONCERN:
        return Diagnosis(
            risk_category=risk_category,
            risk_probability=probability,
            primary_diagnosis=(
                "Tool wear approaching end of useful life."
            ),
            contributing_factors=[
                f"Tool wear at {tool_wear_raw:.0f} minutes (concern threshold {TOOL_WEAR_CONCERN}).",
                "Continued operation risks tool breakage, surface finish degradation, "
                "and secondary damage to the workpiece or spindle.",
            ],
            recommended_action=(
                "Schedule tool replacement at next planned maintenance window. "
                "Before next cycle, inspect cutting edge for chipping, fracture, "
                "or built-up edge. Verify recent parts meet dimensional and "
                "surface-finish specifications. Do not extend the tool beyond "
                "its specification."
            ),
            pattern_id="tool_wear",
        )

    # Pattern 3: Stall zone operation
    # Distinctive operating regime: low rotational speed + high torque
    # Discovered in the EDA notebook (03_data_visualisation.ipynb).
    # This pattern produces both mechanical strain and reduced cooling.
    speed_raw = raw_inputs.get("Rotational_Speed", 1500.0)
    torque_raw = raw_inputs.get("Torque", 40.0)
    if speed_raw < STALL_SPEED_THRESHOLD and torque_raw > STALL_TORQUE_THRESHOLD:
        return Diagnosis(
            risk_category=risk_category,
            risk_probability=probability,
            primary_diagnosis=(
                "Stall-zone operation — high torque at low rotational speed."
            ),
            contributing_factors=[
                f"Operating at {speed_raw:.0f} RPM with {torque_raw:.1f} Nm torque.",
                "This regime produces high mechanical stress on the spindle "
                "and reduced motor self-cooling at low fan speed.",
            ],
            recommended_action=(
                "First, review cutting parameters — aggressive feed rate or "
                "depth of cut at low spindle speed produces this signature "
                "before any mechanical fault exists. If parameters are within "
                "spec, inspect cutting tool for dulling. Only then investigate "
                "spindle drive for binding, misalignment, or bearing condition."
            ),
            pattern_id="stall_zone",
        )

    # Pattern 4: Mechanical overstrain
    # SHAP dominated by Power_W contribution; indicates the model is attributing
    # risk to high mechanical power output. In the new feature set we don't
    # combine with Risk_Heuristic (dropped as redundant for tree models), so
    # we rely on Power_W SHAP alone. Energy_Per_Wear is the closest related
    # feature; we include it as a secondary signal.
    power_shap = shap_values.get("Power_W", 0.0)
    energy_per_wear_shap = shap_values.get("Energy_Per_Wear", 0.0)
    combined_power_shap = power_shap + 0.5 * energy_per_wear_shap
    if combined_power_shap > SHAP_DOMINANT:
        power_raw = raw_inputs.get("Power_W", 0.0)
        return Diagnosis(
            risk_category=risk_category,
            risk_probability=probability,
            primary_diagnosis=(
                "Mechanical overload (overstrain pattern) — power output and "
                "thermal load both elevated."
            ),
            contributing_factors=[
                f"Mechanical power at {power_raw:.0f} W with elevated thermal load.",
                "Combined high load and high temperature shortens bearing life "
                "(load³ effect) and accelerates lubricant degradation. The "
                "interaction is multiplicative, not additive.",
            ],
            recommended_action=(
                "Reduce mechanical load if the process permits. Check spindle "
                "bearing condition via vibration spectrum if available, or by "
                "audible/temperature inspection. Inspect lubricant for "
                "discolouration or contamination. Verify drive coupling or "
                "belt condition. Schedule full bearing inspection if signs "
                "of wear are present."
            ),
            pattern_id="overstrain",
        )

    # Pattern 5: Multi-factor fallback
    # No single dominant pattern; risk is distributed across features.
    # Report the top contributors and direct to SHAP drill-down.
    positive_contributors = [
        (feature, value) for feature, value in shap_values.items()
        if value > SHAP_SIGNIFICANT
    ]
    positive_contributors.sort(key=lambda pair: -pair[1])
    top_features = [feature for feature, _ in positive_contributors[:3]]

    if top_features:
        factors = [
            f"Multiple risk factors elevated: {', '.join(top_features)}.",
            "No single physical failure mode dominates the prediction.",
        ]
    else:
        factors = [
            "Risk is elevated but no individual feature contributes strongly.",
            "Model uncertainty may be a contributing factor; treat with caution.",
        ]

    return Diagnosis(
        risk_category=risk_category,
        risk_probability=probability,
        primary_diagnosis="Multi-factor risk elevation — engineering review recommended.",
        contributing_factors=factors,
        recommended_action=(
            "Open the SHAP drill-down view for full feature attributions. "
            "Have a reliability engineer review before scheduling maintenance."
        ),
        pattern_id="multi_factor",
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Run the translator on representative scenarios and print outputs.

    Each scenario approximates the SHAP/input pattern expected for a specific
    failure mode in the AI4I 2020 dataset. The expected pattern_id is given
    for verification.
    """
    scenarios = [
        {
            "name": "Healthy machine, default operating point",
            "shap": {"Temp_Delta": -0.02, "Power_W": 0.01, "Tool_Wear": 0.0, "Torque": -0.01},
            "raw": {"Temp_Delta": 9.5, "Power_W": 5000, "Tool_Wear": 50,
                    "Torque": 40, "Rotational_Speed": 1700},
            "prob": 0.005,
            "expected_pattern": "healthy_default",
        },
        {
            "name": "Heat dissipation case",
            "shap": {"Temp_Delta": 0.42, "Power_W": 0.08, "Tool_Wear": 0.03, "Torque": 0.02},
            "raw": {"Temp_Delta": 13.5, "Power_W": 6500, "Tool_Wear": 80,
                    "Torque": 45, "Rotational_Speed": 1800},
            "prob": 0.68,
            "expected_pattern": "heat_dissipation",
        },
        {
            "name": "Tool wear case",
            "shap": {"Tool_Wear": 0.38, "Temp_Delta": 0.05, "Power_W": 0.04, "Torque": 0.02},
            "raw": {"Temp_Delta": 9.8, "Power_W": 5500, "Tool_Wear": 215,
                    "Torque": 42, "Rotational_Speed": 1650},
            "prob": 0.55,
            "expected_pattern": "tool_wear",
        },
        {
            "name": "Stall zone case",
            "shap": {"Torque": 0.18, "Rotational_Speed": 0.12, "Power_W": 0.08, "Temp_Delta": 0.04},
            "raw": {"Temp_Delta": 9.6, "Power_W": 4800, "Tool_Wear": 100,
                    "Torque": 68, "Rotational_Speed": 1200},
            "prob": 0.62,
            "expected_pattern": "stall_zone",
        },
        {
            "name": "Mechanical overstrain case",
            "shap": {"Power_W": 0.22, "Energy_Per_Wear": 0.12, "Torque": 0.08, "Temp_Delta": 0.06},
            "raw": {"Temp_Delta": 11.0, "Power_W": 9200, "Tool_Wear": 120,
                    "Torque": 65, "Rotational_Speed": 1500},
            "prob": 0.78,
            "expected_pattern": "overstrain",
        },
        {
            "name": "Multi-factor case (no dominant pattern)",
            "shap": {"Tool_Wear": 0.08, "Temp_Delta": 0.07, "Power_W": 0.06,
                     "Torque": 0.05, "Type_Encoded": 0.04},
            "raw": {"Temp_Delta": 10.5, "Power_W": 6000, "Tool_Wear": 150,
                    "Torque": 50, "Rotational_Speed": 1550},
            "prob": 0.45,
            "expected_pattern": "multi_factor",
        },
    ]

    print("=" * 72)
    print("DIAGNOSTIC TRANSLATOR — SELF TEST")
    print("=" * 72)

    pass_count = 0
    for scenario in scenarios:
        diag = translate_shap(
            shap_values=scenario["shap"],
            raw_inputs=scenario["raw"],
            probability=scenario["prob"],
        )
        match = diag.pattern_id == scenario["expected_pattern"]
        status = "PASS" if match else "FAIL"
        if match:
            pass_count += 1

        print(f"\n--- {scenario['name']} ---")
        print(f"Expected pattern: {scenario['expected_pattern']}")
        print(f"Matched pattern:  {diag.pattern_id}  [{status}]")
        print()
        print(diag)

    print()
    print("=" * 72)
    print(f"Result: {pass_count}/{len(scenarios)} scenarios passed")
    print("=" * 72)


if __name__ == "__main__":
    _self_test()
