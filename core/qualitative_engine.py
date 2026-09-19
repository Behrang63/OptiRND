"""
Deterministic qualitative scoring engine for technical feasibility assessment.

Pure functional module - no side effects, no external dependencies.
"""

from __future__ import annotations


def calculate_technical_success_probability(
    trl_level: int,
    team_capability: str,
    technical_complexity: str,
    supply_dependence: str,
) -> float:
    """
    Calculate deterministic technical success probability based on qualitative factors.

    Weight distribution:
    - TRL: 35%
    - Team Capability: 25%
    - Technical Complexity: 20%
    - Supply Chain Dependency: 20%

    Mapping rules:
    - TRL: 1-3 -> 0.35, 4-6 -> 0.70, 7-9 -> 0.95
    - team_capability: "LOW" -> 0.3, "MEDIUM" -> 0.7, "HIGH" -> 1.0
    - technical_complexity: "HIGH" -> 0.3, "MEDIUM" -> 0.7, "LOW" -> 1.0
    - supply_dependence: "CRITICAL_IMPORT" -> 0.2, "MODERATE_DELAY" -> 0.6, "DOMESTIC" -> 1.0

    Returns probability rounded to 2 decimal places, bounded within [0.10, 0.95].
    """
    # TRL scoring (35% weight)
    if 1 <= trl_level <= 3:
        trl_score = 0.35
    elif 4 <= trl_level <= 6:
        trl_score = 0.70
    elif 7 <= trl_level <= 9:
        trl_score = 0.95
    else:
        trl_score = 0.35  # Default to lowest for invalid TRL

    # Team capability scoring (25% weight)
    team_scores = {"LOW": 0.3, "MEDIUM": 0.7, "HIGH": 1.0}
    team_score = team_scores.get(team_capability.upper(), 0.7)

    # Technical complexity scoring (20% weight) - inverted: HIGH complexity = lower score
    complexity_scores = {"HIGH": 0.3, "MEDIUM": 0.7, "LOW": 1.0}
    complexity_score = complexity_scores.get(technical_complexity.upper(), 0.7)

    # Supply chain dependency scoring (20% weight)
    supply_scores = {"CRITICAL_IMPORT": 0.2, "MODERATE_DELAY": 0.6, "DOMESTIC": 1.0}
    supply_score = supply_scores.get(supply_dependence.upper(), 0.6)

    # Weighted aggregate
    probability = (
        trl_score * 0.35
        + team_score * 0.25
        + complexity_score * 0.20
        + supply_score * 0.20
    )

    # Round to 2 decimal places and bound within [0.10, 0.95]
    probability = round(probability, 2)
    return max(0.10, min(0.95, probability))