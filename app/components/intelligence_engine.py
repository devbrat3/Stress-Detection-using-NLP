import numpy as np


def generate_alerts(result, history):
    alerts = []

    if result["risk"] == "Critical":
        alerts.append("Critical stress detected")

    if result["confidence"] < 60:
        alerts.append("Low confidence prediction")

    if len(history) > 5:
        vals = [c for _, c in history[-5:]]
        if vals[-1] > np.mean(vals) + np.std(vals):
            alerts.append("Sudden stress spike detected")

    return alerts


def generate_insights(result, history):
    insights = []

    if len(history) > 3:
        vals = [c for _, c in history[-5:]]

        if vals[-1] > vals[0]:
            insights.append("Stress trend increasing")

        elif vals[-1] < vals[0]:
            insights.append("Stress improving")

    if result["confidence"] > 80:
        insights.append("High confidence prediction")

    return insights


def generate_intelligence(result, history):
    if not history:
        return {"risk_score": 0, "volatility": 0}

    vals = [c for _, c in history]

    return {
        "risk_score": int(sum(vals) / len(vals)),
        "volatility": round(np.std(vals), 2)
    }


def generate_recommendation(result):
    if result["risk"] == "Critical":
        return "Immediate professional consultation required"

    if result["risk"] == "High":
        return "Reduce workload and monitor stress"

    if result["risk"] == "Moderate":
        return "Practice relaxation techniques"

    return "Maintain healthy routine"


def generate_summary(result, history):
    trend = "Stable"

    if len(history) > 3:
        vals = [c for _, c in history[-5:]]
        if vals[-1] > vals[0]:
            trend = "Increasing"
        elif vals[-1] < vals[0]:
            trend = "Decreasing"

    return f"""
Stress: {result['label']}
Confidence: {result['confidence']}%
Risk: {result['risk']}
Trend: {trend}
Advice: {result['advice']}
"""