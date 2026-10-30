
import numpy as np


def trimf(x, a, b, c):
    """Triangular membership function."""
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x)

    if a == b:
        result[x == a] = 1.0
    else:
        mask = (a < x) & (x <= b)
        result[mask] = (x[mask] - a) / (b - a)

    if b == c:
        result[x == c] = 1.0
    else:
        mask = (b < x) & (x < c)
        result[mask] = (c - x[mask]) / (c - b)

    result[x == b] = 1.0
    return np.clip(result, 0.0, 1.0)


def trapmf(x, a, b, c, d):
    """Trapezoidal membership function."""
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x)

    if a == b:
        result[x == a] = 1.0
    else:
        mask = (a < x) & (x < b)
        result[mask] = (x[mask] - a) / (b - a)

    mask = (b <= x) & (x <= c)
    result[mask] = 1.0

    if c == d:
        result[x == d] = 1.0
    else:
        mask = (c < x) & (x < d)
        result[mask] = (d - x[mask]) / (d - c)

    return np.clip(result, 0.0, 1.0)


def _validate_number(name, value, minimum, maximum):
    if not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be numeric.")
    if not np.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")


def _fuzzify(value, sets):
    x = np.array([value], dtype=float)
    return {name: float(function(x)[0]) for name, function in sets.items()}


def _risk_output(rules):
    """Aggregate Mamdani rules and apply centroid defuzzification."""
    universe = np.linspace(0, 100, 1001)

    risk_sets = {
        "low": trapmf(universe, 0, 0, 20, 45),
        "moderate": trimf(universe, 25, 50, 75),
        "high": trapmf(universe, 55, 80, 100, 100),
    }

    aggregated = np.zeros_like(universe)

    for strength, label in rules:
        aggregated = np.maximum(
            aggregated,
            np.minimum(strength, risk_sets[label]),
        )

    area = np.sum(aggregated)
    if area == 0:
        return None

    return float(np.sum(universe * aggregated) / area)


def calculate_risk(
    disease,
    confidence,
    temperature,
    humidity,
    rainfall,
    wet_days,
):
    """
    Calculate a prototype disease-specific Mamdani risk index.

    disease: 'Downy Mildew' or 'Powdery Mildew'
    confidence: image-model confidence, 0-100
    temperature: five-day average temperature, Celsius
    humidity: five-day average relative humidity, percent
    rainfall: five-day rainfall in the dataset's unspecified units
    wet_days: number of wet days in the five-day period, 0-5

    Returns an index and category, not a validated probability.
    Missing required inputs produce an uncertain result.
    """

    disease_aliases = {
        "downy mildew": "Downy Mildew",
        "powdery mildew": "Powdery Mildew",
    }
    if not isinstance(disease, str) or disease.lower() not in disease_aliases:
        raise ValueError("Disease must be Downy Mildew or Powdery Mildew.")
    disease = disease_aliases[disease.lower()]

    _validate_number("confidence", confidence, 0, 100)

    required = {
        "temperature": temperature,
        "humidity": humidity,
    }

    if disease == "Downy Mildew":
        required.update({
            "rainfall": rainfall,
            "wet_days": wet_days,
        })

    missing = [name for name, value in required.items() if value is None]
    if missing:
        return {
            "risk_index": None,
            "risk_category": "Uncertain",
            "message": "Missing required inputs: " + ", ".join(missing),
        }

    _validate_number("temperature", temperature, -10, 50)
    _validate_number("humidity", humidity, 0, 100)

    if disease == "Downy Mildew":
        _validate_number("rainfall", rainfall, 0, float("inf"))
        _validate_number("wet_days", wet_days, 0, 5)

    confidence_f = _fuzzify(confidence, {
        "low": lambda x: trapmf(x, 0, 0, 30, 55),
        "medium": lambda x: trimf(x, 30, 55, 80),
        "high": lambda x: trapmf(x, 65, 85, 100, 100),
    })

    temp_f = _fuzzify(temperature, {
        "low": lambda x: trapmf(x, -10, -10, 10, 18),
        "moderate": lambda x: trimf(x, 12, 22, 30),
        "high": lambda x: trapmf(x, 25, 32, 50, 50),
    })

    humidity_f = _fuzzify(humidity, {
        "low": lambda x: trapmf(x, 0, 0, 45, 65),
        "moderate": lambda x: trimf(x, 45, 65, 85),
        "high": lambda x: trapmf(x, 70, 85, 100, 100),
    })

    if disease == "Downy Mildew":
        # Preserve the provider's rainfall threshold; its physical unit
        # is unspecified. Wet days >= 2 is the other moisture condition.
        rainfall_f = _fuzzify(rainfall, {
            "low": lambda x: trapmf(x, 0, 0, 0.05, 0.10),
            "high": lambda x: trapmf(x, 0.05, 0.10, 1.0, 1.0),
        })
        wet_days_f = _fuzzify(wet_days, {
            "low": lambda x: trapmf(x, 0, 0, 1, 2),
            "high": lambda x: trapmf(x, 1, 2, 5, 5),
        })

        moisture = max(rainfall_f["high"], wet_days_f["high"])

        environmental_match = min(
            1.0 if temperature <= 23 else 0.0,
            1.0 if humidity >= 70 else 0.0,
            moisture,
        )

    else:
        # The provider's Powdery Mildew rule uses temperature and humidity.
        environmental_match = min(
            1.0 if 16 <= temperature <= 31 else 0.0,
            1.0 if 45 <= humidity <= 90 else 0.0,
        )

    # Very weak image evidence is not enough for a definitive result.
    if confidence < 30:
        return {
            "risk_index": None,
            "risk_category": "Uncertain",
            "message": "Image-model confidence is too low for this assessment.",
        }

    # Disease-specific environmental match controls the fuzzy rules.
    # For Downy Mildew, rainfall and wet days directly contribute above.
    # For Powdery Mildew, these variables are intentionally not used.
    rules = [
        (
            min(confidence_f["high"], environmental_match),
            "high",
        ),
        (
            min(confidence_f["medium"], environmental_match),
            "moderate",
        ),
        (
            min(confidence_f["high"], 1.0 - environmental_match),
            "low",
        ),
        (
            min(confidence_f["medium"], 1.0 - environmental_match),
            "low",
        ),
    ]

    risk_index = _risk_output(rules)

    if risk_index is None:
        return {
            "risk_index": None,
            "risk_category": "Uncertain",
            "message": "No fuzzy rule fired for these inputs.",
        }

    if risk_index < 35:
        category = "Low"
    elif risk_index < 65:
        category = "Moderate"
    else:
        category = "High"

    return {
        "risk_index": round(risk_index, 2),
        "risk_category": category,
        "message": (
            "Expert-designed prototype index; not a validated "
            "probability of disease worsening."
        ),
        "environmental_match": round(environmental_match * 100, 2),
    }
