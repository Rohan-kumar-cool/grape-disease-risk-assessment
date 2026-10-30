import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# 1. Define input and output variables
temperature = ctrl.Antecedent(
    np.arange(-5, 46, 1), "temperature"
)
humidity = ctrl.Antecedent(
    np.arange(0, 101, 1), "humidity"
)
rainfall = ctrl.Antecedent(
    np.arange(0, 11, 0.1), "rainfall"
)

favorability = ctrl.Consequent(
    np.arange(0, 101, 1), "favorability"
)


# 2. Define fuzzy membership functions
temperature["low"] = fuzz.trimf(
    temperature.universe, [-5, 5, 15]
)
temperature["moderate"] = fuzz.trimf(
    temperature.universe, [10, 22, 30]
)
temperature["high"] = fuzz.trimf(
    temperature.universe, [25, 35, 45]
)

humidity["low"] = fuzz.trimf(
    humidity.universe, [0, 0, 50]
)
humidity["moderate"] = fuzz.trimf(
    humidity.universe, [30, 55, 80]
)
humidity["high"] = fuzz.trimf(
    humidity.universe, [65, 85, 100]
)

rainfall["low"] = fuzz.trimf(
    rainfall.universe, [0, 0, 3]
)
rainfall["moderate"] = fuzz.trimf(
    rainfall.universe, [1, 4, 7]
)
rainfall["high"] = fuzz.trimf(
    rainfall.universe, [5, 10, 10]
)

favorability["low"] = fuzz.trimf(
    favorability.universe, [0, 0, 40]
)
favorability["medium"] = fuzz.trimf(
    favorability.universe, [25, 50, 75]
)
favorability["high"] = fuzz.trimf(
    favorability.universe, [60, 100, 100]
)


# 3. Define demonstration rules
rules = [
    ctrl.Rule(
        temperature["moderate"] & humidity["high"],
        favorability["high"]
    ),
    ctrl.Rule(
        temperature["high"] & humidity["high"],
        favorability["high"]
    ),
    ctrl.Rule(
        humidity["high"] & rainfall["high"],
        favorability["high"]
    ),
    ctrl.Rule(
        temperature["moderate"] & humidity["moderate"],
        favorability["medium"]
    ),
    ctrl.Rule(
        humidity["moderate"] & rainfall["moderate"],
        favorability["medium"]
    ),
    ctrl.Rule(
        temperature["low"] & humidity["low"],
        favorability["low"]
    ),
    ctrl.Rule(
        humidity["low"] & rainfall["low"],
        favorability["low"]
    ),
]

system = ctrl.ControlSystem(rules)


def estimate_favorability(temp, hum, rain):
    """
    Estimate demonstration environmental favorability.

    Inputs:
        temp: temperature in degrees Celsius
        hum: relative humidity in percent
        rain: rainfall in the chosen input unit

    Returns:
        A score from 0 to 100.
    """
    simulation = ctrl.ControlSystemSimulation(system)

    simulation.input["temperature"] = temp
    simulation.input["humidity"] = hum
    simulation.input["rainfall"] = rain

    simulation.compute()

    return round(float(simulation.output["favorability"]), 2)


if __name__ == "__main__":
    examples = [
        (15, 35, 0.0),
        (24, 75, 2.0),
        (28, 90, 6.0),
    ]

    for temp, hum, rain in examples:
        score = estimate_favorability(temp, hum, rain)

        print(
            f"Temperature={temp}°C, "
            f"Humidity={hum}%, Rainfall={rain} "
            f"-> Favorability={score}/100"
        )
