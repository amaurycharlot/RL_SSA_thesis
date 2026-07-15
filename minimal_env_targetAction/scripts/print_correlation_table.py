"""
Goes through the correlation json files and prints the values how hey should be copied into latex.
"""

import json
import os

if __name__ == "__main__":
    corr_values = []
    for file in os.listdir("correlation"):
        if file.endswith(".json"):
            with open(os.path.join("correlation", file), "r") as f:
                data = json.load(f)
                for key, value in data.items():
                    if key != "num_runs":
                        corr_values.append(value)
    for i in range(0, len(corr_values), 4):
        print(f"& {corr_values[i]:.2f} & {corr_values[i+1]:.2f} & {corr_values[i+2]:.2f} & {corr_values[i+3]:.2f} \\\\")