#!/usr/bin/python3
import json
from pathlib import Path
import pandas as pd

# ==========================
# 
# Default Variables
# 
# ==========================

# Includes the relevant variable names from the csv as well as their trapezoidal membership function
# Where "None" just means it extends infinitely in that direction
DEFAULT_FUZZY_ANTECEDENTS = pd.DataFrame({
    "study_hours_per_day": {
        "low":    [0,0,3,4.5],
        "medium": [2,3,4.5,6.5],
        "high":   [4.5,6.5,None,None]
    },
    "attendance_percentage": {
        "poor":    [0,0,55,75],
        "average": [60,68,88,95],
        "strong":  [85,92,100,100]
    },
    "sleep_hours": {
        "minimal":  [0,0,4.5,6],
        "passable": [4,5.5,7.5,9],
        "ample":    [6.5,8,None,None]
    },
    "exercise_frequency": {
        "infrequent": [0,0,1.5,3],
        "regular":    [1,2.5,4.5,6.5],
        "frequent":   [3.5,5,None,None]
    }
})

DEFAULT_FUZZY_OUTPUTS = {
    "exam_score": ("fail", "pass")
}

# Rules
with Path(__file__).with_name("rules.json").open(encoding="utf-8-sig") as rules_file:
    DEFAULT_RULES = json.load(rules_file)["rules"]

# ==========================
# 
# Default Variables
# 
# ==========================

class fuzzyExperts():
    def __init__(self, rules=DEFAULT_RULES, fuzzy_features=DEFAULT_FUZZY_ANTECEDENTS):
        self.rules = rules
        self.fuzzy_features = fuzzy_features

    # ==========================
    # 
    # Main Functionality
    # 
    # ==========================
    
    @staticmethod
    def fuzzification(x,a,b,c,d):
        # Check validity of values
        # If a is None, then b has to be as well (same with c and d)
        if (a is None) != (b is None) or (c is None) != (d is None):
            raise ValueError(f"Error during fuzzification process. Trapezoidal components invalid.")

        # Full infinity on both sides
        if a is None and c is None:
            return 1.0

        # Left shoulder: [None, None, c, d]
        if a is None:  
            if x <= c:
                return 1.0
            # Already know x > c from previous if
            if x <= d:
                return (d-x)/(d-c)
            return 0.0

        # Right shoulder: [a, b, None, None]
        if c is None:
            if x < a:
                return 0.0
            # Already know x >= a from previous if
            if x < b:
                return (x-a)/(b-a)
            return 1.0

        # Core fuzzy membership function definition
        if x < a:
            return 0.0
        if x < b:
            return (x-a)/(b-a)
        if x <= c:
            return 1.0
        if x < d:
            return (d-x)/(d-c)
        return 0.0

    # Recursive function for evaluating rules
    def eval_rule(self, inputs, rule):
        if "variable" in rule:
            variable = rule["variable"]
            label = rule["is"]
            a,b,c,d = self.fuzzy_features[variable][label]
            return fuzzyExperts.fuzzification(inputs[variable], a, b, c, d)

        if "all" in rule:
            return fuzzyExperts.AND([self.eval_rule(inputs, condition) for condition in rule["all"]])
        if "any" in rule:
            return fuzzyExperts.OR([self.eval_rule(inputs, condition) for condition in rule["any"]])
        if "not" in rule:
            return fuzzyExperts.NOT(self.eval_rule(inputs, rule["not"]))

        raise ValueError(f"Invalid rule condition: {rule}")

    # "inputs" refers to the actual discrete values that are being tested against the rules.
    # "fuzzy_sets" refers to the defined set of variables, their fuzzy labels, and trapezoidal function definition
    def run_rules(self, inputs):
        # Holds fuzzy sets and their determined label scores
        conclusions = {}
        for rule in self.rules:
            score = self.eval_rule(inputs, rule["if"])

            output = rule["then"]
            variable = output["variable"]
            label = output["is"]

            # Reference conclusions dict and update score using OR
            output_scores = conclusions.setdefault(variable, {})
            output_scores[label] = fuzzyExperts.OR([output_scores.get(label, 0.0), score])

        # Only convert to pandas df at the end since python dicts are far better (time and space wise) at constant additions
        # Using pandas dicts for consistency accros inputs and outputs of various overarching funcs
        return pd.DataFrame(conclusions)

    # Choose the class with the strongest rule activation (ties classified as fail).
    def classify_individual(self, inputs):
        scores = self.run_rules(inputs)
        
        if "exam_score" not in scores:
            raise ValueError("Fuzzy rule results do not contain exam_score.")
        
        strengths = scores["exam_score"]
        if strengths.empty or strengths.max() <= 0:
            raise ValueError("No fuzzy rule fired for the provided inputs.")

        return "pass" if strengths.get("pass", 0.0) > strengths.get("fail", 0.0) else "fail"

    def classify(self, data_set):
        preds = []
        for _, row in data_set.iterrows():
            preds.append(self.classify_individual(row))
        # Using index=... to insure row labels are maintained (since they are randomly ordered)
        return pd.DataFrame({"exam_score": preds}, index=data_set.index)

    # ==========================
    # 
    # Fuzzy Operators
    # 
    # ==========================
    @staticmethod
    def NOT(x):
        return 1-x

    # "Values" is any list of scores
    @staticmethod
    def AND(values):
        return min(values)
    
    @staticmethod
    def OR(values):
        return max(values)
