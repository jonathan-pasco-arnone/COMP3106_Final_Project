#!/usr/bin/python3
# Evaluates fuzzy rules using trapezoidal membership functions and classifies inputs from the rule activations.
import pandas as pd

class fuzzyExperts():
    def __init__(self, rules, fuzzy_features, fuzzy_outputs=None):
        self.rules = rules
        self.fuzzy_features = fuzzy_features
        self.fuzzy_outputs = fuzzy_outputs or {}

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
        # Set 0s as default values 
        conclusions = {
            resp_var: {label: 0.0 for label in output.get("labels", [])}
            for resp_var, output in self.fuzzy_outputs.items()
        }
        for rule in self.rules:
            score = self.eval_rule(inputs, rule["if"])

            output = rule["then"]
            variable = output["variable"]
            label = output["is"]

            # Reference conclusions dict and update score using OR
            output_scores = conclusions.setdefault(variable, {})
            output_scores[label] = fuzzyExperts.OR([output_scores.get(label, 0.0), score])

        return conclusions

    # Choose the class with the strongest rule activation.
    def classify_individual(self, inputs):
        scores = self.run_rules(inputs)
        
        ending_decisions = {}
        for resp_var, output_scores in scores.items():
            ending_decisions[resp_var] = max(output_scores, key=output_scores.get)

        return ending_decisions

    def classify(self, data_set):
        preds = {}
        for _, row in data_set.iterrows():
            classification = self.classify_individual(row)
            for resp_var, val in classification.items():
                if resp_var not in preds:
                    preds[resp_var] = []
                preds[resp_var].append(val)
        # Using index=... to insure row labels are maintained (since they are randomly ordered)
        return pd.DataFrame(preds, index=data_set.index)

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
