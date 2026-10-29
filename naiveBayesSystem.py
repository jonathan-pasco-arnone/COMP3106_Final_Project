#!/usr/bin/python3
# Utilizing Gaussian Naive Bayes
import math
import pandas as pd

class GaussianNaiveBayes:
    def __init__(self, training_prems, training_resps):
        # Split data into passes and fails
        passing_i = training_resps.eq(["pass"], axis="columns").all(axis="columns")
        failing_i = training_resps.eq(["fail"], axis="columns").all(axis="columns")
        
        passing_prems = training_prems.loc[passing_i]
        passing_resps = training_resps.loc[passing_i]
    
        failing_prems = training_prems.loc[failing_i]
        failing_resps = training_resps.loc[failing_i]
    
        # Calculate mean and standard deviations for each variable for both passing and failing sets
        self.passing_probs = {"mean": {}, "std": {}}
        self.failing_probs = {"mean": {}, "std": {}}
        self.prob_pass = len(passing_resps)/(len(passing_resps)+len(failing_resps))
        
        for (var, prems_pass), (_, prems_fail) in zip(passing_prems.items(), failing_prems.items()):
            self.passing_probs["mean"][var] = prems_pass.mean()
            self.passing_probs["std"][var] = prems_pass.std()

            self.failing_probs["mean"][var] = prems_fail.mean()
            self.failing_probs["std"][var] = prems_fail.std()
        
    # Classify an individual set of inputs to the single output of either pass or fail
    # NOTE: Implementation was made for easy interpretation, not purely optimization of time/space complexity
    def classify_individual(self, inputs):
        if inputs.empty:
            raise ValueError("Cannot classify without any feature variables")

        # Compute probability of each feature belonging to "pass" or "fail"
        probs = {"pass": {}, "fail": {}}
        for var, val in inputs.items():
            # Probability of this val given a pass
            # P(var == val | pass)
            probs["pass"][var] = GaussianNaiveBayes.gaussian(val, self.passing_probs["mean"][var], self.passing_probs["std"][var])

            # Probability of this val given a fail
            # P(var == val | fail)
            probs["fail"][var] = GaussianNaiveBayes.gaussian(val, self.failing_probs["mean"][var], self.failing_probs["std"][var])

        # Naive Bayes assumption that all features are independant give the classification
        # P(X|pass) = P(x1|pass)*P(x2|pass)...*P(xn|pass)
        naive_bayes_probs = {"pass": 1, "fail": 1}
        for prob_pass, prob_fail in zip(probs["pass"].values(), probs["fail"].values()):
            naive_bayes_probs["pass"] *= prob_pass
            naive_bayes_probs["fail"] *= prob_fail

        # Multiply by prior probability of each class
        # P(pass)*P(X|pass)
        ending_probs = {"pass": naive_bayes_probs["pass"]*self.prob_pass,
                        "fail": naive_bayes_probs["fail"]*(1.0-self.prob_pass)}

        return "pass" if ending_probs["pass"] > ending_probs["fail"] else "fail"

    # Wrapper function to classify each data point individually
    def classify(self, data_set):
        preds = []
        for _, row in data_set.iterrows():
            preds.append(self.classify_individual(row))
        return pd.DataFrame({"exam_score": preds}, index=data_set.index)

    # Complex formula for Gaussian probability
    @staticmethod
    def gaussian(val, mean, std):
        coef = 1.0 / (math.sqrt(2.0 * math.pi * std**2))
        exp = math.exp(-((val - mean)**2 / (2 * std**2)))
        return coef * exp