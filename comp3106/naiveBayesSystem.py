#!/usr/bin/python3
# Implements a Gaussian Naive Bayes classifier for numeric input features and categorical output labels.
# The model estimates per-class feature distributions from training data and predicts the most likely class.
import math
import pandas as pd

# Floor for standard deviation to prevent against 0 or Nan values
STD_FLOOR = 1e-7

class GaussianNaiveBayes:
    def __init__(self, training_prems, training_resps):
        # Split data indices based on each possible label
        resp_label_indices = {}
        for resp_var, row in training_resps.items():
            resp_label_indices[resp_var] = {}
            for i, label in row.items():
                if label not in resp_label_indices[resp_var]:
                    resp_label_indices[resp_var][label] = []
                resp_label_indices[resp_var][label].append(i)

        # Assemble necessary variables
        self.means = {}
        self.stds = {}
        self.resp_label_probs = {}
        for resp_var, labels in resp_label_indices.items():
            self.means[resp_var] = {}
            self.stds[resp_var] = {}
            self.resp_label_probs[resp_var] = {}

            # By default, labels are only "Pass" or "Fail"
            for label, indices in labels.items():
                data = training_prems.loc[indices]

                # Calculate mean and standard deviations
                self.means[resp_var][label] = data.mean()
                
                # Using ddof=0 changes std formula from dividing by n-1 to use just n
                # (for the unique case of only 1 training sample)
                std = data.std(ddof=0).fillna(0)
                self.stds[resp_var][label] = std.clip(lower=STD_FLOOR)

                # Probability of each classification (within training set)
                self.resp_label_probs[resp_var][label] = len(indices)/len(training_resps)
        
    # Classify an individual set of inputs to the single output of either pass or fail
    # NOTE: Implementation was made for easy interpretation, not purely optimization of time/space complexity
    def classify_individual(self, inputs):
        if inputs.empty:
            raise ValueError("Cannot classify without any feature variables")

        # Compute probability of each feature belonging to each label/classification (default of "pass" or "fail")
        ending_decisions = {}
        for resp_var, labels in self.resp_label_probs.items():
            ending_probs = {}
            for label in labels:
                naive_bayes_prob = 1

                # Compute Naive Bayes probabilities for each prem_var
                for prem_var, val in inputs.items():                    
                    # Probability of this val given a label
                    # E.g: P(prem_var == val | pass)
                    prob = GaussianNaiveBayes.gaussian(val, self.means[resp_var][label][prem_var], self.stds[resp_var][label][prem_var])
                    
                    # Naive Bayes assumption that all features are independant given the classification
                    # E.g: P(X|pass) = P(x1|pass)*P(x2|pass)...*P(xn|pass)
                    naive_bayes_prob *= prob

                # Track final probability of each label
                ending_probs[label] = naive_bayes_prob * self.resp_label_probs[resp_var][label]
            
            # Determine label with highest probability
            ending_decisions[resp_var] = max(ending_probs, key=ending_probs.get)

        return ending_decisions

    # Wrapper function to classify each data point individually
    def classify(self, data_set):
        preds = {}
        for _, row in data_set.iterrows():
            classification = self.classify_individual(row)
            for resp_var, val in classification.items():
                if resp_var not in preds:
                    preds[resp_var] = []
                preds[resp_var].append(val)

        return pd.DataFrame(preds, index=data_set.index)

    # Complex formula for Gaussian probability
    @staticmethod
    def gaussian(val, mean, std):
        coef = 1.0 / (math.sqrt(2.0 * math.pi * std**2))
        exp = math.exp(-((val - mean)**2 / (2 * std**2)))
        return coef * exp