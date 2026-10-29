#!/usr/bin/python3
# Hybrid Fuzzy + Naive Bayes Classification System

from fuzzySystem import fuzzyExperts
from naiveBayesSystem import GaussianNaiveBayes

class hybridFuzzyBayes(GaussianNaiveBayes):
    def __init__(self, training_prems, training_resps, rules=None, fuzzy_features=None):
        # Determine fuzzy parameters to pass
        # Need to do this in order to make use of fuzzyExperts' default parameters
        fuzzy_params = {}
        if rules is not None:
            fuzzy_params["rules"] = rules
        if fuzzy_features is not None:
            fuzzy_params["fuzzy_features"] = fuzzy_features
        self.fuzzyExpert = fuzzyExperts(**fuzzy_params)
        
        self.add_fuzzy_features(training_prems)

        super().__init__(training_prems, training_resps)

    def classify(self, data_set):
        self.add_fuzzy_features(data_set)
        return super().classify(data_set)

    # Add the fuzzy_pass and fuzzy_fail features
    def add_fuzzy_features(self, data_set):
        # Retreive fuzzy scores
        fuzzy_passing_scores = []
        fuzzy_failing_scores = []
        for _, row in data_set.iterrows():
            cur_scores = self.fuzzyExpert.run_rules(row)
            fuzzy_passing_scores.append(cur_scores.loc["pass"]["exam_score"])
            fuzzy_failing_scores.append(cur_scores.loc["fail"]["exam_score"])

        # Add fuzzy scores as additional features
        data_set["fuzzy_pass"] = fuzzy_passing_scores
        data_set["fuzzy_fail"] = fuzzy_failing_scores