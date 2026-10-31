#!/usr/bin/python3
# Combines fuzzy-rule activation scores with numeric features and classifies them using Gaussian Naive Bayes.

from .fuzzySystem import fuzzyExperts
from .naiveBayesSystem import GaussianNaiveBayes

class hybridFuzzyBayes(GaussianNaiveBayes):
    def __init__(self, training_prems, training_resps, rules, fuzzy_features, fuzzy_outputs=None):
        self.fuzzy_expert = fuzzyExperts(rules, fuzzy_features, fuzzy_outputs)
        
        self.add_fuzzy_features(training_prems)

        super().__init__(training_prems, training_resps)

    # Classify using fuzzy scores as additional features
    def classify(self, data_set):
        self.add_fuzzy_features(data_set)
        return super().classify(data_set)

    def add_fuzzy_features(self, data_set):
        fuzzy_scores = {}
        for _, row in data_set.iterrows():
            cur_scores = self.fuzzy_expert.run_rules(row)
            for resp_var, labels in cur_scores.items():
                for label, val in labels.items():
                    feature_name = f"{resp_var}_{label}"
                    fuzzy_scores.setdefault(feature_name, []).append(val)

        for feature_name, values in fuzzy_scores.items():
            data_set[feature_name] = values