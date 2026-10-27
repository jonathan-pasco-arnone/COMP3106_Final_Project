import pandas as pd
import fuzzySystem

# Import File
student_habits = pd.read_csv(
    "student_habits_performance.csv",
    usecols=fuzzySystem.DEFAULT_FUZZY_SETS.keys()
)

# Testing
loss = []
for i in range(len(student_habits)):
    row = student_habits.iloc[i]
    sample_inputs = row.drop("exam_score")
    true_score = row["exam_score"]

    scores = fuzzySystem.run_rules(sample_inputs)
    final_score = fuzzySystem.weighted_final_score(scores)
    loss.append(abs(final_score["exam_score"] - true_score))


print("MAE (Mean Absolute Error):", sum(loss) / len(loss))