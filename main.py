import pandas as pd
import fuzzySystem

# Includes the relevant variable names from the csv as well as their trapezoidal membership function
# Where "None" just means it extends infinitely in that direction
relevant_vars = {
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
    },
    # Low end is FAR larger than high end due to the assumption that any exam_score < 50 means a fail, and thus could not
    # be considered "medium"
    "exam_score": {
        "low":    [0,0,55,65],
        "medium": [50,60,82,90],
        "high":   [72,84,100,100]
    },
}

# student_habits = pd.read_csv(
#     "student_habits_performance.csv",
#     usecols=relevant_vars.keys()
# )

# print(student_habits.iloc[0])

print(fuzzySystem.full_fuzzification(relevant_vars["study_hours_per_day"], 3))