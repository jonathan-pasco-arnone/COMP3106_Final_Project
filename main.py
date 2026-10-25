import pandas as pd

relevant_vars = [
    "study_hours_per_day",
    "attendance_percentage",
    "sleep_hours",
    "exercise_frequency",
    "exam_score"
]

student_habits = pd.read_csv(
    "student_habits_performance.csv",
    usecols=relevant_vars
)

print(student_habits.iloc[0])