import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("students.csv")

X = data[["study_hours", "attendance", "sleep_hours"]]
y = data["final_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

study_hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance percentage: "))
sleep_hours = float(input("Enter sleep hours: "))

new_student = [[study_hours, attendance, sleep_hours]]
prediction = model.predict(new_student)

print("Predicted final score:", prediction[0])

score = model.score(X_test, y_test)
print("Model accuracy:" , score)