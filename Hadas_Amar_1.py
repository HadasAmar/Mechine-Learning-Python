import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv("student_habits_performance.csv")

#חקירת נתונים
print(df.info())
print(df.describe())

#עיבוד נתונים
df = df.drop(['student_id'], axis=1)
df = pd.get_dummies(df, drop_first=True)
df=df.dropna()

corr_matrix = df.corr()
print(corr_matrix['exam_score'].sort_values(ascending=False))

X = df.drop('exam_score', axis=1)
y = df['exam_score']

scaler = StandardScaler()
X = scaler.fit_transform(X)

#פיצול נתונים
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#בניית מודלים
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "KNN": KNeighborsRegressor()
}

predictions = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions[name] = model.predict(X_test)

#הערכת מודלים
def print_model_metrics(y_test, y_pred, model_name):
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'{model_name}\nMSE: {mse}, R²: {r2}\n')

for name, preds in predictions.items():
    print_model_metrics(y_test, preds, name)

#השוואה
best_model = None
best_r2 = None

for i, (name, preds) in enumerate(predictions.items(), start=1):

#בדיקה לפי R2 מה המודל הטוב ביותר
    r2 = r2_score(y_test, preds)
    if best_r2 is None or r2 > best_r2:
        best_r2 = r2
        best_model = name

    plt.subplot(2, 2, i)
    plt.scatter(y_test, preds, alpha=0.6, marker='+')
    plt.xlabel('Real Score')
    plt.ylabel('Predicted Score')
    plt.title(name)

    min_val = min(y_test.min(), preds.min())
    max_val = max(y_test.max(), preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red')#קו לינארי

print(f"המודל הכי מדויק הוא: {best_model} עם R² של {best_r2}")

plt.tight_layout()
plt.show()


