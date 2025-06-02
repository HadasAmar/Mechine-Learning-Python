# 🎓 Student Habits & Exam Score Prediction

🔍 Final project for the Python AI course – predicting exam scores based on students' habits and lifestyle data.

---

## 🧠 Project Description

This machine learning project analyzes the relationship between students' daily habits and their exam performance. The goal is to predict final exam scores using variables like sleep hours, study time, physical activity, and more.

---

## 📁 Dataset

Data source: `student_habits_performance.csv`

- **exam_score** – The target variable (exam result)
- **Other features** – Includes sleep, study routines, exercise, diet, tech use, and more.

---

## 🔍 Workflow

1. **Data loading and inspection**
2. **Preprocessing**:
   - Dropping unique identifiers
   - One-Hot Encoding for categorical features
   - Handling missing values
3. **Train/Test split**
4. **Feature scaling (Standardization)**
5. **Model building and training**:
   - Linear Regression
   - Decision Tree Regressor
   - K-Nearest Neighbors (KNN)
6. **Model evaluation using MSE and R²**
7. **Comparison of results with visual plots**

---

## 📊 Visualizations

Each model is visualized with a scatter plot:
- Real vs. predicted exam scores
- A red reference line for visual comparison

---

## 🏆 Results

The model with the highest R² score is selected as the best-performing model.

---

## 🛠️ Technologies Used

- Python
- pandas, matplotlib
- scikit-learn

---

## 🚀 How to Run

1. Make sure the `student_habits_performance.csv` file is in the working directory.
2. Install required packages:
   ```bash
   pip install pandas scikit-learn matplotlib
   ```
3. Run the Python script.

---

## 📄 License

This project is licensed under the MIT License – feel free to use, fork, and contribute!
