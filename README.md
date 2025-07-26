# 🚢 Titanic Survival Prediction using Logistic Regression 🎯

This project predicts whether a passenger survived the Titanic disaster using a **Logistic Regression** model. It involves cleaning real-world data, performing exploratory analysis, and building a predictive binary classification model. The goal is to understand the influence of different features like age, gender, and passenger class on survival outcomes.

---

## 📄 Dataset

- **Source:** [Titanic - Machine Learning from Disaster (Kaggle)](https://www.kaggle.com/c/titanic)
- **Description:** The dataset contains demographic and travel details of passengers aboard the RMS Titanic, along with their survival status.
  - `Survived`: Target variable (0 = Did not survive, 1 = Survived)
  - `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`: Feature columns used for prediction
  - `Name`, `Ticket`, `Cabin`, etc., are either dropped or preprocessed

---

## ⚙️ Technologies Used

- Python  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-learn  

---

## 📊 Project Workflow

### 1️⃣ Data Loading & Cleaning
- Load dataset using Pandas.
- Remove or fill missing values (e.g., drop `Cabin`, fill `Age` with mean, fill `Embarked` with mode).
- Encode categorical features (`Sex`, `Embarked`) into numerical values.

### 2️⃣ Exploratory Data Analysis (EDA)
- Use Seaborn and Matplotlib for visualizations:
  - Survival distribution
  - Gender vs survival
  - Passenger class impact on survival
- Identify feature importance and correlation.

### 3️⃣ Feature Engineering
- Drop irrelevant columns: `PassengerId`, `Name`, `Ticket`
- Separate dataset into features (`X`) and target (`Y`)

### 4️⃣ Data Splitting
- Split the data into training and test sets (80/20) using `train_test_split`.

### 5️⃣ Model Training
- Train a **Logistic Regression** model on the training dataset.

### 6️⃣ Model Evaluation
- Evaluate model performance using accuracy on both training and test sets.

---

## ✅ Results

- **Training Accuracy:** `80.76%`  
- **Test Accuracy:** `78.21%`  
- The model demonstrates decent generalization and helps understand key factors influencing survival.

---

## 💡 Key Learnings

- Hands-on experience with data preprocessing, encoding, and visualization.
- Understanding logistic regression for binary classification.
- Handling missing values and categorical variables in real-world datasets.
- Evaluating model performance and avoiding overfitting.

---

## 📥 How to Run

1️⃣ **Clone this repository:**

```bash
git clone https://github.com/RONAKBAGRI/Titanic-Survival-Prediction.git
```

2️⃣ **Install dependencies:**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

3️⃣ **Run the notebook:**
```bash
jupyter notebook Titanic_Survival_Prediction.ipynb
```