import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessing import preprocess
from src.model import train_model, evaluate, save_model

# Load data
df = pd.read_csv("data/employee_data.csv")

# Preprocess
X, y = preprocess(df)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train
model = train_model(X_train, y_train)

# Evaluate
evaluate(model, X_test, y_test)

# Save
save_model(model)

import matplotlib.pyplot as plt
import seaborn as sns

# Plot performance distribution
sns.countplot(x='performance', data=df)
plt.title("Employee Performance Distribution")

# Save image
plt.savefig("outputs/performance_distribution.png")

plt.show()


plt.figure()

sns.boxplot(x='performance', y='salary', data=df)
plt.title("Salary vs Performance")

plt.savefig("outputs/salary_vs_performance.png")

plt.show()