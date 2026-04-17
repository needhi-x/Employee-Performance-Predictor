import pandas as pd
import numpy as np

def generate_data(n=500):
    np.random.seed(42)

    data = pd.DataFrame({
        'age': np.random.randint(22, 60, n),
        'experience': np.random.randint(1, 35, n),
        'salary': np.random.randint(20000, 150000, n),
        'training_hours': np.random.randint(5, 100, n),
        'department': np.random.choice(['HR', 'IT', 'Sales', 'Finance'], n)
    })

    # Create performance score
    data['performance_score'] = (
        0.3 * data['experience'] +
        0.2 * data['training_hours'] +
        0.3 * (data['salary'] / 10000) +
        np.random.normal(0, 5, n)
    )

    # Convert to category
    data['performance'] = pd.cut(
        data['performance_score'],
        bins=[0, 10, 20, 100],
        labels=['Low', 'Medium', 'High']
    )

    return data

if __name__ == "__main__":
    df = generate_data()
    df.to_csv("data/employee_data.csv", index=False)