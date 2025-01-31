import numpy as np
import pandas as pd

def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'user_id': np.random.choice(['user1', 'user2', 'user3'], size=num_samples),
        'amount': np.random.uniform(50, 5000, size=num_samples),  # Random transaction amounts
        'location': np.random.choice(['NY', 'California', 'India', 'Brazil'], size=num_samples),
        'transaction_type': np.random.choice(['purchase', 'refund', 'withdrawal'], size=num_samples),
        'merchant': np.random.choice(['amazon', 'eBay', 'Walmart'], size=num_samples),
        'device_info': np.random.choice([{'device_id': 'device1', 'device_type': 'mobile'}, {'device_id': 'device2', 'device_type': 'laptop'}], size=num_samples),
        'ip_address': np.random.choice(['192.168.1.1', '192.168.2.1'], size=num_samples),
        'description': np.random.choice(['electronics', 'clothing', 'groceries'], size=num_samples),
        'isFraud': np.random.choice([0, 1], size=num_samples)  # Random labels (0 = non-fraud, 1 = fraud)
    }
    return pd.DataFrame(data)
training_data=generate_synthetic_data()

X = training_data.drop(columns=["isFraud"])
print(type(X['user_id'][0]))
