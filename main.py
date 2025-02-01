
# detection.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pymongo import MongoClient
import gridfs
from pydantic import BaseModel
from typing import Optional
from torch import nn
from transformers import BertTokenizer, BertModel
import torch
import joblib
from sklearn.ensemble import RandomForestClassifier
import io
import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pymongo.server_api import ServerApi
# Helper functions
def adjust_risk_score(transaction, risk_score):
    if transaction['amount'] > 1000000:  # Extremely high amount
        risk_score = min(risk_score + 0.3, 1.0)
    if not is_valid_ip(transaction['ip_address']):  # Invalid IP
        risk_score = min(risk_score + 0.2, 1.0)
    if transaction['merchant'] in ['bhai', 'unknown']:  # Suspicious merchant
        risk_score = min(risk_score + 0.2, 1.0)
    return risk_score

def is_valid_ip(ip):
    try:
        parts = list(map(int, ip.split('.')))
        return len(parts) == 4 and all(0 <= part <= 255 for part in parts)
    except:
        return False

# Initialize FastAPI and MongoDB
app = FastAPI(title="KAIRO")
print("DBCONNECT")
uri = "mongodb+srv://dawdameet6338:ptsMMO0z8GtG3GsY@kairocluster.mhycn.mongodb.net/?retryWrites=true&w=majority&appName=kairocluster"
# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['fraud-detection-db']
fs = gridfs.GridFS(db)
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)
# Data preprocessing functions
def encode_categorical_columns(df):
    label_encoders = {}
    categorical_columns = ['user_id', 'location', 'transaction_type', 'merchant', 'description']

    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    return df, label_encoders

def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'user_id': np.random.choice(['user1', 'user2', 'user3'], size=num_samples),
        'amount': np.random.uniform(50, 5000, size=num_samples),
        'location': np.random.choice(['NY', 'California', 'India', 'Brazil'], size=num_samples),
        'transaction_type': np.random.choice(['purchase', 'refund', 'withdrawal'], size=num_samples),
        'merchant': np.random.choice(['amazon', 'eBay', 'Walmart'], size=num_samples),
        'device_info': np.random.choice([{'device_id': 'device1', 'device_type': 'mobile'}, {'device_id': 'device2', 'device_type': 'laptop'}], size=num_samples),
        'ip_address': np.random.choice(['192.168.1.1', '192.168.2.1'], size=num_samples),
        'description': np.random.choice(['electronics', 'clothing', 'groceries'], size=num_samples),
        'isFraud': np.random.choice([0, 1], size=num_samples)
    }
    return pd.DataFrame(data)

def preprocess_device_info(df):
    device_info_df = pd.json_normalize(df['device_info'])
    for col in device_info_df.columns:
        le = LabelEncoder()
        device_info_df[col] = le.fit_transform(device_info_df[col])
    df = df.join(device_info_df)
    df = df.drop(columns=['device_info'])
    return df

def preprocess_ip_address(df):
    le = LabelEncoder()
    df['ip_address'] = le.fit_transform(df['ip_address'])
    return df

def get_training_data():
    df = generate_synthetic_data(1000)
    device_info_df = pd.json_normalize(df['device_info'])
    for col in device_info_df.columns:
        le = LabelEncoder()
        device_info_df[col] = le.fit_transform(device_info_df[col])
    le_ip = LabelEncoder()
    df['ip_address'] = le_ip.fit_transform(df['ip_address'])
    df = df.drop(columns=['device_info']).join(device_info_df)
    df, label_encoders = encode_categorical_columns(df)
    label_encoders['ip_address'] = le_ip
    X = df.drop(columns=['isFraud'])
    y = df['isFraud']
    return X, y, label_encoders

# Pydantic model for transaction data
class Transaction(BaseModel):
    user_id: str
    amount: float
    location: str
    transaction_type: str
    merchant: str
    device_info: Optional[dict]
    ip_address: Optional[str]
    description: Optional[str]

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user1",
                "amount": 1299.99,
                "location": "NY",
                "transaction_type": "purchase",
                "merchant": "amazon",
                "device_info": {
                    "device_id": "device1",
                    "device_type": "mobile"
                },
                "ip_address": "192.168.1.1",
                "description": "electronics"
            }
        }

# Neural network models
class KairoAnalyst(nn.Module):
    def __init__(self, bert_model="bert-base-uncased"):
        super(KairoAnalyst, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(0.1)
        self.layer1 = nn.Linear(768, 256)
        self.layer2 = nn.Linear(256, 512)
        self.layer3 = nn.Linear(512, 1024)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs[0][:, 0, :]
        x = self.dropout(x)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class TransactionEncoder(nn.Module):
    def __init__(self):
        super(TransactionEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 264)
        )

    def forward(self, x):
        return self.encoder(x)

# Model manager
class M_Manager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.KairoFraudAnalystModel = self.load_or_create_model()
        self.transactionEncoder = self.load_or_create_encoder()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.randomForestModel = self.load_rfModel()
        self.label_encoders = {}
        if not hasattr(self.randomForestModel, "estimators_"):
            print("Training RandomForest model...")
            self.train_rf_model()

    def load_rfModel(self):
        try:
            model_info = db.models.find_one(
                {"model_type": "rfModel", "status": "active"},
                sort=[("version", -1)]
            )
            model_file = fs.get(model_info['model_file_id'])
            rf_model = joblib.load(model_file)
            if not hasattr(rf_model, "estimators_"):
                print("RandomForest model is not fitted, training now...")
            return rf_model
        except Exception as e:
            print("Error loading RandomForest model:", e)
            return RandomForestClassifier(n_estimators=100, random_state=42)

    def train_rf_model(self):
        df = generate_synthetic_data(1000)
        label_encoders = {}
        categorical_columns = ['user_id', 'location', 'transaction_type', 'merchant', 'description']
        for column in categorical_columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le
        device_info_df = pd.json_normalize(df['device_info'])
        for col in device_info_df.columns:
            le = LabelEncoder()
            device_info_df[col] = le.fit_transform(device_info_df[col])
        ip_le = LabelEncoder()
        df['ip_address'] = ip_le.fit_transform(df['ip_address'])
        label_encoders['ip_address'] = ip_le
        df = df.drop(columns=['device_info']).join(device_info_df)
        X = df.drop(columns=['isFraud'])
        y = df['isFraud']
        self.randomForestModel.fit(X, y)
        self.label_encoders = label_encoders
        print("RandomForest model trained successfully.")
        return label_encoders

    def load_or_create_model(self):
        try:
            model_file = fs.get(db.models.find_one(
                {"model_type": "kairoFraudAnalyst", "status": "active"},
                sort=[("version", -1)]
            )['model_file_id'])
            model = KairoAnalyst()
            model.load_state_dict(torch.load(model_file))
        except:
            model = KairoAnalyst()
        return model.to(self.device)

    def load_or_create_encoder(self):
        try:
            model_file = fs.get(db.models.find_one(
                {"model_type": "transactionEncoder", "status": "active"},
                sort=[("version", -1)]
            )['model_file_id'])
            model = TransactionEncoder()
            model.load_state_dict(torch.load(model_file))
        except:
            model = TransactionEncoder()
        return model.to(self.device)

    def decode_model_output(self, risk_score):
        if risk_score > 0.7:
            return "High risk transaction detected immediate flagging advised"
        elif risk_score > 0.4:
            return "Medium risk transaction detected additional verification advised"
        else:
            return "Low risk transaction proceed normally"

    def generateFraudAnalysis(self, transaction_data, risk_score):
        desc = transaction_data.get('description', '')
        encoding = self.tokenizer(
            desc,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        self.KairoFraudAnalystModel.eval()
        with torch.no_grad():
            out = self.KairoFraudAnalystModel(
                encoding['input_ids'].to(self.device),
                encoding['attention_mask'].to(self.device)
            )
        fraud_prob = torch.sigmoid(out.mean()).item()

        # Combine risk_score and fraud_prob
        unified_risk_score = (risk_score + fraud_prob) / 2
        print(f"RandomForest score: {risk_score}, BERT score: {fraud_prob}, Unified score: {unified_risk_score}")
        return self.decode_model_output(unified_risk_score)

# Initialize model manager
model_manager = M_Manager()

# Transaction processor
class TransactionProcessor:
    def __init__(self):
        X, _, label_encoders = get_training_data()
        self.feature_columns = list(X.columns)
        self.label_encoders = label_encoders
        self.required_fields = {'user_id', 'amount', 'location', 'transaction_type', 'merchant'}

    def validate_transaction(self, transaction_data):
        missing_fields = self.required_fields - set(transaction_data.keys())
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

    def preprocess_transaction(self, transaction_data):
        df = pd.DataFrame([transaction_data])
        df = preprocess_device_info(df)
        df = preprocess_ip_address(df)
        for column, encoder in self.label_encoders.items():
            if column in df.columns:
                try:
                    df[column] = encoder.transform(df[column])
                except ValueError:
                    print(f"Warning: New category in {column}, using fallback value")
                    df[column] = encoder.transform([encoder.classes_[0]])
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_columns]
        return df

# API endpoint
@app.post("/api/transactions/")
async def process_transaction(
    transaction: Transaction,
    backgroundTasks: BackgroundTasks
):
    try:
        processor = TransactionProcessor()
        transaction_dict = transaction.model_dump()

        processor.validate_transaction(transaction_dict)
        features_df = processor.preprocess_transaction(transaction_dict)

        try:
            # Get risk score and convert to Python float
            risk_score = float(model_manager.randomForestModel.predict_proba(features_df)[0][1])
            risk_score = adjust_risk_score(transaction_dict, risk_score)
            risk_score = float(risk_score)  # Ensure native float
        except ValueError as e:
            print(f"Prediction error: {str(e)}")
            print(f"Features shape: {features_df.shape}")
            print(f"Features columns: {features_df.columns}")
            raise HTTPException(
                status_code=500,
                detail="Error making prediction. Model may need retraining."
            )

        fraudAnalysis = model_manager.generateFraudAnalysis(transaction_dict, risk_score)

        transactionData = {
            **transaction_dict,
            'risk': risk_score,
            'isFraud': bool(risk_score > 0.54),  # Native Python bool
            'timestamp': pd.Timestamp.now(),  # Native Python datetime
            'fraudAnalysis': fraudAnalysis
        }

        result = db.transactions.insert_one(transactionData)

        response_data = {
            'transaction_id': str(result.inserted_id),
            'risk_score': risk_score,
            'fraud_analysis': fraudAnalysis,
            'investigation_needed': risk_score > 0.7,
            **{k: v for k, v in transaction_dict.items()}
        }

        return response_data

    except Exception as e:
        print(f"Error processing transaction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)






'''
eg.
{
  "user_id": "user1",
  "amount": 1000000.99,
  "location": "NY",
  "transaction_type": "purchase",
  "merchant": "amazon",
  "device_info": {
    "device_id": "device1",
    "device_type": "mobile"
  },
  "ip_address": "192.168.1.1",
  "description": "electronics"
}


'''
