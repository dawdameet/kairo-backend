from fastapi import *
from pymongo import MongoClient
import gridfs
from pydantic import BaseModel
from typing import Optional, List
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
def encode_categorical_columns(df):
    label_encoders = {}
    categorical_columns = ['user_id', 'location', 'transaction_type', 'merchant', 'description']
    
    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    return df, label_encoders
app=FastAPI(title="KAIRO")
client=MongoClient("mongodb://localhost:27017")
db=client['fraud-detection']
fs=gridfs.GridFS(db)
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
class Transaction(BaseModel):
    user_id: str
    amount: float
    transaction_type: str
    location: str
    merchant: str
    device_info: Optional[dict]
    ip_address: Optional[str]
    description: Optional[str]
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
        self.encoder=nn.Sequential(
            nn.Linear(5,64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128,264) 
        )
    def forward(self, x):
        return self.encoder(x)
class M_Manager:
    def __init__(self):
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.KairoFraudAnalystModel=self.load_or_create_model()
        self.transactionEncoder=self.load_or_create_encoder()
        self.tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
        self.randomForestModel=self.load_rfModel()
        self.label_encoders={}
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
            model_file=fs.get(db.models.find_one(
                {"model_type": "kairoFraudAnalyst", "status": "active"},
                sort=[("version",-1)]
            )['model_file_id'])
            model=KairoAnalyst()
            model.load_state_dict(torch.load(model_file))
        except:
            model=KairoAnalyst()
        return model.to(self.device)
    def load_or_create_encoder(self):
        try:
            model_file=fs.get(db.models.find_one(
                {"model_type": "transactionEncoder", "status": "active"},
                sort=[("version",-1)]
            )['model_file_id'])
            model=TransactionEncoder()
            model.load_state_dict(torch.load(model_file))
        except:
            model=TransactionEncoder()
        return model.to(self.device)
    def load_rfModel(self):
        try:
            model_info=db.models.find_one(
                {"model_type": "rfModel", "status": "active"},
                sort=[("version",-1)]
            )
            model_file=fs.get(model_info['model_file_id'])
            return joblib.load(model_file)
        except:
            return RandomForestClassifier(n_estimators=100, random_state=42)
    def train_models(self, training_data):        
        X=training_data
        y=training_data['isFraud']
        self.randomForestModel.fit(X,y)
        self.trainKairoModel(training_data)
        self.save_model()
    def trainKairoModel(self, training_data):
        desc=training_data['description'].toList()
        encodings = self.tokenizer(desc, truncation=True, padding=True,max_length=512, return_tensors="pt")
        optr = torch.optim.Adam(list(self.KairoFraudAnalystModel.parameters()) +list(self.transactionEncoder.parameters()))
        loss_fn=nn.MSELoss()
        for epoch in range(0, 10):
            self.KairoFraudAnalystModel.train()
            self.transactionEncoder.train()
            optr.zero_grad()
            fraud_outs=self.KairoFraudAnalystModel(
                encodings['in_ids'].to(self.device),
                encodings['attn_mask'].to(self.device)
            )
            loss=loss_fn(fraud_outs, torch.zeros_like(fraud_outs))
            loss.backward()
            optr.step()
    def save_model(self):
        randomForest_model_bytes=joblib.dump(self.randomForestModel)
        randomForest_model_id=fs.put(randomForest_model_bytes)
        fraudBytes=io.BytesIO()
        torch.save(self.KairoFraudAnalystModel.state_dict(), fraudBytes)
        fraudId=fs.put(fraudBytes.getvalue())
        encoderBytes=io.BytesIO()
        torch.save(self.transactionEncoder.state_dict(), encoderBytes)
        encoderId=fs.put(encoderBytes.getvalue())
        version = self.get_next_version()
        timestamp = pd.Timestamp.now()

        db.models.insert_many([
            {
                "version": version,
                "timestamp": timestamp,
                "model_type": "rfModel",
                "model_file_id": randomForest_model_id,
                "status": "active"
            },
            {
                "version": version,
                "timestamp": timestamp,
                "model_type": "kairoFraudAnalyst",
                "model_file_id": fraudId,
                "status": "active"
            },
            {
                "version": version,
                "timestamp": timestamp,
                "model_type": "transactionEncoder",
                "model_file_id": encoderId,
                "status": "active"
            }
        ])
    def generateFraudAnalysis(self,transaction_data):
        desc=transaction_data.get('description', '')
        encoding=self.tokenizer(
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
        analysis=self.decode_model_output(out)
        return analysis
    def decode_model_output(self,out):
        fraud_prob=torch.sigmoid(out.mean()).item()           
        if(fraud_prob>0.7):
            return "High risk transaction detected immidiate flagging adviced"
        elif(fraud_prob>0.4):
            return "Medium risk transaction detected additional verification adviced"
        else:
            return "low risk transaction proceed normally"
from sklearn.preprocessing import LabelEncoder
def encode_categorical_features(transaction, label_encoders):
    encoded_features = []
    for column in ['user_id', 'location', 'transaction_type', 'merchant', 'description']:
        if column in transaction:
            if column not in label_encoders:
                label_encoders[column] = LabelEncoder()
            encoded_features.append(label_encoders[column].fit_transform([transaction[column]])[0])
    return encoded_features, label_encoders
def extract_features(transaction, user_history, label_encoders):
    features = []
    categorical_columns = ['user_id', 'location', 'transaction_type', 'merchant', 'description']
    for column in categorical_columns:
        if column not in label_encoders:
            label_encoders[column] = LabelEncoder()
        features.append(label_encoders[column].transform([transaction.get(column, 'unknown')])[0])
    features.append(transaction['amount']) 
    if user_history:
        history_df = pd.DataFrame(user_history)
        features.append(history_df['amount'].mean()) 
        features.append(len(history_df))  
        features.append((pd.Timestamp.now() - pd.to_datetime(history_df['timestamp'].max())).total_seconds())
    else:
        features.extend([0, 0, 0]) 
    return np.array(features).reshape(1, -1), label_encoders

model_manager=M_Manager()   

@app.post("/api/transactions/")
async def proces_transaction(
    transaction: Transaction,
    backgroundTasks: BackgroundTasks
):
    try:
        transaction_dict = transaction.model_dump()
        user_history = list(db.transactions.find({"user_id": transaction.user_id}))
        label_encoders = model_manager.label_encoders
        features, updated_label_encoders = extract_features(transaction_dict, user_history, label_encoders)
        risk = model_manager.randomForestModel.predict_proba(features)[0][1]
        fraudAnalysis = model_manager.generateFraudAnalysis(transaction_dict)
        transactionData = {
        **transaction_dict,
        "risk": float(risk),
        "isFraud": bool(risk > 0.7),
        "timestamp": pd.Timestamp.now(),
        "fraudAnalysis": fraudAnalysis
        }
        for key, value in transactionData.items():
            if hasattr(value, 'item'):
                transactionData[key] = value.item()
        result = db.transactions.insert_one(transactionData)
        
        backgroundTasks.add_task(
            # update_model_with_transaction,
            transactionData
        )
        recentData={
            "transaction_id": str(result.inserted_id),
            "risk_score": risk,
            "fraud_analysis": fraudAnalysis,
            "investigation_needed": bool(risk > 0.7),
            **transactionData
        }
        return recentData
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=8000)    