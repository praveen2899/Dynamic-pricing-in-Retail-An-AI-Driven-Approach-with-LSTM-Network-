import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from sklearn.preprocessing import LabelEncoder

def load_resources(supermarket):

    demand_df = pd.read_csv('myapp/dataset/Demand_Dataset1.csv')
    if supermarket == 'ASDA':
        df = pd.read_csv('myapp/dataset/Updated_Common_Data_ASDA.csv')
        model = load_model('myapp/models/ASDA_lstm_model.h5', compile=False)
        with open('myapp/models/ASDA_label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
    elif supermarket == 'Morrisons':
        df = pd.read_csv('myapp/dataset/Updated_Common_Data_Morrisons.csv')
        model = load_model('myapp/models/Morrisons_lstm_model1.h5', compile=False)
        with open('myapp/models/Morrisons_label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
    elif supermarket == 'Tesco':
        df = pd.read_csv('myapp/dataset/Updated_Common_Data_Tesco.csv')
        model = load_model('myapp/models/Tesco_lstm_model1.h5', compile=False)
        with open('myapp/models/Tesco_label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
    elif supermarket == 'Sains':
        df = pd.read_csv('myapp/dataset/Updated_Common_Data_Sains.csv')
        model = load_model('myapp/models/Sains_lstm_model1.h5', compile=False)
        with open('myapp/models/Sains_label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
    else:
        raise ValueError("Supermarket not recognized")
    
    return df, model, label_encoders, demand_df

def prepare_latest_sequence(df, product_name, sequence_length, label_encoders):
    df_product = df[df['names'] == product_name]
    df_product = df_product.sort_values(by=['date'])
    latest_sequence = df_product.tail(sequence_length)
    
    if len(latest_sequence) < sequence_length:
        padding = sequence_length - len(latest_sequence)
        latest_sequence = pd.concat([df_product] * (padding + 1), ignore_index=True).tail(sequence_length)
    

    for column, le in label_encoders.items():
        if column in latest_sequence.columns:
            if set(latest_sequence[column].unique()).difference(set(le.classes_)):
             
                new_classes = list(set(latest_sequence[column].unique()).difference(set(le.classes_)))
                le.classes_ = np.append(le.classes_, new_classes)
            latest_sequence[column] = le.transform(latest_sequence[column])
    
 
    all_features = list(label_encoders.keys()) + ['unit_price']
    for feature in all_features:
        if feature not in latest_sequence.columns:
            latest_sequence[feature] = 0  
    

    latest_sequence = latest_sequence[all_features]
    
    
    X_latest = latest_sequence.drop(columns=['unit_price']).select_dtypes(include=[np.number])
    
 
    expected_features = 11
    if X_latest.shape[1] < expected_features:
        padding = expected_features - X_latest.shape[1]
        X_latest = np.pad(X_latest.values, ((0, 0), (0, padding)), mode='constant')
    elif X_latest.shape[1] > expected_features:
        X_latest = X_latest.iloc[:, :expected_features]
    
    return X_latest

def predict_prices(supermarket, product_name, sequence_length, num_days, weekday):
    """
    Predict prices for a given product, sequence length, number of days, and weekday for a selected supermarket.
    """

    df, model, label_encoders, demand_df = load_resources(supermarket)
    

    latest_sequence_df = prepare_latest_sequence(df, product_name, sequence_length, label_encoders)
    

    X_latest = latest_sequence_df.reshape(1, sequence_length, latest_sequence_df.shape[1])
    X_latest = X_latest.astype(np.float32)
    
    predictions = []
    for _ in range(num_days):
        next_day_price = model.predict(X_latest)
        predictions.append(next_day_price[0][0])
        
        new_sequence = np.zeros((sequence_length, X_latest.shape[2]))
        new_sequence[0:-1] = X_latest[0, 1:, :]
        new_sequence[-1] = np.append(X_latest[0, -1, :-1], next_day_price[0][0])
        X_latest = new_sequence.reshape(1, sequence_length, X_latest.shape[2])
    

    dynamic_prices = [calculate_dynamic_price(price, supermarket, product_name, weekday) for price in predictions]
    
    return predictions, dynamic_prices
 



def calculate_dynamic_price(predicted_price, supermarket, product_name, weekday):
    weekday_mapping = {
        'Monday': 0,
        'Tuesday': 1,
        'Wednesday': 2,
        'Thursday': 3,
        'Friday': 4,
        'Saturday': 5,
        'Sunday': 6
    }
    
 
    weekday_num = weekday_mapping.get(weekday, 0)  
    print(predicted_price)
    print(supermarket)
    print(product_name)
    print(weekday, "mapped to", weekday_num)
    
    demand_df = pd.read_csv('myapp/dataset/Demand_Dataset1.csv')
    product_demand = demand_df[(demand_df['product'] == product_name) & (demand_df['weekday'] == weekday_num)]
    
    print(product_demand)
    

    supermarket_sales_mean = product_demand[product_demand['supermarket'] == supermarket]['total_sale'].mean()
    print("Supermarket sales mean:", supermarket_sales_mean)
    

    competitor_sales_mean = product_demand[product_demand['supermarket'] != supermarket]['total_sale'].mean()
    print("Competitor sales mean:", competitor_sales_mean)
    
    if supermarket_sales_mean > 0:
        price_adjustment_factor = competitor_sales_mean / (supermarket_sales_mean + 1)
        dynamic_price = predicted_price * (1 + price_adjustment_factor)
    else:
        dynamic_price = predicted_price
    
    return dynamic_price
