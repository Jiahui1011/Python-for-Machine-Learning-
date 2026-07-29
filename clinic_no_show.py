import pandas as pd
from sklearn.tree import DecisionTreeClassifier # method
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score # check accuracy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. load data 
file_path = '/Users/jiahui/Documents/Document /Machine learning_python class/Independent_case/Dataset.xlsx'
df = pd.read_excel(file_path)
print(df.columns)

print('\nfirst 5 rows:')
print(df.head())

print('\nRows and columns:')
print(df.shape)

print('\nColumns name:')
print(df.columns)

print('\nColumn indormation:')
print(df.info())

print('\nSummary statistics:')
print(df.describe())

print('\nExact missing value:')
print(df.isnull().sum())

print('\nExact duplicate rows:')
print(df.duplicated().sum())

columns_to_check = [
    'sms_reminder', 
    'chronic_condition', 
    'appointment_day', 
    'time_slot', 
    'insurance_type',
    'missed_appointment'
]

for col in columns_to_check:
    print('------------')
    print(df[col].value_counts())

df_clean = df.copy()
print(df_clean.shape)

# clean column names
df_clean.columns = df_clean.columns.str.strip()

# replace common missing value symbols
missing_value = ['', ' ', 'N/A', 'na', 'n/a', '-', '?']
print(df_clean)

# remove duplicated records
print(df_clean.shape[0])
df_clean = df_clean.drop_duplicates() 
print(df_clean.shape)

df_clean = df_clean.drop_duplicates(subset = ['appointment_id'], keep='first') 
print(df_clean.shape)

# clean numerical columns 
print(df_clean.info())
num_columns = [
    'age',
    'waiting_days', 
    'previous_no_shows', 
    'distance_to_clinic_km'
]

for col in num_columns:
    df_clean[col] = (
        df_clean[col]
        .astype(str) 
        .str.replace('$', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.strip()
    )

    df_clean[col] = pd.to_numeric(df_clean[col], errors = 'coerce')
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
   
    cat_columns = [
    'sms_reminder',
    'insurance_type'
    ]

for col in cat_columns:
    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

print("Duplicate rows remaining:")
print(df_clean.duplicated().sum())

print("Missing values remaining:")
print(df_clean.isnull().sum())

X = df_clean.drop(
    columns=['appointment_id', 'booking_date', 'missed_appointment'], errors='ignore'
    )
y = df_clean['missed_appointment']

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y,
    test_size=0.3,
    random_state=42
)

clinic_model = DecisionTreeClassifier(random_state=42)
clinic_model.fit(X_train, y_train)

clinic_prediction = clinic_model.predict(X_test)

print(len(X_train))

print(len(X_test))

correct_predictions = (clinic_prediction == y_test).sum()

print(correct_predictions)

accuracy = accuracy_score(y_test, clinic_prediction)
print('Accuracy:', accuracy)