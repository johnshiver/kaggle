import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

# Load the dataset
train_file_path = "/Volumes/HDD2/datasets/house-prices-advanced-regression-techniques/train.csv"
dataset_df = pd.read_csv(train_file_path)

# Drop the 'Id' column
dataset_df = dataset_df.drop('Id', axis=1)

# Separate target from features
X = dataset_df.drop('SalePrice', axis=1)
y = dataset_df['SalePrice']

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Preprocessing for numerical data: fill missing values with median and standardize
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data: fill missing values with most frequent and one-hot encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Preprocess the data
X_preprocessed = preprocessor.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train.toarray(), dtype=torch.float32)
X_test = torch.tensor(X_test.toarray(), dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Define the neural network with dropout
class HousePricePredictor(nn.Module):
    def __init__(self, input_dim):
        super(HousePricePredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)  # Add dropout with probability 0.5
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout after first layer
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model, loss function, and optimizer
input_dim = X_train.shape[1]
model = HousePricePredictor(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with early stopping
num_epochs = 1000
early_stopping_patience = 50
best_rmse = float('inf')
epochs_no_improve = 0

for epoch in range(num_epochs):
    model.train()
    
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Evaluate the model on the validation set
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        rmse = torch.sqrt(criterion(predictions, y_test)).item()
    
    if rmse < best_rmse:
        best_rmse = rmse
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'best_model.pth')  # Save the best model
    else:
        epochs_no_improve += 1
    
    if epochs_no_improve == early_stopping_patience:
        print(f'Early stopping at epoch {epoch+1}')
        break
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, RMSE: {rmse:.4f}')

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))

# Evaluate the final model
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    final_rmse = torch.sqrt(criterion(predictions, y_test)).item()
    print(f'Final RMSE on test set: {final_rmse:.4f}')

# Create the output file with predictions
test_file_path = "/Volumes/HDD2/datasets/house-prices-advanced-regression-techniques/test.csv"
test_df = pd.read_csv(test_file_path)
ids = test_df['Id']
test_df = test_df.drop('Id', axis=1)

# Preprocess the test data
X_test_preprocessed = preprocessor.transform(test_df)
X_test = torch.tensor(X_test_preprocessed.toarray(), dtype=torch.float32)

# Make predictions using the best model
model.eval()
with torch.no_grad():
    predictions = model(X_test).numpy()

# Create the output DataFrame
output_df = pd.DataFrame({
    'Id': ids,
    'SalePrice': predictions.flatten()
})

# Save the predictions to a CSV file
output_file_path = "predictions.csv"
output_df.to_csv(output_file_path, index=False)

print(f"Predictions saved to {output_file_path}")
