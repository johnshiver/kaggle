from data import *

dataset_df = load_dataset(TRAIN_FILE_PATH)
X_preprocessed, y, preprocessor = preprocess_features(dataset_df)

X_train, X_test, y_train, y_test = train_test_split(
    X_preprocessed, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
print(f"Train RMSE: {train_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")

test_df = pd.read_csv(TEST_FILE_PATH)
ids = test_df["Id"]
test_df = test_df.drop("Id", axis=1)

X_test_preprocessed = preprocess_test_data(test_df, preprocessor)

test_predictions = model.predict(X_test_preprocessed)

output_file_path = "random_forest_predictions.csv"
write_predictions(ids, test_predictions, output_file_path)
