from statistics import LinearRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


# Load the dataset
play_data_path = "/Volumes/HDD2/datasets/nfl-big-data-bowl-2025/plays.csv"
nfl_py_data_path = "/Volumes/HDD2/datasets/nfl-big-data-bowl-2025/play_by_play_2022.csv"
play_data = pd.read_csv(play_data_path)
nfl_py_data = pd.read_csv(nfl_py_data_path)

merged_data = pd.merge(
    play_data,
    nfl_py_data,
    left_on=["playId", "gameId"],
    right_on=["play_id", "old_game_id_x"],
    how="inner",
)
filtered_data = merged_data[(merged_data["pass"] == 1) | (merged_data["rush"] == 1)]

# set custom column
filtered_data["outcome"] = np.where(
    filtered_data["rush"] == 1,
    "run",
    np.where(filtered_data["pass"] == 1, "pass", None),
)


# List of pre-snap feature columns (replace with your actual column names)
pre_snap_columns = [
    # Game context
    "quarter",
    "down_x",  # Current down
    "yardsToGo",  # Yards needed for a first down
    "yardlineNumber",  # Yard line number
    "yardlineSide",  # Side of the field
    "absoluteYardlineNumber",  # Absolute yard line number
    "gameClock",  # Time remaining in the quarter
    "score_differential",  # Difference in score between teams
    "preSnapHomeScore",
    "preSnapVisitorScore",
    "game_half",  # First half or second half
    "quarter_seconds_remaining",
    "half_seconds_remaining",
    "game_seconds_remaining",
    "goal_to_go",  # Indicates if it's a goal-to-go situation
    # Team information
    # "home_team",
    # "away_team",
    # "posteam",  # Possession team
    # "defteam",  # Defensive team
    # "posteam_type",  # Home or away
    "side_of_field",  # Which team's side of the field
    "home_timeouts_remaining",
    "away_timeouts_remaining",
    # "posteam_timeouts_remaining",
    # "defteam_timeouts_remaining",
    # Formation and personnel
    "offenseFormation",
    # "offense_personnel",
    # "defense_personnel",
    # "defenders_in_box",
    # "number_of_pass_rushers",
    # "receiverAlignment",
    # "n_offense",  # Number of offensive players
    # "n_defense",  # Number of defensive players
    # "shotgun",  # Indicates if the play is from shotgun formation
    "no_huddle",  # Indicates if the offense is in no-huddle
    # "playAction",  # Indicates if play-action is planned
    # "pff_runConceptPrimary",
    # "pff_runConceptSecondary",
    # "pff_runPassOption",
    # "pff_passCoverage",
    # "pff_manZone",
    # "defense_man_zone_type",
    # "defense_coverage_type",
    # Situational factors
    # "playClockAtSnap",  # Play clock time at snap
    "drive",  # Current drive number
    "time",  # Time of play
    "home_opening_kickoff",  # Indicates if home team is receiving opening kickoff
    "weather",
    "temp",
    "wind",
    "roof",  # Type of stadium roof
    "surface",  # Type of playing surface
    "stadium",
    "stadium_id",
    "game_stadium",
    # Probabilities and expected values
    # "preSnapHomeTeamWinProbability",
    # "preSnapVisitorTeamWinProbability",
    # "expectedPoints",
    # "no_score_prob",
    # "fg_prob",
    # "safety_prob",
    # "td_prob",
    # "extra_point_prob",
    # "two_point_conversion_prob",
    # "xpass",  # Expected pass probability
    # "cp",  # Completion probability
    # "cpoe",  # Completion percentage over expected
    # Coaching and strategy
    # "home_coach",
    # "away_coach",
    # "playAction",  # Whether play-action is planned
]

X = filtered_data[pre_snap_columns]
y = filtered_data["outcome"]


# TODO: look into this a bit more

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

# Preprocessing pipelines
numerical_transformer = SimpleImputer(strategy="mean")
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

clf = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        # ("classifier", RandomForestClassifier(random_state=25)),
        ("classifier", GradientBoostingClassifier(random_state=42)),
        # ("classifier", LogisticRegression(max_iter=10000)),
    ]
)

# Train the model
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))


cross_val_scores = cross_val_score(clf, X, y, cv=5)
print("Cross-validated accuracy: %.2f" % cross_val_scores.mean())


# voting classifier code

# voting_clf = VotingClassifier(
#     estimators=[
#         ("rf", RandomForestClassifier(random_state=42)),
#         (
#             "xgb",
#             XGBClassifier(
#                 use_label_encoder=True, eval_metric="mlogloss", random_state=42
#             ),
#         ),
#         ("lr", LogisticRegression(max_iter=1000)),
#     ],
#     voting="soft",
# )

# clf = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", voting_clf)])

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # Train the model
# clf.fit(X_train, y_train)

# # Predict and evaluate
# y_pred = clf.predict(X_test)
# print(classification_report(y_test, y_pred))
