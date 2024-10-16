import pandas as pd

import nfl_data_py as nfl

# Set display options
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)


# TODO: years should match our problem data
years = [2019]
# TODO: once we know all columns we need can optimize
# columns = []

pbp_data = nfl.import_pbp_data(years, downcast=True, cache=False, alt_path=None)


selected_columns = ["play_id", "game_id", "old_game_id_x", "rush", "pass", "play_type"]

# print(pbp_data[selected_columns].head(10))

# Define the game_id you want to filter by
game_id_filter = "2019_01_ATL_MIN"

# Filter the DataFrame to get only the plays from that game
filtered_plays = pbp_data[pbp_data["game_id"] == game_id_filter]

# Display the filtered DataFrame
print(filtered_plays[selected_columns])


# TODO: lets start by trying to predict run / pass just with existing column data, then add in more
