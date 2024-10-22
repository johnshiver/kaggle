# Define variables for the database connection
DB_USER = your_username
DB_PASSWORD = your_password
DB_HOST = localhost
DB_PORT = 5432
DB_NAME = your_database
TABLE_NAME = your_table_name
CSV_DIR = /Volumes/HDD2/datasets/nfl-big-data-bowl-2025

# Command to import CSV into PostgreSQL
import_csv:
	@echo "Importing CSV into PostgreSQL..."
	@csvsql --db postgresql://$(DB_USER):$(DB_PASSWORD)@$(DB_HOST):$(DB_PORT)/$(DB_NAME) \
	        --table $(TABLE_NAME) --insert $(CSV_FILE)
	@echo "CSV successfully imported into table $(TABLE_NAME)!"

# Example target for creating and importing CSV in one step
create_and_import:
	@echo "Creating table and importing CSV into PostgreSQL..."
	@csvsql --db postgresql://$(DB_USER):$(DB_PASSWORD)@$(DB_HOST):$(DB_PORT)/$(DB_NAME) \
	        --table $(TABLE_NAME) --insert $(CSV_FILE)
	@echo "Table created and CSV imported into table $(TABLE_NAME)!"