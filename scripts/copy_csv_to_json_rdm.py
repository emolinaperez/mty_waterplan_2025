import os
import shutil
import time

start_time = time.time()

# --- Define Paths ---
SCRIPTS_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(SCRIPTS_DIR)
JSON_RDM_DIR = os.path.join(PARENT_DIR, 'json_RDM')
CSV_FILES_DIR = os.path.join(SCRIPTS_DIR, 'csv_files')

csv_files = ["anc_corrected.csv", "demand_corrected.csv", "projects_final_v4.csv", "supply_corrected.csv"]  # Replace with your CSV file names

# Iterate over each folder in the target directory
for folder_name in os.listdir(JSON_RDM_DIR):
    folder_path = os.path.join(JSON_RDM_DIR, folder_name)
    # Ensure it's a directory
    if os.path.isdir(folder_path):
        # Create a new directory in the folder
        csv_folder_path = os.path.join(folder_path, 'csv_files')
        os.makedirs(csv_folder_path, exist_ok=True)
        for csv_file in csv_files:
            source_file_path = os.path.join(CSV_FILES_DIR, csv_file)
            destination_file_path = os.path.join(csv_folder_path, csv_file)
            shutil.copy2(source_file_path, destination_file_path)  # Copies the file and preserves metadata
        print(f"Copied files to {csv_folder_path}")

print("File copying completed.")
finish_time = time.time()
print(f"Total execution time: {finish_time - start_time} secs")