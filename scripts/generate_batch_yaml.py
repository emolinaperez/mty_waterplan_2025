import yaml
import os
# Define the total number of directories and directories per file
total_directories = 972
batch_size = 81

# --- Define Paths ---
SCRIPTS_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(SCRIPTS_DIR)
CONFIG_DIR = os.path.join(PARENT_DIR, 'config')
YAML_DIR = os.path.join(CONFIG_DIR, 'yaml')

# Loop to create 100 YAML files
for i in range(12):
    # Calculate the starting and ending directory numbers for this batch
    start = i * batch_size + 1
    end = min(start + batch_size, total_directories + 1)

    # Create a list of directory names for this batch
    directories = [f"po{j}" for j in range(start, end)]

    # Define the YAML file path
    yaml_filename = os.path.join(YAML_DIR, f"batch_{i+1}.yaml")
    
    # Write the directories to the YAML file
    with open(yaml_filename, 'w') as file:
        yaml.dump({"directories": directories}, file)

    # print(f"Created {yaml_filename} with directories: {directories}")
print('Done!')
