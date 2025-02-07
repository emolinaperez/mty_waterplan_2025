import yaml

class Utils:

    @staticmethod
    def read_yaml_file(file_path):
        """
        Reads a YAML file and returns its contents as a Python dictionary.

        Args:
            file_path (str): The path to the YAML file to be read.

        Returns:
            dict: The contents of the YAML file as a dictionary.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            yaml.YAMLError: If there is an error parsing the YAML file.
        """
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)