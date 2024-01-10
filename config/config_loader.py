import os
from dotenv import load_dotenv
import yaml

load_dotenv()


def load_config():
    project_root = os.environ.get('PROJECT_ROOT')
    config_path = os.path.join(project_root, 'config.yaml')

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    return config


