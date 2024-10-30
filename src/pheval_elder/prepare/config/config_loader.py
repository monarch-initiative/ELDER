import os
import yaml
from pathlib import Path

def load_config():
    project_root = Path(__file__).resolve().parents[4]
    config_path = os.path.join(project_root, 'elder_config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    load_config()

if __name__ == "__main__":
    main()