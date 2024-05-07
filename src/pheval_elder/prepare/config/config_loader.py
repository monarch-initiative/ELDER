from pydantic import BaseModel, Field


class ElderConfig(BaseModel):
    """
    Class for defining ELDER configurations in tool_specific_configurations field,
    within the input_dir config.yaml.

    Args:
        chroma_db_path (str): Path to ChromaDB path.
    """
    chroma_db_path: str = Field(...)

# def load_config():
#     project_root = os.environ.get('PROJECT_ROOT')
#     print(f"project root{project_root}")
#     config_path = os.path.join(project_root, 'elder_config.yaml')
#     print(config_path)
#
#     with open(config_path, 'r') as file:
#         config = yaml.safe_load(file)
#
#     return config
