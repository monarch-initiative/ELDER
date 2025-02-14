def possible_emb_models():
    model_list =["ada002", "bge-m3", "large3", "small3", "nomic", "mxbai-l"]
    for model in model_list:
        print(model)
if __name__ == "__main__":
    possible_emb_models()