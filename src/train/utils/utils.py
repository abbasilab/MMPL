import yaml

def get_config_from_dataset(dataset):
    config_path = "configs/" + dataset + "/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

def get_train_path_from_dataset(dataset):
    return "data/" + dataset + "/processed/train.ts"

def get_test_path_from_dataset(dataset):
    return "data/" + dataset + "/processed/test.ts"