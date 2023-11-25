import torch
import yaml

from src.comparisons.one_stage.model import AutoencoderPrototypeModel
from src.models.encoding import Encoder
from src.models.single_variable_prototypes import SingleVariablePrototypesWrapper
from src.models.multivariable_prototypes import MultivariableModule

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_config_from_dataset(dataset):
    config_path = "configs/" + dataset + "/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

def get_comparison_config_from_dataset(comparison, dataset):
    config_path = "configs/" + comparison + "/" + dataset + "/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

def get_train_path_from_dataset(dataset):
    return "data/" + dataset + "/processed/train.ts"

def get_val_path_from_dataset(dataset):
    return "data/" + dataset + "/processed/val.ts"

def get_test_path_from_dataset(dataset, train=True):
    if dataset.startswith("simulated") and train:
        return "data/" + dataset + "/processed/val.ts"
    return "data/" + dataset + "/processed/test.ts"

def load_encoders(config):
    encoding_config = config['encoding']
    encoders = []
    for i in range(config['num_variables']):
        encoder = Encoder(
            input_dim=encoding_config['input_dim'],
            hidden_dim=encoding_config['hidden_dim'],
            latent_dim=encoding_config['latent_dim']
        )
        encoder.load_state_dict(torch.load(encoding_config['save_dir'] + "encoder" + str(i+1) + ".pth", map_location=device))
        encoders.append(encoder)
    return encoders

def load_single_variable_prototypes_wrapper(config):
    encoding_config = config['encoding']
    single_variable_prototypes_config = config['single_variable_prototypes']
    encoders = load_encoders(config)
    wrapper = SingleVariablePrototypesWrapper(
        encoders=encoders,
        num_variables=config['num_variables'],
        num_classes=config['num_classes'],
        num_prototypes=single_variable_prototypes_config['num_prototypes'],
        latent_dim=encoding_config['latent_dim'],
        num_layers=single_variable_prototypes_config['num_layers']
    )
    save_name = single_variable_prototypes_config['save_dir'] + "single_variable_prototypes.pth"
    wrapper.load_state_dict(torch.load(save_name, map_location=device))
    return wrapper

def load_multivariable_prototypes(config):
    single_variable_prototypes_config = config['single_variable_prototypes']
    multivariable_config = config['multivariable_prototypes']
    wrapper = load_single_variable_prototypes_wrapper(config)
    multivariable_prototypes = MultivariableModule(
        wrapper=wrapper,
        num_classes=config['num_classes'],
        num_variables=config['num_variables'],
        num_sv_prototypes=single_variable_prototypes_config['num_prototypes'],
        num_layers=multivariable_config['num_layers']
    )
    save_name = multivariable_config['save_dir'] + "multivariable_prototypes.pth"
    multivariable_prototypes.load_state_dict(torch.load(save_name, map_location=device))
    return multivariable_prototypes

def load_one_stage_model(config):
    one_stage_config = config['one_stage']
    model = AutoencoderPrototypeModel(
        input_dim=one_stage_config['input_dim'],
        hidden_dim=one_stage_config['hidden_dim'],
        latent_dim=one_stage_config['latent_dim'],
        num_prototypes=one_stage_config['num_prototypes'],
        seq_len=one_stage_config['seq_len'],
        num_classes=one_stage_config['num_classes'],
        num_layers=one_stage_config['num_layers']
    )
    save_name = one_stage_config['save_dir'] + "model.pth"
    model.load_state_dict(torch.load(save_name, map_location=device))
    return model

def get_class_to_pattern_map():
    class_to_pattern_map = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                class_to_pattern_map.append([i, j, k])
    return torch.Tensor(class_to_pattern_map)

def get_patterns_from_labels(labels):
    class_to_pattern_map = get_class_to_pattern_map()
    patterns = []
    for label in labels:
        patterns.append(class_to_pattern_map[label])
    return patterns

def get_single_variable_patterns_from_labels(labels, var):
    class_to_pattern_map = get_class_to_pattern_map()
    patterns = []
    for label in labels:
        patterns.append(class_to_pattern_map[label][var])
    return patterns