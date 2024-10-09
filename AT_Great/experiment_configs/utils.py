import yaml
import os
import ast

def get_configs(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config



def update_yaml_with_vals(config_data, property_path, new_value, join_path=False):
    """Update a value in a yaml file. The property_path is a string that represents the path to the property to be updated.

        Args:
            config_data (dict): The dictionary containing the yaml file data.
            property_path (str): The path to the property to be updated.
            new_value (str): The new value to be set.

        Returns:
            dict: The updated dictionary.

             update_yaml_with_vals that updates a value in a YAML file. Let's break down the code and understand what it does.

    The function starts by checking if the new_value is a string representation of a list or dictionary.
    If it is, the function converts it back to a list or dictionary using the ast.literal_eval() function.
    This is done to ensure that the updated value is of the correct data type.

    Next, the property_path is split into individual keys using the dot (.) as the separator.
    This allows the function to traverse through nested dictionaries within the config_data dictionary.

    The code then initializes a variable called sub_data with the value of config_data.
    This variable will be used to keep track of the current level of the nested dictionaries as we traverse through the keys.

    The function then iterates over all the keys except the last one in the keys list.
    For each key, it checks if it exists in the sub_data dictionary. If the key does not exist,
    a new empty dictionary is created at that key. This ensures that the nested structure is maintained.

    Finally, the last key in the keys list is used to set the new_value in the sub_data dictionary.
    After all the keys have been processed, the function returns the updated config_data dictionary.

    ========= Example =========
    config_data = {
        "app": {
            "name": "MyApp",
            "version": "1.0",
            "settings": {
                "debug": True,
                "timeout": 10
            }
        }
    }

    # ------- new input values -------
    property_path = "app.settings.timeout"
    new_value = 20

    # ------- update the yaml file -------
    updated_data = update_yaml_with_vals(config_data, property_path, new_value)
    print(updated_data)

    # ------- Output: -------
    {
        "app": {
            "name": "MyApp",
            "version": "1.0",
            "settings": {
                "debug": True,
                "timeout": 20
            }
        }
    }

    In this example, the function updates the value of the timeout property under the app.settings path to 20.
    The resulting config_data dictionary reflects this change.
    """
    # Check if the value is a string representation of a list or dictionary
    # If it is, convert it to a list or dictionary
    try:
        new_value = ast.literal_eval(new_value)
    except ValueError:
        pass

    keys = property_path.split(".")
    sub_data = config_data
    for key in keys[:-1]:
        if key not in sub_data:
            sub_data[key] = {}
        sub_data = sub_data[key]
    if join_path:
        temp = os.path.join(
            new_value, sub_data[keys[-1]]
        )  # new_value in this case is the root_path
        sub_data[keys[-1]] = temp
    else:
        sub_data[keys[-1]] = new_value
    return config_data


def update_yaml_with_unknown_args(config, unknown_args):
    for i in range(0, len(unknown_args), 2):  # iterate over the unknown args in a list
        config = update_yaml_with_vals(config, unknown_args[i][2:], unknown_args[i + 1])
    return config


def update_yaml_with_hyperparams(config, hyperparams):
    for key, value in hyperparams.items():
        # key = key.replace('-', '.')
        new_config = update_yaml_with_vals(config, key, value)
    return new_config


def update_yamlpaths_with_vals(config_data, property_path, new_value, join_path=False):
    keys = property_path.split(".")
    sub_data = config_data
    for key in keys[:-1]:
        if key not in sub_data:
            sub_data[key] = {}
        sub_data = sub_data[key]
    if join_path:
        temp = os.path.join(new_value)
        # if sub_data[keys[-1]] in [None, 'None']: 
        #     temp = os.path.join(new_value)
        # else: 
        #     temp = os.path.join(
        #         new_value, sub_data[keys[-1]]
        #     )  # new_value in this case is the root_path
        sub_data[keys[-1]] = temp
        print("New path:\n", sub_data[keys[-1]])
    return config_data


def update_datapaths(config, join_path=True, **kwargs):
    if not kwargs:
        # ROOT dirs for source and target data and fold configs
        kwargs = {
            "source.data_path": "/home/kasia/AT_Great/AT_Great/data",
            "target.data_path": "/home/kasia/AT_Great/AT_Great/data",
            "fold_config": "/home/kasia/AT_Great/AT_Great/experiment_configs/fold_generator",
            "fold_params": "/home/kasia/AT_Great/AT_Great/experiment_configs/fold_params",
        }

    for key, value in kwargs.items():
        new_config = update_yamlpaths_with_vals(config, key, value, join_path=join_path)
    return new_config

def config_assert_datadir(config, parsed_source_sub=None, parsed_target_sub=None):
    if config['name'] == 'at_great':
        if config['feats'] == 'aug':
            _dir ='processed_data_144x256feats_augstride10and10'
            _dir_fold_config = 'up_processed_data_144x256feats_augstride10and10'
            _dir_fold_params = 'up_processed_data_144x256feats_augstride10and10'
        elif config['feats'] == 'ftd_base':
            _dir = "processed_data_16x9feats"
            _dir_fold_config = 'up_processed_data_16x9feats'
            _dir_fold_params = 'up_processed_data_16x9feats'
    # print("Parsed source sub: ", parsed_source_sub, ' type: ', type(parsed_source_sub))
    # print("Parsed target sub: ", parsed_target_sub)
    if parsed_source_sub and parsed_target_sub:
        source_dir = os.path.join(_dir, 'participant_' +parsed_source_sub)
        _dir_fold_config = os.path.join(_dir_fold_config, "fold_generator", 'participant_' +parsed_source_sub +'.yaml')
        _dir_fold_params = os.path.join(_dir_fold_params,  "fold_params", 'participant_' +parsed_source_sub +'.yaml')
        target_dir = os.path.join(_dir, 'participant_' +parsed_target_sub)
        return source_dir, _dir_fold_config, _dir_fold_params, target_dir
    else:
        return _dir, _dir_fold_config, _dir_fold_params
    
def assert_config_properties(config):
    
    # assert if Many-to-One domain adaptation then compute_norm is True
    if len(config['dataset']['config']['source']['pos']) > 1:
        assert config['dataset']['config']['compute_norms'] == True, "For Many-to-One domain adaptation compute_norm should be True"
    elif len(config['dataset']['config']['source']['pos']) == 1:
        assert config['dataset']['config']['compute_norms'] == False, "For One-to-One domain adaptation compute_norm should be False and read params from file."
    
    # assert model input based on feats
    if config['dataset']['config']['feats'] == 'ftd_base':
        assert config['model']["hyperparameters"]['in_channels'][0] == 16, "For ftd_base feats the input should be 16"
    elif config['dataset']['config']['feats'] == 'aug':
        assert config['model']["hyperparameters"]['in_channels'][0] == 144, "For aug feats the input should be 144"

    

   
# def get_nested_value(d, keys):
#     for key in keys:
#         if key in d:
#             d = d[key]
#         else:
#             return None
#     return d


# def set_nested_value(d, keys, value):
#     for key in keys[:-1]:
#         d = d.setdefault(key, {})
#     d[keys[-1]] = value


# def find_matching_key(config, hpo_key):
#     keys = hpo_key.split(".")
#     current = config
#     matched_keys = []
#     for key in keys:
#         if isinstance(current, dict) and key in current:
#             matched_keys.append(key)
#             current = current[key]
#         else:
#             break
#     return matched_keys if matched_keys else None
