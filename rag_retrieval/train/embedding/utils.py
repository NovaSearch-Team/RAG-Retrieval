from transformers import AutoConfig, AutoModel
from pathlib import Path
from safetensors.torch import load_file, save_file
import re


def get_safetensors_keys(model_name_or_path):
    path = Path(model_name_or_path)
    safetensors_files = list(path.rglob('*.safetensors'))
    state_dict = {}
    for file in safetensors_files:
        state_dict.update(load_file(file))
    return state_dict.keys()

def get_model_keys(model_name_or_path):
    config = AutoConfig.from_pretrained(model_name_or_path)
    model = AutoModel.from_config(config)
    return model.state_dict().keys()

def fix_sub_safetensors(model_name_or_path):
    path = Path(model_name_or_path)
    safetensors_files = list(path.glob('*.safetensors'))
    state_dict = {}
    for file in safetensors_files:
        state_dict.update(load_file(file))
    pattern = re.compile(r'^\d+_')
    for sub_dir in path.iterdir():
        if sub_dir.is_dir() and pattern.match(sub_dir.name):
            prefix = sub_dir.name.split('_')[0] + '.'
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith(prefix):
                    new_state_dict[k[len(prefix):]] = v
            if new_state_dict:
                save_file(new_state_dict, sub_dir / "model.safetensors")

def fix_safetensors(model_name_or_path, original_keys=None, mapping_rule=None):
    path = Path(model_name_or_path)
    safetensors_files = list(path.glob('*.safetensors'))
    for file in safetensors_files:
        state_dict = load_file(file)
        new_state_dict = {}
        for k, v in state_dict.items():
            for original_key in original_keys:
                if mapping_rule is None:
                    if original_key in k:
                        new_state_dict[original_key] = v
                        break
                else:
                    if mapping_rule(original_key, k):
                        new_state_dict[original_key] = v
                        break
        save_file(new_state_dict, file)
