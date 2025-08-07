
import os
import huggingface_hub
from utils.ssl import no_ssl_verification
from utils import resolve_vars, dict_to_namespace
import yaml

def download(config):

    with no_ssl_verification():

        huggingface_hub.snapshot_download(config.HF_model_ticker, 
                        local_dir=config.sft.model_name_or_path,
                        local_dir_use_symlinks=False, 
                        token=config.HF_token,
                        resume_download=False, 
                        force_download=False)
        
        
MODELS = [ 
        {"model_provider":"hf-internal-testing", "model_name":"tiny-random-GPT2"},
           #{"model_provider":"Qwen", "model_name":"Qwen2.5-Math-1.5B-Instruct"},
           #{"model_provider":"Qwen", "model_name":"Qwen2.5-Math-7B-Instruct"},
           #{"model_provider":"Qwen", "model_name":"Qwen2.5-Math-72B-Instruct"} 
            ]

def download_all_models(config, models):
    scratch_path = config.scratch_model_path
    os.makedirs(scratch_path, exist_ok=True)

    with no_ssl_verification():

        for model in MODELS:
            model_provider, model_name = model["model_provider"], model["model_name"]
            print( "Downloading model {}...".format(model_name) )
            dest = os.path.join(scratch_path, "{}".format(model_name) )
            huggingface_hub.snapshot_download(
                repo_id="{}/{}".format(model_provider,model_name),
                local_dir=dest,
                local_dir_use_symlinks=False,
                token=config.HF_token,
                resume_download=False,
                force_download=False
            )
            print(f"Model {model_name} downloaded")

if __name__ == '__main__':

    with open("./configs/default_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    config = resolve_vars(config)
    config = dict_to_namespace(config)

    download_all_models(config,MODELS)
