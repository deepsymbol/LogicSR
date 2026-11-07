# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
from copy import copy
import os
import torch
from .embedders import LinearPointEmbedder
from .transformer import TransformerModel
from .model_wrapper import Boolformer
import torch.nn as nn
import gdown

logger = getLogger()

def load_boolformer(mode="noiseless"):
    # 首先尝试从external/boolformer文件夹加载本地模型
    current_dir = os.path.dirname(os.path.abspath(__file__))
    external_model_path = os.path.join(current_dir, "..", "..", f"boolformer_{mode}.pt")
    
    # 检查external文件夹中是否有模型文件
    if os.path.exists(external_model_path):
        print(f"Loading {mode} model from external folder: {external_model_path}")
        boolformer_model = torch.load(external_model_path, map_location='cpu', weights_only=False)
        print(f"Successfully loaded {mode} model from local file")
        return boolformer_model
    
    # 如果external文件夹没有，则尝试当前目录
    model_path = f"boolformer_{mode}.pt" 
    if os.path.exists(model_path):
        print(f"Loading {mode} model from current directory: {model_path}")
        boolformer_model = torch.load(model_path, map_location='cpu', weights_only=False)
        print(f"Successfully loaded {mode} model from current directory")
        return boolformer_model
    
    # 最后尝试下载（如果本地都没有）
    print(f"Model not found locally, attempting to download...")
    try:
        if mode=="noiseless":
            url = "https://drive.google.com/uc?id=1cULlE16yKBqUZMMGv7CC5fgHXXJ7OQQQ"
        elif mode=="noisy":
            url = "https://drive.google.com/uc?id=1IFSc_sHfiTckMy-cwggvVMBH1YgXas8G"
        
        gdown.download(url, model_path, quiet=False)
        boolformer_model = torch.load(model_path, map_location='cpu', weights_only=False)
        print(f"Successfully downloaded and loaded {mode} model")
        return boolformer_model
    except Exception as e:
        print(f"Failed to download model: {e}")
        print("Please manually download the model and place it in external/boolformer/ folder")
        print(f"Model should be named: boolformer_{mode}.pt")
        if mode == "noiseless":
            print("Download URL: https://drive.google.com/uc?id=1cULlE16yKBqUZMMGv7CC5fgHXXJ7OQQQ")
        elif mode == "noisy":
            print("Download URL: https://drive.google.com/uc?id=1IFSc_sHfiTckMy-cwggvVMBH1YgXas8G")
        raise ImportError(f"Could not load boolformer_{mode}.pt model")

def check_model_params(params):
    """
    Check models parameters.
    """
    if params.dec_emb_dim is None:
        params.dec_emb_dim = params.enc_emb_dim
    if params.n_dec_layers is None:
        params.n_dec_layers = params.n_enc_layers
    if params.n_dec_heads is None:
        params.n_dec_heads = params.n_enc_heads
    # model dimensions
    assert params.enc_emb_dim % params.n_enc_heads == 0
    assert params.dec_emb_dim % params.n_dec_heads == 0

    # reload a pretrained model
    if params.reload_model != "":
        print("Reloading model from ", params.reload_model)
        assert os.path.isfile(params.reload_model)


def build_modules(env, params):
    """
    Build modules.
    """
    modules = {}
    modules["embedder"] = LinearPointEmbedder(params, env)

    modules["encoder"] = TransformerModel(
        params,
        env.input_id2word,
        is_encoder=True,
        with_output=False,
        use_prior_embeddings=True,
        positional_embeddings=params.enc_positional_embeddings,
    )

    modules["decoder"] = TransformerModel(
        params,
        env.output_id2word,
        is_encoder=False,
        with_output=True,
        use_prior_embeddings=False,
        positional_embeddings=params.dec_positional_embeddings,
    )

    # reload pretrained modules
    if params.reload_model != "":
        logger.info(f"Reloading modules from {params.reload_model} ...")
        reloaded = torch.load(params.reload_model)
        for k, v in modules.items():
            assert k in reloaded
            if all([k2.startswith("module.") for k2 in reloaded[k].keys()]):
                reloaded[k] = {
                    k2[len("module.") :]: v2 for k2, v2 in reloaded[k].items()
                }
            v.load_state_dict(reloaded[k])

    # log
    for k, v in modules.items():
        logger.debug(f"{v}: {v}")
    for k, v in modules.items():
        logger.info(
            f"Number of parameters ({k}): {sum([p.numel() for p in v.parameters() if p.requires_grad])}"
        )

    # cuda
    if not params.cpu:
        for v in modules.values():
            v.cuda()

    return modules
