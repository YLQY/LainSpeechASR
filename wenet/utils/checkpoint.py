# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import re

import yaml
import torch
from collections import OrderedDict

import datetime


def load_checkpoint(model: torch.nn.Module, path: str) -> dict:
    rank = int(os.environ.get('RANK', 0))
    logging.info('[Rank {}] Checkpoint: loading from checkpoint {}'.format(
        rank, path))
    checkpoint = torch.load(path, map_location='cpu', mmap=True)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint,
                                                          strict=False)
    if rank == 0:
        for key in missing_keys:
            logging.info("missing tensor: {}".format(key))
        for key in unexpected_keys:
            logging.info("unexpected tensor: {}".format(key))
    info_path = re.sub('.pt$', '.yaml', path)
    configs = {}
    if os.path.exists(info_path):
        with open(info_path, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
    return configs


def save_state_dict_and_infos(state_dict, path: str, infos=None):
    rank = int(os.environ.get('RANK', 0))
    logging.info('[Rank {}] Checkpoint: save to checkpoint {}'.format(
        rank, path))
    torch.save(state_dict, path)
    info_path = re.sub('.pt$', '.yaml', path)
    if infos is None:
        infos = {}
    infos['save_time'] = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    with open(info_path, 'w') as fout:
        data = yaml.dump(infos)
        fout.write(data)


def save_checkpoint(model: torch.nn.Module, path: str, infos=None):
    '''
    Args:
        infos (dict or None): any info you want to save.
    '''
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
        
    elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
        is_llm=False
        for name,params in state_dict.items():
            if "llm" in name:
                is_llm=True
                break
            pass
        if is_llm:
            # 只保存Adapter，encoder，lora
            state_dict = save_encoder_adapter_lora(state_dict)
    else:
        state_dict = model.state_dict()
        
    save_state_dict_and_infos(state_dict, path, infos)


def save_encoder_adapter_lora(state_dict):
    save_dict_encoder_adapter_lora = {}
    for name,params in state_dict.items():
        if "encoder" in name or "lora_" in name:
            save_dict_encoder_adapter_lora[name] = params
            pass
        pass
    return save_dict_encoder_adapter_lora

def filter_modules(model_state_dict, modules):
    rank = int(os.environ.get('RANK', 0))
    new_mods = []
    incorrect_mods = []
    mods_model = model_state_dict.keys()
    for mod in modules:
        if any(key.startswith(mod) for key in mods_model):
            new_mods += [mod]
        else:
            incorrect_mods += [mod]
    if incorrect_mods and rank == 0:
        logging.warning(
            "module(s) %s don't match or (partially match) "
            "available modules in model.",
            incorrect_mods,
        )
        logging.warning("for information, the existing modules in model are:")
        logging.warning("%s", mods_model)

    return new_mods


def load_trained_modules(model: torch.nn.Module, args: None):
    # Load encoder modules with pre-trained model(s).
    enc_model_path = args.enc_init
    enc_modules = args.enc_init_mods
    main_state_dict = model.state_dict()
    logging.warning("model(s) found for pre-initialization")
    if os.path.isfile(enc_model_path):
        logging.info('Checkpoint: loading from checkpoint %s for CPU' %
                     enc_model_path)
        model_state_dict = torch.load(enc_model_path, map_location='cpu')
        modules = filter_modules(model_state_dict, enc_modules)
        partial_state_dict = OrderedDict()
        for key, value in model_state_dict.items():
            if any(key.startswith(m) for m in modules):
                partial_state_dict[key] = value
        main_state_dict.update(partial_state_dict)
    else:
        logging.warning("model was not found : %s", enc_model_path)

    model.load_state_dict(main_state_dict)
    configs = {}
    return configs
