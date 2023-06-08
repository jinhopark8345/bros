"""
BROS
Copyright 2022-present NAVER Corp.
Apache License v2.0
"""

from model.bros_bies import BROSBIESModel
# from model.bros_bio import BROSBIOModel
from model.bros_spade import BROSSPADEModel
from model.bros_spade_rel import BROSSPADERELModel


def get_model(cfg):
    if cfg.model.head == "bies":
        model = BROSBIESModel(cfg=cfg)
    elif cfg.model.head == "bio":
        # model = BROSBIOModel(cfg=cfg)
        from bros import BrosForTokenClassification
        from bros import BrosConfig
        config_path = "/Users/jinho/Projects/bros/saved_models/config.json"
        config = BrosConfig.from_pretrained(config_path)
        model = BrosForTokenClassification(config)

    elif cfg.model.head == "spade":
        model = BROSSPADEModel(cfg=cfg)
    elif cfg.model.head == "spade_rel":
        model = BROSSPADERELModel(cfg=cfg)
    else:
        raise ValueError(f"Unknown cfg.model.head={cfg.model.head}")

    return model
