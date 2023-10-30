from .img_enhance_model import (
    UWImgEnhanceModel
)


def create_model(name, cfg):
    if name == 'uie':
        model = UWImgEnhanceModel(cfg)
    else:
        assert f"<{name}> not exist!"
    return model


__all__ = [
    create_model
]