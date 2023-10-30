from typing import Dict, Any

from .ranet import RANet


def create_network(cfg: Dict[str, Any]):
    name = cfg['name']
    if name == 'ra':
        net = RANet(
            cfg['input_nc'], cfg['output_nc'],
            cfg['n_blocks'], cfg['n_down'],
            ngf = cfg['ngf'],
            padding_type = cfg['padding_type'],
            use_dropout = cfg['use_dropout'],
            use_att_down = cfg['use_att_down'],
            use_att_up = cfg['use_att_up']
        )
    else:
        assert f"<{name}> is not supported!"

    return net