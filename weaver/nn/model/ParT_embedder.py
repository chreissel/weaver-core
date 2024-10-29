import torch
import torch.nn as nn
import torch.nn.functional as F
from weaver.nn.model.ParticleTransformer import ParticleTransformer
from weaver.utils.logger import _logger
import sys
sys.path.append("/n/home03/creissel/weaver-core/")
from weaver.nn.loss.SimCLRLoss import SimCLRWrapper

class ContraParT(nn.Module):
    def __init__(self,projector_dims,projector_out_dim,projector_act=nn.ReLU(),**kwargs):
        super().__init__()
        self.part = ParticleTransformer(**kwargs)
        self.embed_dim = kwargs['num_classes']
        fcs = []
        in_dim = self.embed_dim
        for dim in projector_dims:
            fcs.append(nn.Sequential(nn.Linear(in_dim,dim),projector_act))
            in_dim = dim
        fcs.append(nn.Linear(in_dim,projector_out_dim))
        self.projector = nn.Sequential(*fcs)

    def embed(self,x, v=None, mask=None, uu=None, uu_idx=None):
        x = self.part(x,v=v,mask=mask,uu=uu,uu_idx=uu_idx)
        x = F.normalize(x,dim=1)
        return x
    
    def project(self,x):
        x = self.projector(x)
        x = F.normalize(x,dim=1)
        return x
    
    def forward(self, points, features, lorentz_vectors, mask, project=True):
        x = self.embed(features,v=lorentz_vectors,mask=mask)
        if project:
            x = self.project(x)
        return x
    
def get_model(data_config,**kwargs):
    assert "out_dim" in kwargs.keys()

    cfg = dict(
        input_dim=len(data_config.input_dicts['pf_features']),
        num_classes=kwargs['out_dim'],
        embed_dims=[128,512,128],
        pair_embed_dims=[64,64,64],
        num_heads=8,
        num_layers=4,
        num_cls_layers=2,
        fc_params=[(128,0.0),(64,0.0)]
    )
    #cfg.update(**kwargs)
    _logger.info('Model config: %s' % str(cfg))

    projector_dims = [kwargs['out_dim'],8]
    projector_out_dim = 8

    model = ContraParT(projector_dims,projector_out_dim,**cfg)

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': [],
        'dynamic_axes': {},
        'model_type': 'ParT'
    }
    
    return model,model_info

def get_loss(data_config,**kwargs):
    return SimCLRWrapper()
