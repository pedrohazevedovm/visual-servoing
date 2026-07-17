
from models.decoder_model import ENCODER_DECODER
from models.exictationhed_model import EXCITHED
from models.octave_model import OCTHED
from models.hed_model import HED
from models.fourier_model import FFCHED
from models.side_outputs_dense_model import DENSEHED
from models.decoder_model_octave import OCTENCODER_DECODER
from models.octave_model_full import OCTHEDFULL
from models.octave_model_full_side_extra_layer import OCTHEDFULL_SIDE_EXTRA
from models.excitationoctave_model import EXCITOCTHED

import torch.nn as nn

def get_model(args, device = 'cpu'):
    alpha = args.alpha
    target_layers = args.octave_layers
    model_name = args.model

    print(target_layers)
    MODELS = {
        'HED': lambda: HED(device),
        'OCTHED': lambda: OCTHED(device, alpha=float(alpha), octave_layers=target_layers),
        'OCTHEDFULL': lambda: OCTHEDFULL(device, alpha = float(alpha), octave_layers=target_layers),
        'OCTHEDFULL_SIDE_EXTRA' : lambda: OCTHEDFULL_SIDE_EXTRA(device, alpha = float(alpha), octave_layers = target_layers),
        'EXCITHED': lambda: EXCITHED(device),
        'EXCITOCTHED': lambda: EXCITOCTHED(device, alpha = float(alpha), octave_layers=target_layers),
        'ENCODER_DECODER': lambda: ENCODER_DECODER(device),
        'OCTENCODER_DECODER' : lambda: OCTENCODER_DECODER(device, float(alpha)),
        'FFCHED' : lambda: FFCHED(device, ratio = float(alpha), fourier_layer=target_layers),
        'DENSEHED' : lambda: DENSEHED(device),
    }
    
    if model_name not in MODELS:
        raise ValueError(f"Can't recognize {model_name}.")
    print((model_name))
    return nn.DataParallel(MODELS[model_name]())