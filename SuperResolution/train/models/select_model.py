"""
# --------------------------------------------
# define training model
# --------------------------------------------
"""


def define_Model(opt):
    model = opt['model']  # one input: L

    if model == 'plain':
        from SuperResolution.train.models.model_plain import ModelPlain as M

    elif model == 'plain2':  # two inputs: L, C
        from SuperResolution.train.models.model_plain2 import ModelPlain2 as M

    elif model == 'gan':  # one input: L
        from SuperResolution.train.models.model_gan import ModelGAN as M

    else:
        raise NotImplementedError('Model [{:s}] is not defined.'.format(model))

    m = M(opt)

    print('Training model [{:s}] is created.'.format(m.__class__.__name__))
    return m
