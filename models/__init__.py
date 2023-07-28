from .resnet import Resnet18,Resnet34,Resnet50,Resnet101
from .googlenet import googlenet
from .bn_inception import bn_inception

# Create the model
def load(cfg, pretrained):
    model_embedding_size = int(cfg.embedding_size)
    
    if cfg.backbone.find('googlenet')+1:
        model = googlenet(embedding_size=model_embedding_size, pretrained=pretrained, is_norm=cfg.l2_norm, bn_freeze = cfg.bn_freeze)
    elif cfg.backbone.find('bn_inception')+1:
        model = bn_inception(embedding_size=model_embedding_size, pretrained=pretrained, is_norm=cfg.l2_norm, bn_freeze = cfg.bn_freeze)
    elif cfg.backbone.find('resnet18')+1:
        model = Resnet18(embedding_size=model_embedding_size, pretrained=pretrained, is_norm=cfg.l2_norm, bn_freeze = cfg.bn_freeze)
    elif cfg.backbone.find('resnet34')+1:
        model = Resnet34(embedding_size=model_embedding_size, pretrained=pretrained, is_norm=cfg.l2_norm, bn_freeze = cfg.bn_freeze)
    elif cfg.backbone.find('resnet50')+1:
        model = Resnet50(embedding_size=model_embedding_size, pretrained=pretrained, is_norm=cfg.l2_norm, bn_freeze = cfg.bn_freeze)
    elif cfg.backbone.find('resnet101')+1:
        model = Resnet101(embedding_size=model_embedding_size, pretrained=pretrained, is_norm=cfg.l2_norm, bn_freeze = cfg.bn_freeze)

    else:
        raise ValueError('Unsupported model depth,\
                         must be one of resnet18, resnet34, resnet50, resnet101,\
                         googlenet, bn_inception')
    
    return model