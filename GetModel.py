from models import phc_models, real_models

def GetModel(str_model, n, num_classes=1, weights=None, shared=False, patch_weights=True, visualize=False):
    """
    Get model from str_model.

    Parameters:
    - str_model can be: resnet18, phcresnet18, resnet50, phcresnet50, sbonet, physbonet, senet, physenet.
    - weights: path tho weights. Needed for physbonet and physenet.
    - shared: parameter of physbonet.
    - patch_weghts: parameter of physenet.
    """

    print('Model:', str_model)
    print()
    
    ## Two-view models ##

    if str_model == 'resnet18':
        return real_models.ResNet18(num_classes, channels=n, visualize=visualize)
    elif str_model == 'phcresnet18':
        return phc_models.PHCResNet18(channels=2, n=n, num_classes=num_classes, visualize=visualize)
        
    if str_model == 'resnet50':
        return real_models.ResNet50(num_classes, channels=n)
    elif str_model == 'phcresnet50':
        return phc_models.PHCResNet50(channels=2, n=n, num_classes=num_classes)

    ## Four-view models ##
    
    if str_model == 'sbonet':
        return real_models.SEnet(shared=shared, num_classes=num_classes, weights=weights)
    elif str_model == 'physbonet':
        return phc_models.PHYSBOnet(n=n, shared=shared, num_classes=num_classes, weights=weights)

    if str_model == 'senet':
        return real_models.SEnet(num_classes=num_classes, weights=weights, patch_weights=patch_weights, visualize=visualize)
    elif str_model == 'physenet':
        return phc_models.PHYSEnet(n=n, num_classes=num_classes, weights=weights, patch_weights=patch_weights, visualize=visualize)

    else:
        raise ValueError ('Model not implemented, check allowed models (-help) \n \
             Check the model you typed.')
        
        
