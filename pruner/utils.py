import copy 
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

# Pruning operation
def pruning_model(model, px):

    print('Apply Unstructured L1 Pruning Globally (all conv layers)')
    parameters_to_prune =[]
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            parameters_to_prune.append((m,'weight'))

    parameters_to_prune = tuple(parameters_to_prune)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )

def pruning_model_structured(model, px):

    print('Apply Unstructured L1 Pruning Globally (all conv layers)')
    # parameters_to_prune =[]
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.ln_structured(
                m,
                name='weight',
                amount=px,
                dim=0, 
                n=1, # l1 loss
            )

def prune_model_custom(model, mask_dict):

    print('Pruning with custom mask (all conv layers)')
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            mask_name = name+'.weight_mask'
            if mask_name in mask_dict.keys():
                prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name+'.weight_mask'])
            else:
                print('Can not find [{}] in mask_dict'.format(mask_name))

def remove_prune(model):
    
    print('Remove hooks for multiplying masks (all conv layers)')
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.remove(m,'weight')

# Mask operation function
def extract_mask(model_dict):

    new_dict = {}
    for key in model_dict.keys():
        if 'mask' in key:
            new_dict[key] = copy.deepcopy(model_dict[key])

    return new_dict

# Mask statistic function
def check_sparsity(model):
    
    sum_list = 0
    zero_sum = 0

    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            sum_list = sum_list+float(m.weight.nelement())
            zero_sum = zero_sum+float(torch.sum(m.weight == 0))  

    if zero_sum:
        remain_weight_ratie = 100*(1-zero_sum/sum_list)
        print('* remain weight ratio = ', 100*(1-zero_sum/sum_list),'%')
    else:
        print('no weight for calculating sparsity')
        remain_weight_ratie = None

    return remain_weight_ratie

