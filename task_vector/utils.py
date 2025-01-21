import torch
from argparse import Namespace
import os
import torch.nn as nn

from ttab.model_adaptation import get_model_adaptation_method
from ttab.model_selection import get_model_selection_method
from ttab.loads.define_model import define_model, load_pretrained_model
import ttab.loads.define_dataset as define_dataset
from ttab.scenarios import (
    CrossMixture,
    HeterogeneousNoMixture,
    HomogeneousNoMixture,
    InOutMixture,
    Scenario,
    TestCase,
    TestDomain,
)
from ttab.loads.datasets.dataset_shifts import (
    NaturalShiftProperty,
    NoShiftProperty,
    SyntheticShiftProperty,
    data2shift,
)
from ttab.loads.models import resnet

def softmax_entropy(x:torch.Tensor)->torch.Tensor:
    return -(x.softmax(1)*x.log_softmax(1)).sum(1)

def get_data(cifar10,shift_name,batch_size=64,seed=2022):
    base_data_name = 'cifar10' if cifar10 else 'cifar100'
    config = Namespace(
        model_name='',
        data_path='./datasets/',
        base_data_name=base_data_name,
        seed=seed,
        device='cuda:0',
        data_size=None
    )
    scenario = Scenario(
        base_data_name = base_data_name,
        model_adaptation_method='',
        model_name='',
        model_selection_method='',
        src_data_name= base_data_name,
        task='classification',
        test_case=TestCase(
            batch_size=batch_size,
            data_wise='batch_wise',
            episodic=None,
            inter_domain=HomogeneousNoMixture(has_mixture=False),
            intra_domain_shuffle=True,
            offline_pre_adapt=None
        ),
        test_domains=[
            TestDomain(
                base_data_name=base_data_name,
                data_name=f'{base_data_name}_c_deterministic-{shift_name}-5',
                domain_sampling_name='uniform',
                domain_sampling_ratio=1.0,
                domain_sampling_value=None,
                shift_property=SyntheticShiftProperty(
                    shift_degree=5,
                    shift_name=shift_name,
                    version='deterministic',
                ),
                shift_type='synthetic'
            )
        ]
    )
    test_data_cls = define_dataset.ConstructTestDataset(config)
    test_loader = test_data_cls.construct_test_loader(scenario=scenario)
    return test_loader.iterator(batch_size=batch_size)

def get_cifar10_26(ckpt_path=os.path.join(os.curdir,'pretrained_ckpts','classification','resnet26_with_head','cifar10','rn26_bn.pth')):
    model = resnet('cifar10',26).cuda()
    ckpt = torch.load(ckpt_path,map_location='cuda:0')
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model

def get_cifar10_26_gn(ckpt_path=os.path.join(os.curdir,'pretrained_ckpts','classification','resnet26_with_head','cifar10','rn26_gn.pth')):
    model = resnet('cifar10',26,group_norm_num_groups=8).cuda()
    ckpt = torch.load(ckpt_path,map_location='cuda:0')
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model

def get_cifar100_26(ckpt_path=os.path.join(os.curdir,'pretrained_ckpts','classification','resnet26_with_head','cifar100','rn26_bn.pth')):
    model = resnet('cifar100',26).cuda()
    ckpt = torch.load(ckpt_path,map_location='cuda:0')
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model


def load_weight(state_dict,model):
    for key,param in state_dict.items():
        module_name,attr_name = key.rsplit('.',1)
        module = model
        for sub_name in module_name.split('.'):
            module = getattr(module,sub_name)
        setattr(module,attr_name,param)

