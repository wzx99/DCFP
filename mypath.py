# resnet50 = 'pretrained_models/resnet50-imagenet.pth'
resnet50 = 'pretrained_models/resnet50-25c4b509.pth'
# resnet101 = 'pretrained_models/resnet101-imagenet.pth'
resnet101 = 'pretrained_models/resnet101-2a57e44d.pth'
resnet152 = 'pretrained_models/resnet152-0d43d698.pth'

CS_root_dir = 'data/CS'
CS_train_data_list_dir = './datasets/list/cityscapes/train.lst'
CS_trainval_data_list_dir = './datasets/list/cityscapes/trainval.lst'
CS_val_data_list_dir = './datasets/list/cityscapes/val.lst'
CS_test_data_list_dir = './datasets/list/cityscapes/test.lst'

CTX_root_dir = 'data/CTX'  
CTX_train_data_list_dir = './datasets/list/ctx/train.txt'
CTX_val_data_list_dir = './datasets/list/ctx/val.txt'

ADE_root_dir = 'data'
ADE_train_data_list_dir = './datasets/list/ade/training.odgt'
ADE_val_data_list_dir = './datasets/list/ade/validation.odgt'

COCO_root_dir = 'data/cocostuff'
COCO_train_data_list_dir = './datasets/list/cocostuff/train.txt'
COCO_val_data_list_dir = './datasets/list/cocostuff/test.txt'

def get_dataset(dataset):
    if dataset.startswith('CS'):
        return 'CS'
    elif dataset.startswith('CTX'):
        return 'CTX'
    elif dataset.startswith('ADE'):
        return 'ADE'
    elif dataset.startswith('COCO'):
        return 'COCO'

class Path(object):
    @staticmethod
    def data_dir(dataset,split):
        dataset = get_dataset(dataset)
        return eval(dataset+'_root_dir'), eval(dataset+'_'+split+'_data_list_dir')
    
    @staticmethod
    def pretrained_dir(model):
        return eval(model)