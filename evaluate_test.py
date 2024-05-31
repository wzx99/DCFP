import sys
sys.path.insert(1,'/data/pylib/')

import argparse
import numpy as np
import sys
import json
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import networks
from datasets import build_dataset
import os
from math import ceil
from PIL import Image as PILImage

from utils.pyt_utils import load_model
from engine import Engine
from evaluate import predict_multiscale, generate_size_image, pad_inf
import pruners


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_parser():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DCFP")
    parser.add_argument("--dataset", type=str, default='CS',
                        help="choose dataset.")
    parser.add_argument("--ignore-label", type=int, default=255,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--restore-from", type=str, default='xx.pth',
                        help="Where restore model parameters from.")
    parser.add_argument("--input-size", type=str, default='769,769',
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--longsize", type=int, default=-1)
    parser.add_argument("--shortsize", type=int, default=-1)
    parser.add_argument("--num-workers", type=int, default=8,
                        help="choose the number of recurrence.")
    parser.add_argument("--ddp", type=str2bool, default='True')
    parser.add_argument("--align-corner", type=str2bool, default='True',
                        help="choose align corner.")
    parser.add_argument("--whole", type=str2bool, default='False',
                        help="use whole input size.")
    parser.add_argument("--flip", type=str2bool, default='False',
                        help="flip test.")
    parser.add_argument("--ms", type=str, default='1',
                        help="multi scale")
    parser.add_argument("--model", type=str, default='None',
                        help="choose model.")
    parser.add_argument("--backbone", type=str, default='renet101',
                        help="backbone")
    parser.add_argument("--backbone-para", type=str, default='{}')
    parser.add_argument("--model-para", type=str, default='{}')
    parser.add_argument("--channel-cfg", type=str, default=None, help="path to channel_cfg.")
    return parser


def main():
    """Create the model and start the evaluation process."""
    parser = get_parser()

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()

        cudnn.benchmark = True

        h, w= map(int, args.input_size.split(','))
        input_size = (h,w)
         
        args.ms = [float(s) for s in args.ms.split(',')]
        # args.ms = [0.5,0.75,1.0,1.25,1.5,1.75,2.0]
        
        if (not engine.distributed) or (engine.distributed and engine.local_rank == 0):
            print("Running with config:")
            for k,v in vars(args).items():
                print('{}: {}'.format(k,v))

        dataset = build_dataset(args.dataset, split='test', data_dir='test')
        test_loader, test_sampler = engine.get_test_loader(dataset)
        if engine.distributed:
            test_sampler.set_epoch(0)
            
        backbone_para = json.loads(args.backbone_para)
        model_para = json.loads(args.model_para)
        seg_model = eval('networks.'+args.model+'.Seg_Model')(
                        backbone=args.backbone,
                        backbone_para=backbone_para,
                        model_para=model_para,
                        num_classes=dataset.num_classes,
                        align_corner=args.align_corner)
        
        if args.channel_cfg is not None:
            channel_cfg = torch.load(args.channel_cfg)
            pruners.init_pruned_model(seg_model, channel_cfg)
        load_model(seg_model, args.restore_from)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seg_model.to(device)

        model = engine.data_parallel(seg_model)
        model.eval()

        # palette = get_palette(256)
        palette = list(dataset.cmap_labels.reshape(-1))

        save_path = os.path.join(os.path.dirname(args.restore_from), 'outputs')
        pred_id_path = os.path.join(save_path, 'test_id')
        pred_path = os.path.join(save_path, 'test_pred')
        for p in [save_path,pred_id_path,pred_path]:
            if not os.path.exists(p):
                if (not engine.distributed) or (engine.distributed and engine.local_rank == 0):
                    os.makedirs(p)

        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(len(test_loader)), file=sys.stdout,
                    bar_format=bar_format)
        dataloader = iter(test_loader)
        for idx in pbar:
            with torch.no_grad():
                data = dataloader.next()
                image, img_meta = data["img"], data["img_meta"]
                if args.longsize > 0:
                    image = generate_size_image(image, args.longsize, 'long')
                elif args.shortsize > 0:
                    image = generate_size_image(image, args.shortsize, 'short')
                size_scale = image.shape[2:]
                if args.whole and args.align_corner:
                    image = pad_inf(image) 
                output = predict_multiscale(model, image, input_size, args.ms, dataset.num_classes, args.flip, args.align_corner, args.whole)
                output = output[:,:,:size_scale[0],:size_scale[1]]
                if args.longsize > 0 or args.shortsize > 0:
                    output = F.interpolate(output, size=(img_meta[0]["size"][0], img_meta[0]["size"][1]), mode='bilinear', align_corners=False)
            output = output.numpy().transpose(0,2,3,1)
            seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)

            for i in range(image.size(0)):
                #save id
                output_id = PILImage.fromarray(dataset.id2trainId(seg_pred[i], reverse=True))
                if output_id.mode != 'L':
                    output_id = output_id.convert('L')
                output_id.save(os.path.join(pred_id_path, img_meta[i]["name"].split('_leftImg8bit')[0] + '.png'))

                #save img
                output_im = PILImage.fromarray(seg_pred[i])
                output_im.putpalette(palette)
                output_im.save(os.path.join(pred_path, img_meta[i]["name"]+'.png'))

            print_str = ' Iter{}/{}'.format(idx + 1, len(test_loader))
            pbar.set_description(print_str, refresh=False)

        print('end')

if __name__ == '__main__':
    main()
