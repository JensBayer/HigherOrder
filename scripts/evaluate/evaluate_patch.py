import time
import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as T
import kornia as K

from tqdm import tqdm, trange

import time
import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as T
import kornia as K

from tqdm import tqdm, trange

import json
import sys

from adv_utils.applier import PatchApplier
from adv_utils.dataset import HallucinatedPatchedDetectionDataset
from adv_utils.utils import padded_resize, coco_transform, Meter, evaluate
from adv_utils.loss import ObjectnessLoss, ValidityLoss, SmoothnessLoss
from adv_utils.extractor import Extractor

from ultralytics import YOLO

    
def evaluate_patch(args):
    device = args.device
    FILL_VALUE = args.fill_value
    MODEL = args.model
    size = args.size
    glob_pattern = args.glob

    if args.output and Path(args.output).exists():
        return
    
    if args.patch is not None:
        patch = Path(args.patch)
        assert patch.exists()
    else:
        patch = None

    def transforms(x, y):
        return coco_transform(
            x, 
            [l for l in y if l['area'] / float(x.width*x.height) >= args.bbox_perc_minsize],
            size=args.image_size,
            id_filter=[1])

    ds_test = torchvision.datasets.CocoDetection(
        args.dataset_root, 
        args.dataset_anno,
        transforms=transforms)
    
    ds_test = HallucinatedPatchedDetectionDataset(
        0.5,
        ds_test,
        PatchApplier(
            resize_range=(size, size),
            probability=0.5,
        ),
        lambda x : x[:,:-1]
    )
    
    
    def collate(data):
        images = torch.stack([img for img, _ in data])
        xyxy = [xyxy for _, xyxy in data]
        return images, xyxy
    
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=16, shuffle=False, pin_memory=True, num_workers=16, collate_fn=collate)
    
    model = YOLO(MODEL)
    model = Extractor(model, model.model.model[-1])
    model.enable_extract(False)

    imgsz = args.image_size
    _ = model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())), verbose=False)

    
    cudnn.benchmark = True
    if patch is not None and patch.is_dir():
        patches = sorted(list(patch.glob(glob_pattern)))
    else:
        patches = [patch]
    for p in patches:
        if p is not None:
            patch = torchvision.io.read_image(p)/255
            ds_test.applier.patch = patch.detach().clamp(0,1)
            mAP = evaluate(model, dl_test, img_size=args.image_size, box_format='xywh', normalized=True, target_class=1)
            mAP = {'model': str(args.model), 'name': str(p), **{k: v.item() for k, v in mAP.items()}}
        else:
            mAP = evaluate(model, dl_test, img_size=args.image_size, box_format='xywh', normalized=True, target_class=1)
            mAP = {'model': str(args.model), 'name': 'None', **{k: v.item() for k, v in mAP.items()}}

    if args.output:
        with Path(args.output).open('w') as fd:
            json.dump(mAP, fd)
    else:
        print(mAP)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='[Required] Model to be evaluated')
    parser.add_argument('--patch', type=str, required=False, help='Patch(es) to be evaluated.')
    parser.add_argument('--dataset_root', type=str, default='/data/coco2017/images/val2017')
    parser.add_argument('--dataset_anno', type=str, default='/data/coco2017/annotations/instances_val2017.json')
    parser.add_argument('--output', type=str, required=False, help='If set, results are written to this file.')
    parser.add_argument('--bbox_perc_minsize', type=float, default=0, help='Boxes smaller than this threshold are removed from the ground-truth.')
    
    parser.add_argument('--image_size', type=int, required=True, help='[Required] Images are resized to this size.')
    parser.add_argument('--size', type=float, default=0.75, help='[0.75] fixed resize range factor.')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--fill_value', type=float, default=-1, help='fill value used in the transformations.')
    parser.add_argument('--glob', type=str, default='*.png', help='[*.png] If patch is a directory, this is the glob used to retrieve a list of patches to be checked.')
    args = parser.parse_args()
    evaluate_patch(args)
