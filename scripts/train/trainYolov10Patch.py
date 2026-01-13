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
from adv_utils.applier import PatchApplier
from adv_utils.dataset import PatchedDetectionDataset
from adv_utils.utils import padded_resize, coco_transform, Meter, evaluate
from adv_utils.loss import ObjectnessLoss, ValidityLoss, SmoothnessLoss
from adv_utils.extractor import Extractor

from ultralytics import YOLO

    
def gen_patch(args):
    device = args.device
    NPATCHES = args.npatches
    EPOCHS = args.epochs
    PATCH_INIT = args.patch_init
    PATCH_SHAPE = args.patch_shape
    LR = args.lr
    SCHEDULER_STEP_SIZE = args.scheduler_step_size
    RESIZE_RANGE = args.resize_range
    FILL_VALUE = args.fill_value
    SMT_WEIGHT = args.smt_weight
    VAL_WEIGHT = args.val_weight
    OUTPUT_PATH = args.output_path
    MODEL = args.model
    
    RUN_ID = int(time.time())
    
    
    ds = torchvision.datasets.CocoDetection(
        '/data/Inria/Train/pos', 
        '/data/Inria/inriaperson_correct.json', 
        transforms=coco_transform)

    ds_test = torchvision.datasets.CocoDetection(
        '/data/Inria/Test/pos', 
        '/data/Inria/inriaperson_test_correct.json', 
        transforms=coco_transform)
    
    ds_test = PatchedDetectionDataset(ds_test, PatchApplier(resize_range=(0.75,0.75)), lambda x : x[:,:-1])
    
    def collate(data):
        images = torch.stack([img for img, _ in data])
        xyxy = [xyxy for _, xyxy in data]
        return images, xyxy
    
    dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True, pin_memory=True, num_workers=16, collate_fn=collate)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=16, shuffle=False, pin_memory=True, num_workers=16, collate_fn=collate)
    
    model = YOLO(MODEL)
    model = Extractor(model, model.model.model[-1])
    imgsz = 640
    _ = model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())), verbose=False)
    
    
    applier = PatchApplier(
        None, 
        patch_transforms=T.Compose([
            T.ColorJitter(0.3, 0.1, 0.1, 0.1),
            T.RandomRotation(30, fill=FILL_VALUE),
            T.RandomPerspective(fill=FILL_VALUE),
        ]),
        mode=PatchApplier.MODE_RANDOM_BBOX,
        resize_range=RESIZE_RANGE
    )

    post_transforms = T.Compose([
        T.Lambda(lambda x: x.clip(0,1)),
        K.augmentation.RandomEqualize(),
        K.augmentation.RandomAutoContrast(p=0.5),
        ])


    obj = ObjectnessLoss('v10')
    smt = SmoothnessLoss()
    val = ValidityLoss()
    
    cudnn.benchmark = True
    def train(epoch=0, save_patches=True):
        meters = {
            key : Meter() for key in ['obj', 'smt', 'val']
        }
        model.enable_extract()
    
        for i, (imgs, labels) in enumerate(dl):
            optimizer.zero_grad()
            imgs = torch.stack([
                applier(img, label[:,:-1], patch=patch)
                for img, label in zip(imgs, labels)
            ])
            
            imgs = post_transforms(imgs)
            imgs = imgs.clamp(0,1)

            with torch.amp.autocast('cuda'):
                y, (_, outs) = model(imgs.to(device), verbose=False)
                outs = {
                        'one2one': model.model.model.model[-1]._inference(outs[1]['one2one']),
                        'one2many': model.model.model.model[-1]._inference(outs[1]['one2many']),
                }
                obj_ = obj(outs)
                smt_ = smt(patch)
                val_ = val(patch)
                loss = obj_ + SMT_WEIGHT * smt_ + VAL_WEIGHT * val_
            loss.backward()
            optimizer.step()

            del y, outs # required to prevent memory leak :?
    
        if save_patches:
            T.ToPILImage()(imgs[0].detach().cpu()).save(f'{Path(OUTPUT_PATH)}/latest_example.png')
            T.ToPILImage()(patch.clamp(0,1).detach().cpu()).save(f'{Path(OUTPUT_PATH)}/latest_patch.png')
    
        return {'obj': meters['obj'](obj_.item()), 'smt': meters['smt'](smt_.item()), 'val': meters['val'](val_.item())}

    
    path = Path(OUTPUT_PATH) / f"{RUN_ID}"
    path.mkdir(parents=True, exist_ok=True)
    
    for n in range(NPATCHES):
        if PATCH_INIT == 'rand':
            patch = torch.rand(PATCH_SHAPE)
        else:
            patch = torch.full(PATCH_SHAPE, PATCH_INIT)
        
        patch = patch.requires_grad_()
    
        optimizer = torch.optim.AdamW([patch], LR)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, SCHEDULER_STEP_SIZE)
        applier.patch = patch

        tloader = tqdm(range(0, EPOCHS))
        for epoch in tloader:
            train_loss_dict = train(epoch)

            model.enable_extract(False)
            ds_test.applier.patch = patch.detach().clamp(0,1)
            mAP = evaluate(model, dl_test)['map'].item()
            model.enable_extract()

            tloader.set_postfix({'mAP': mAP, **train_loss_dict})
            T.ToPILImage()(patch.clamp(0,1).detach().cpu()).save(path / f'patch_{n:03d}.png')
            torch.save(patch.detach().cpu(), path / f'patch_{n:03d}.pt')
            scheduler.step()

        params = {
            'run_id': RUN_ID,
            'epochs': EPOCHS,
            'patch/shape': str(PATCH_SHAPE),
            'patch/initial': PATCH_INIT,
            'batch_size': dl.batch_size,
            'optimizer/type': type(optimizer).__name__,
            'optimizer/lr': optimizer.defaults['lr'],
            'scheduler/type': type(scheduler).__name__,
            'scheduler/step_size': scheduler.state_dict().get('step_size', -1),
            'scheduler/T_max': scheduler.state_dict().get('T_max', -1),
            'applier/resize_range/min': applier.resize_range[0],
            'applier/resize_range/max': applier.resize_range[0],
            'applier/mode': applier.mode,
            'applier/transforms': str(applier.patch_transforms),
            'model': MODEL,
            'n_patches': NPATCHES,
        }
        with (path / "params.json").open('w') as fd:
            json.dump(params, fd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--npatches', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--patch_init', type=str, default='rand')
    parser.add_argument('--patch-shape', type=tuple, default=(3, 256, 256))
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--scheduler_step_size', type=int, default=50)
    parser.add_argument('--resize_range', type=tuple, default=(0.3, 0.6))
    parser.add_argument('--fill_value', type=float, default=-1)
    parser.add_argument('--smt_weight', type=float, default=2)
    parser.add_argument('--val_weight', type=float, default=1)
    parser.add_argument('--output_path', type=str, default='outputs')
    args = parser.parse_args()
    gen_patch(args)

    
