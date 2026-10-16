#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from celeba_experiment.data import extract, set_backbone_checkpoint
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/rate_selection_robust.json')
    p.add_argument('--cache',default='cache/celeba_paper')
    p.add_argument('--backbone-checkpoint',type=Path,help='Multitask backbone.pt')
    args=p.parse_args()
    config=json.loads(Path(args.config).read_text())
    set_backbone_checkpoint(config,args.backbone_checkpoint)
    print(json.dumps(extract(config,args.cache),indent=2))
