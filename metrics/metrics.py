from utils.class_registry import ClassRegistry
import os
import torch


metrics_registry = ClassRegistry()


@metrics_registry.add_to_registry(name="fid")
class FID:
    def __call__(self, orig_path, synt_path):
        if torch.cuda.is_available():
            fid = os.popen(f'python3 -m pytorch_fid {orig_path} {synt_path} --device cuda:0').read()
        else:
            fid = os.popen(f'python3 -m pytorch_fid {orig_path} {synt_path}').read()
        print(fid)
        return fid
    
    def get_name(self):
        return 'FID'
