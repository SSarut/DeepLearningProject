import torch
import os

weight_path = "/home/sarut/Projects/DeepLearningProject/webapp/app/weights/resnet34_v1_best.pth"
if os.path.exists(weight_path):
    checkpoint = torch.load(weight_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        checkpoint = checkpoint['model_state_dict']
    
    print("Top 10 keys in weight file:")
    keys = list(checkpoint.keys())
    for k in keys[:10]:
        print(f"  {k}")
    
    print("\nKeys containing 'fc':")
    for k in keys:
        if 'fc' in k:
            print(f"  {k}")
else:
    print(f"Weight path not found: {weight_path}")
