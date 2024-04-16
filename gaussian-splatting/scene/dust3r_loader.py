import os 
import torch
from utils.graphics_utils import focal2fov 


def prepare_dust3r_extrinsics(path): 
    camera_extrinsics = [] 
    
    poses = torch.load(os.path.join(path, 'poses.pt')) 
    imgs = torch.load(os.path.join(path, 'imgs.pt'))  
    depths = torch.load(os.path.join(path, 'depths.pt'))   
    masks = torch.load(os.path.join(path, 'confidence_masks.pt'))    

    for i in range(len(poses)): 
        cam_info = {} 
        cam_info['rot'] = poses[i].detach()[:3,:3].cpu().numpy() 
        cam_info['tvec'] = poses[i].detach()[:3, 3].cpu().numpy()  
        cam_info['image'] = imgs[i]  
        cam_info['depths'] = depths[i].detach().cpu().numpy() 
        cam_info['mask'] = masks[i].cpu().numpy()   
        cam_info['height'], cam_info['width'] = imgs[i].shape[:2] 
        camera_extrinsics.append(cam_info) 
    
    return camera_extrinsics

def prepare_dust3r_intrinsics(path): 
    camera_intrinsics = [] 

    focals = torch.load(os.path.join(path, 'focals.pt'))  
    intrinsics = torch.load(os.path.join(path, 'intrinsics.pt'))   

    for i in range(len(focals)): 
        cam_info = {} 
        cam_info['focal'] = focals[i].item()   
        cam_info['K'] = intrinsics[i].detach().cpu().numpy()  
        camera_intrinsics.append(cam_info)

    return camera_intrinsics
        



