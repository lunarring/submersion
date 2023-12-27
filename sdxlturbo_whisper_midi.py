#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, StableDiffusionXLControlNetPipeline
from custom_pipe_controlnet import call_custom
import torch
import time
from diffusers import AutoencoderTiny
import xformers
import triton
import lunar_tools as lt
from PIL import Image
import numpy as np
from diffusers.utils.torch_utils import randn_tensor
import numpy as np
import cv2
from diffusers import ControlNetModel
from prompt_blender import PromptBlender
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from img_utils import pad_image_to_width, pad_image_to_width, blend_images, process_cam_img, stitch_images, weighted_average_images
from datetime import datetime
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

#%% PARAMETERS

ctrlnet_type = "diffusers/controlnet-canny-sdxl-1.0-mid"
# ctrlnet_type = "diffusers/controlnet-depth-sdxl-1.0-mid"
use_ctrlnet = True
compile_with_sfast = False
shape_cam=(600,800)
size_diff_img = (512, 512) 
num_inference_steps = 2
# %% INITS
cam = lt.WebCam(cam_id=0, shape_hw=shape_cam)
cam.cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)

depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")

# initialize the models and pipeline
if use_ctrlnet:
    controlnet = ControlNetModel.from_pretrained(
        ctrlnet_type,
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
    ).to("cuda")
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained("stabilityai/sdxl-turbo", controlnet=controlnet, torch_dtype=torch.float16, variant="fp16")
else:
    pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    
pipe = pipe.to("cuda")

pipe.vae = AutoencoderTiny.from_pretrained('madebyollin/taesdxl', torch_device='cuda', torch_dtype=torch.float16)
pipe.vae = pipe.vae.cuda()
pipe.set_progress_bar_config(disable=True)

if compile_with_sfast:
    from sfast.compilers.diffusion_pipeline_compiler import (compile, CompilationConfig)
    pipe.enable_xformers_memory_efficient_attention()
    config = CompilationConfig.Default()
    config.enable_xformers = True
    config.enable_triton = True
    config.enable_cuda_graph = True
    pipe = compile(pipe, config)

#%%
def filter_edges(ctrl_img, min_size=10):
    arr = np.asarray(ctrl_img)[:,:,0]
    # connectivity=9
    # output = cv2.connectedComponentsWithStats(arr, connectivity, cv2.CV_32S)
        # Define your minimum size
      # Change this to your desired minimum size
    
    arr_binary = arr.astype(np.uint8)
    
    # Apply connected components with stats
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(arr_binary, 8, cv2.CV_32S)

    # Create an output array, initialized to zero (background)
    output = np.zeros_like(arr, dtype=np.uint8)
    
    # Loop through the found components
    for i in range(1, num_labels):  # Start from 1 to ignore the background
        size = stats[i, cv2.CC_STAT_AREA]
        if size >= min_size:
            # Add this component to the output image
            output[labels == i] = 255
            
    return Image.fromarray(output)

def get_ctrl_img(cam_img, ctrlnet_type, low_threshold=100, high_threshold=200):
    
    
    if "canny" in ctrlnet_type:
        cam_img = np.array(cam_img)
        ctrl_image = cv2.Canny(cam_img, low_threshold, high_threshold)
        ctrl_image = ctrl_image[:, :, None]
        ctrl_image = np.concatenate([ctrl_image, ctrl_image, ctrl_image], axis=2)
        ctrl_image = Image.fromarray(ctrl_image)
    else:
        depth_map = get_depth_map(cam_img)

        
        image = torch.cat([depth_map] * 3, dim=1)
    
        image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
        image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
        ctrl_image = image
        
    return ctrl_image    

def get_depth_map(cam_img, return_type="pt"):
    cam_img = np.array(cam_img)
    image = feature_extractor(images=cam_img, return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad(), torch.autocast("cuda"):
        depth_map = depth_estimator(image).predicted_depth
        
    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=size_diff_img,
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    
    if return_type == "np":
        depth_map = depth_map.cpu().numpy()[0][0]
        depth_map = depth_map.clip(0, 1)
    
    return depth_map


#%%
# Example usage
blender = PromptBlender(pipe)
generator = torch.Generator(device="cuda")

# base = 'skeleton person head skull terrifying'
# prompt = 'very bizarre and grotesque zombie monster'
# prompt = 'very bizarre alien with spaceship background'
# prompt = 'a funny weird frog'
# prompt = 'metal steampunk gardener'
# prompt = 'a strange city'
# prompt = 'a strange football match'
# prompt = 'a very violent boxing match'
# prompt = 'ballet dancing'
# prompt = 'a beautiful persian belly dancers'
# prompt = 'a medieval scene with people dressed in strange clothes'
# prompt = 'a bird with two hands'
# base = 'a beautiful redhaired mermaid'
# base = 'terror pig party'
# base = 'dirty and slimy bug monster'
prompt = 'a telepathic cyborg steampunk'
# base = 'a man wearing very expensive and elegant attire'
# base = 'a very beautiful young queen in france in versaille'
# base = 'a gangster party'

negative_prompt = 'blurry, tiled, wrong, bad art, pixels, amateur drawing, haze'
# sz = (990, 1980)
sz = (int(size_diff_img[0]*1.8), int(size_diff_img[1]*1.8))
width_renderer = width=2*sz[1]
    
image_diffusion = Image.fromarray(np.zeros((size_diff_img[1], size_diff_img[0], 3), dtype=np.uint8))
ctrl_img = Image.fromarray(np.zeros((size_diff_img[1], size_diff_img[0], 3), dtype=np.uint8))
latents1 = torch.randn((1,4,64//1,64)).half().cuda()
latents2 = torch.randn((1,4,64//1,64)).half().cuda()
prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blender.get_prompt_embeds(prompt, negative_prompt)

renderer = lt.Renderer(width=width_renderer, height=sz[0])
speech_detector = lt.Speech2Text()
akai_lpd8 = lt.MidiInput("akai_lpd8")

fps = 5

currently_recording_vid = False

# pipe hack
pipe.last_down_block_res_samples = [None]*num_inference_steps
pipe.last_mid_block_res_sample = [None]*num_inference_steps

while True:
    torch.manual_seed(420)
    
    controlnet_conditioning_scale = akai_lpd8.get("E0", val_min=0, val_max=1, val_default=0.5)
    
    do_record_mic = akai_lpd8.get("A0", button_mode="is_pressed")
    if do_record_mic:
        if not speech_detector.audio_recorder.is_recording:
            speech_detector.start_recording()
    elif not do_record_mic:
        if speech_detector.audio_recorder.is_recording:
            try:
                prompt = speech_detector.stop_recording()
            except Exception as e:
                print(f"FAIL {e}")
            print(f"New prompt: {prompt}")
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blender.get_prompt_embeds(prompt, negative_prompt)
            stop_recording = False
            
    do_record_vid = akai_lpd8.get("A1", button_mode="toggle")
    if do_record_vid and not currently_recording_vid:
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%y%m%d_%H%M")
        fp_out = f"{formatted_datetime}.mp4"
        movie = lt.MovieSaver(fp_out, fps=fps)
        currently_recording_vid = True
    
    elif currently_recording_vid and not do_record_vid:
        movie.finalize()
        currently_recording_vid = False
    
    fract_mixing = 0
    latents = blender.interpolate_spherical(latents1, latents2, fract_mixing)        

    cam_img = process_cam_img(cam.get_img())
    cam_img_orig = cam_img.copy()
    
    # Smooth Cam image?
    do_smoothing = akai_lpd8.get("C1", button_mode="toggle")
    if do_smoothing:
        fwhm = 1+2*int(akai_lpd8.get("G1", val_min=1, val_max=30, val_default=3))
        cam_img = Image.fromarray(cv2.GaussianBlur(np.asarray(cam_img), (fwhm, fwhm), 0))
    
    do_depth_filter = akai_lpd8.get("C0", button_mode="toggle")
    if do_depth_filter:
        depth_map = get_depth_map(cam_img, return_type='np')
        threshold_depth = akai_lpd8.get("G0", val_default=0.33)
        cam_img_np = np.array(cam_img)
        cam_img_np[depth_map < threshold_depth] = 0
        cam_img = Image.fromarray(cam_img_np)

    low_threshold = akai_lpd8.get("F0", val_default=0, val_min=0, val_max=255)
    high_threshold = akai_lpd8.get("F1", val_default=70, val_min=0, val_max=255)
    
    guidance_scale = 0.0
    t0 = time.time()
    if use_ctrlnet:
        averaging_weight = 1- akai_lpd8.get("E1", val_default=0)
        ctrl_img_prev =  ctrl_img.copy()   
        ctrl_img = get_ctrl_img(cam_img, ctrlnet_type, low_threshold=low_threshold, high_threshold=high_threshold)
        pipe, image_diffusion = call_custom(pipe, averaging_weight=averaging_weight, image=ctrl_img, latents=latents, controlnet_conditioning_scale=controlnet_conditioning_scale, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds)
        image_diffusion = image_diffusion.images[0]
        # image_diffusion = pipe(image=ctrl_img, latents=latents, controlnet_conditioning_scale=controlnet_conditioning_scale, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds).images[0]
    else:
        image_diffusion = pipe(image=cam_img, latents=latents, num_inference_steps=num_inference_steps, strength=0.9999, guidance_scale=guidance_scale, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds, generator=generator).images[0]
    dt = time.time() - t0
    fps = 1/dt
    print(f"\rCurrent fps: {fps:.1f}", end="")
    
    stitch_cam = akai_lpd8.get("D0", button_mode="toggle")
    show_ctrlnet_img = akai_lpd8.get("D1", button_mode="toggle")
    
    if stitch_cam:
        if show_ctrlnet_img and use_ctrlnet:
            image_show = stitch_images(ctrl_img, image_diffusion)
        else:
            image_show = stitch_images(cam_img_orig, image_diffusion)
            
    else:
        image_show = pad_image_to_width(image_diffusion, image_diffusion.size[0]*2)
        
    renderer.render(image_show)
    
    if do_record_vid and currently_recording_vid:
        movie.write_frame(np.asarray(image_show))
    
    # if np.random.rand() > 0.95:
    #     xxx

"""
- append prompt

"""
# #%%
# import lpips
# lpipsnet = lpips.LPIPS(net='alex').cuda('cuda')
# def get_lpips_similarity(lpipsnet, imgA, imgB):
#     r"""
#     Computes the image similarity between two images imgA and imgB.
#     Used to determine the optimal point of insertion to create smooth transitions.
#     High values indicate low similarity.
#     """
#     imgA = np.asarray(imgA)
#     imgB = np.asarray(imgB)
#     tensorA = torch.from_numpy(imgA).float().cuda('cuda')
#     tensorA = 2 * tensorA / 255.0 - 1
#     tensorA = tensorA.permute([2, 0, 1]).unsqueeze(0)
#     tensorB = torch.from_numpy(imgB).float().cuda('cuda')
#     tensorB = 2 * tensorB / 255.0 - 1
#     tensorB = tensorB.permute([2, 0, 1]).unsqueeze(0)
#     lploss = lpipsnet(tensorA, tensorB)
#     lploss = float(lploss[0][0][0][0])
#     return lploss



#%%






#%%

# from torch.optim import Adam

# latents_prev = pipe(image=ctrl_img_prev, latents=latents, controlnet_conditioning_scale=controlnet_conditioning_scale, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds, generator=generator, output_type='latent')[0]
# # Assuming your latents and reference_latents are already defined as tensors
# latents_delta = torch.tensor(latents, requires_grad=True)

# # Define your diffuse function here
# def diffuse(latents_delta):
#     # latents_current = latents + latents_delta
#     latents_new = pipe(image=ctrl_img_prev, latents=latents_delta, controlnet_conditioning_scale=controlnet_conditioning_scale, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds, generator=generator, output_type='latent')[0]
#     return latents_new

# # Loss function
# loss_fn = torch.nn.MSELoss()

# # Optimizer (using Adam here)
# optimizer = Adam([latents_delta], lr=0.001)  # learning rate may need adjustment

# # Optimization loop
# for _ in range(100):  # number of iterations
#     optimizer.zero_grad()   # Clear gradients
#     latents_new = diffuse(latents_delta)  # Apply your model
#     loss = loss_fn(latents_new, latents_prev)  # Compute the loss
#     loss.backward()  # Compute gradients
#     optimizer.step()  # Update parameters

#     print(f"Current loss: {loss.item()}")




 #%%
# diff_best = 1
# image_diffusion_base = image_diffusion.copy()
# #%%
# prev_image_diffusion
# #%%
# while True:
#     latents2 = torch.randn((1,4,64//1,64)).half().cuda()
#     generator.manual_seed(420)
#     fract_mixing = np.random.rand()*0.2#akai_lpd8.get("H0", val_default=0.0)
#     latents = blender.interpolate_spherical(latents1, latents2, fract_mixing) 
#     # latents = blender.interpolate_spherical(latents1, latents2, fract_mixing) 
#     image_diffusion = pipe(image=ctrl_img, latents=latents, controlnet_conditioning_scale=controlnet_conditioning_scale, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds, generator=generator).images[0]
#     diff = get_lpips_similarity(lpipsnet, prev_image_diffusion, image_diffusion)
    
#     if diff < diff_best:
#         diff_best = diff
#         print(diff)
#         best_fract_mixing = np.copy(fract_mixing)
#         image_diffusion_best = image_diffusion.copy()


# #%%
# x = stitch_images(image_diffusion_base, prev_image_diffusion)
# y = stitch_images(x, image_diffusion_best)
