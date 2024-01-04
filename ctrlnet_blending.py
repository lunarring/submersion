#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, StableDiffusionXLControlNetPipeline
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
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
# from img_utils import pad_image_to_width, pad_image_to_width, blend_images, process_cam_img, stitch_images, weighted_average_images
from datetime import datetime


import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from diffusers.utils.import_utils import is_invisible_watermark_available

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import (
    FromSingleFileMixin,
    IPAdapterMixin,
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.models import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import is_compiled_module, is_torch_version, randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput

from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from tqdm import tqdm

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def get_ctrl_img(cam_img, low_threshold=100, high_threshold=200):
    cam_img = np.array(cam_img)
    ctrl_image = cv2.Canny(cam_img, low_threshold, high_threshold)
    ctrl_image = ctrl_image[:, :, None]
    ctrl_image = np.concatenate([ctrl_image, ctrl_image, ctrl_image], axis=2)
    ctrl_image = Image.fromarray(ctrl_image)
    return ctrl_image

def blend_images(img1, img2, weight):
    # Convert images to numpy arrays
    arr1 = np.array(img1, dtype=np.float64)
    arr2 = np.array(img2, dtype=np.float64)

    # Blend images
    blended_arr = arr1 * (1-weight) + arr2 * weight
    blended_arr = np.clip(blended_arr, 0, 255)

    # Convert back to image
    return Image.fromarray(blended_arr.astype(np.uint8))

@torch.no_grad()
def call_custom(
    self,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    image: PipelineImageInput = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    inject_ctrl_blocks = False,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    ip_adapter_image: Optional[PipelineImageInput] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
    guess_mode: bool = False,
    control_guidance_start: Union[float, List[float]] = 0.0,
    control_guidance_end: Union[float, List[float]] = 1.0,
    original_size: Tuple[int, int] = None,
    crops_coords_top_left: Tuple[int, int] = (0, 0),
    target_size: Tuple[int, int] = None,
    negative_original_size: Optional[Tuple[int, int]] = None,
    negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
    negative_target_size: Optional[Tuple[int, int]] = None,
    clip_skip: Optional[int] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    averaging_weight=0,
    **kwargs,
):

    callback = kwargs.pop("callback", None)
    callback_steps = kwargs.pop("callback_steps", None)

    if callback is not None:
        deprecate(
            "callback",
            "1.0.0",
            "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
        )
    if callback_steps is not None:
        deprecate(
            "callback_steps",
            "1.0.0",
            "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
        )

    controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet
    
    if not inject_ctrl_blocks:
        self.last_down_block_res_samples = [None]*num_inference_steps
        self.last_mid_block_res_sample = [None]*num_inference_steps
    else:
        assert len(self.last_down_block_res_samples) == num_inference_steps
        assert len(self.last_mid_block_res_sample) == num_inference_steps

    # align format for control guidance
    if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
        control_guidance_start = len(control_guidance_end) * [control_guidance_start]
    elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
        control_guidance_end = len(control_guidance_start) * [control_guidance_end]
    elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
        mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
        control_guidance_start, control_guidance_end = (
            mult * [control_guidance_start],
            mult * [control_guidance_end],
        )

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        prompt_2,
        image,
        callback_steps,
        negative_prompt,
        negative_prompt_2,
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
        controlnet_conditioning_scale,
        control_guidance_start,
        control_guidance_end,
        callback_on_step_end_tensor_inputs,
    )

    self._guidance_scale = guidance_scale
    self._clip_skip = clip_skip
    self._cross_attention_kwargs = cross_attention_kwargs

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
        controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

    global_pool_conditions = (
        controlnet.config.global_pool_conditions
        if isinstance(controlnet, ControlNetModel)
        else controlnet.nets[0].config.global_pool_conditions
    )
    guess_mode = guess_mode or global_pool_conditions

    # 3.1 Encode input prompt
    text_encoder_lora_scale = (
        self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
    )
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = self.encode_prompt(
        prompt,
        prompt_2,
        device,
        num_images_per_prompt,
        self.do_classifier_free_guidance,
        negative_prompt,
        negative_prompt_2,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        lora_scale=text_encoder_lora_scale,
        clip_skip=self.clip_skip,
    )



    # 4. Prepare image
    if isinstance(controlnet, ControlNetModel):
        image = self.prepare_image(
            image=image,
            width=width,
            height=height,
            batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            dtype=controlnet.dtype,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            guess_mode=guess_mode,
        )
        height, width = image.shape[-2:]
    elif isinstance(controlnet, MultiControlNetModel):
        images = []

        for image_ in image:
            image_ = self.prepare_image(
                image=image_,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                guess_mode=guess_mode,
            )

            images.append(image_)

        image = images
        height, width = image[0].shape[-2:]
    else:
        assert False

    # 5. Prepare timesteps
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps
    self._num_timesteps = len(timesteps)

    # 6. Prepare latent variables
    num_channels_latents = self.unet.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 6.5 Optionally get Guidance Scale Embedding
    timestep_cond = None
    if self.unet.config.time_cond_proj_dim is not None:
        guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
        timestep_cond = self.get_guidance_scale_embedding(
            guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
        ).to(device=device, dtype=latents.dtype)

    # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 7.1 Create tensor stating which controlnets to keep
    controlnet_keep = []
    for i in range(len(timesteps)):
        keeps = [
            1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
            for s, e in zip(control_guidance_start, control_guidance_end)
        ]
        controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

    # 7.2 Prepare added time ids & embeddings
    if isinstance(image, list):
        original_size = original_size or image[0].shape[-2:]
    else:
        original_size = original_size or image.shape[-2:]
    target_size = target_size or (height, width)

    add_text_embeds = pooled_prompt_embeds
    if self.text_encoder_2 is None:
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
        text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

    add_time_ids = self._get_add_time_ids(
        original_size,
        crops_coords_top_left,
        target_size,
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    )

    if negative_original_size is not None and negative_target_size is not None:
        negative_add_time_ids = self._get_add_time_ids(
            negative_original_size,
            negative_crops_coords_top_left,
            negative_target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
    else:
        negative_add_time_ids = add_time_ids

    if self.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

    # 8. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    is_unet_compiled = is_compiled_module(self.unet)
    is_controlnet_compiled = is_compiled_module(self.controlnet)
    is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # Relevant thread:
            # https://dev-discuss.pytorch.org/t/cudagraphs-in-pytorch-2-0/1428
            if (is_unet_compiled and is_controlnet_compiled) and is_torch_higher_equal_2_1:
                torch._inductor.cudagraph_mark_step_begin()
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

            if inject_ctrl_blocks:
                down_block_res_samples = self.last_down_block_res_samples[i]
                mid_block_res_sample = self.last_mid_block_res_sample[i]
            else:
                # controlnet(s) inference
                if guess_mode and self.do_classifier_free_guidance:
                    # Infer ControlNet only for the conditional batch.
                    control_model_input = latents
                    control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                    controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                    controlnet_added_cond_kwargs = {
                        "text_embeds": add_text_embeds.chunk(2)[1],
                        "time_ids": add_time_ids.chunk(2)[1],
                    }
                else:
                    control_model_input = latent_model_input
                    controlnet_prompt_embeds = prompt_embeds
                    controlnet_added_cond_kwargs = added_cond_kwargs
    
                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]
    
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    control_model_input,
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=image,
                    conditioning_scale=cond_scale,
                    guess_mode=guess_mode,
                    added_cond_kwargs=controlnet_added_cond_kwargs,
                    return_dict=False,
                )
    
                if guess_mode and self.do_classifier_free_guidance:
                    # Infered ControlNet only for the conditional batch.
                    # To apply the output of ControlNet to both the unconditional and conditional batches,
                    # add 0 to the unconditional batch to keep it unchanged.
                    down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                    mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])



                self.last_down_block_res_samples[i] = down_block_res_samples
                self.last_mid_block_res_sample[i] = mid_block_res_sample
            # print(i)
# 
            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=self.cross_attention_kwargs,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, latents)

    # manually for max memory savings
    if self.vae.dtype == torch.float16 and self.vae.config.force_upcast:
        self.upcast_vae()
        latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

    if not output_type == "latent":
        # make sure the VAE is in float32 mode, as it overflows in float16
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

        if needs_upcasting:
            self.upcast_vae()
            latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)
    else:
        image = latents

    if not output_type == "latent":
        # apply watermark if available
        if self.watermark is not None:
            image = self.watermark.apply_watermark(image)

        image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
        return (image,)

    return self, StableDiffusionXLPipelineOutput(images=image)

def interpolate_linear(list0, list1, fract_mixing, scale=1.0):
    w1 = fract_mixing
    w2 = 1 - w1
    list_new = [None]*len(list0)
    for i in range(len(list0)):
        weighted_avg = (list0[i] * w1) + (list1[i] * w2) 
        list_new[i] = weighted_avg * scale
    return list_new

def get_img_mixed(pipe, fract):
    scale = 1.0 + 1.5*(0.5-np.abs(0.5-fract))
    pipe.last_mid_block_res_sample = interpolate_linear(mid_block1, mid_block2, fract, scale)
    pipe.last_down_block_res_samples[0] = interpolate_linear(down_block1[0], down_block2[0], fract, scale)
    pipe.last_down_block_res_samples[1] = interpolate_linear(down_block1[1], down_block2[1], fract, scale)
    
    prompt_embeds_mix, negative_prompt_embeds_mix, pooled_prompt_embeds_mix, negative_pooled_prompt_embeds_mix = blender.blend_prompts(embeds1, embeds2, fract)
    
    torch.manual_seed(seed)
    _, imagemix = call_custom(pipe, image=ctrl_img_empty, latents=latents1, inject_ctrl_blocks=True, controlnet_conditioning_scale=css_mid, guidance_scale=0.0, num_inference_steps=num_inference_steps, prompt_embeds=prompt_embeds_mix, negative_prompt_embeds=negative_prompt_embeds_mix, pooled_prompt_embeds=pooled_prompt_embeds_mix, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds_mix)
    imagemix = imagemix.images[0]
    return imagemix

import lpips
lpipsnet = lpips.LPIPS(net='alex').cuda('cuda')

def get_lpips_similarity(lpipsnet, imgA, imgB):
    r"""
    Computes the image similarity between two images imgA and imgB.
    Used to determine the optimal point of insertion to create smooth transitions.
    High values indicate low similarity.
    """
    imgA = np.asarray(imgA)
    imgB = np.asarray(imgB)
    tensorA = torch.from_numpy(imgA).float().cuda('cuda')
    tensorA = 2 * tensorA / 255.0 - 1
    tensorA = tensorA.permute([2, 0, 1]).unsqueeze(0)
    tensorB = torch.from_numpy(imgB).float().cuda('cuda')
    tensorB = 2 * tensorB / 255.0 - 1
    tensorB = tensorB.permute([2, 0, 1]).unsqueeze(0)
    lploss = lpipsnet(tensorA, tensorB)
    lploss = float(lploss[0][0][0][0])
    return lploss

def find_highest_similarity_and_inject_new_image(lpipsnet, list_imgs, list_fracts, pipe):
    # Compute similarities between consecutive images
    similarities = [get_lpips_similarity(lpipsnet, list_imgs[i], list_imgs[i + 1]) for i in range(len(list_imgs) - 1)]
    # similarities[0]*=0.1
    # similarities[-1]*=0.1
    # Find the index with the highest similarity
    highest_similarity_index = similarities.index(max(similarities))
    
    # print(f"{highest_similarity_index} {similarities}")

    # Calculate new fract value
    new_fract = (list_fracts[highest_similarity_index] + list_fracts[highest_similarity_index + 1]) / 2

    # Generate the new mixed image
    img_mixed = get_img_mixed(pipe, new_fract)

    # Inject new image and fract into the lists
    list_imgs.insert(highest_similarity_index+1, img_mixed)
    list_fracts.insert(highest_similarity_index+1, new_fract)
    return list_imgs, list_fracts


#%% inits
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0-mid",
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
).to("cuda")
pipe = StableDiffusionXLControlNetPipeline.from_pretrained("stabilityai/sdxl-turbo", controlnet=controlnet, torch_dtype=torch.float16, variant="fp16")
pipe = pipe.to("cuda")

# pipe.vae = AutoencoderTiny.from_pretrained('madebyollin/taesdxl', torch_device='cuda', torch_dtype=torch.float16)
# pipe.vae = pipe.vae.cuda()
pipe.set_progress_bar_config(disable=True)

blender = PromptBlender(pipe)


#%%
prompt1 = "painting of a tree"
prompt2 = "painting of a cloud that looks like flowers"
negative_prompt = "blurry, ugly"
num_inference_steps = 2
nmb_total_images = 60
ccs_begin = 0.3
seed = 421

# basis on CONSTANT latents_start for everything
size_diff_img = (512, 512)
ctrl_img_empty = Image.fromarray(np.zeros((size_diff_img[1], size_diff_img[0], 3), dtype=np.uint8))
latents1 = torch.randn((1,4,64//1,64)).half().cuda()
embeds1 = blender.get_prompt_embeds(prompt1)
embeds2 = blender.get_prompt_embeds(prompt2)

prompt_embeds1, negative_prompt_embeds1, pooled_prompt_embeds1, negative_pooled_prompt_embeds1 = embeds1
prompt_embeds2, negative_prompt_embeds2, pooled_prompt_embeds2, negative_pooled_prompt_embeds2 = embeds2

# make image1_raw with controlnet_conditioning_scale=0
torch.manual_seed(seed)
image1_raw = pipe(image=ctrl_img_empty, latents=latents1, controlnet_conditioning_scale=0.0, guidance_scale=0.0, num_inference_steps=num_inference_steps, prompt_embeds=prompt_embeds1, negative_prompt_embeds=negative_prompt_embeds1, pooled_prompt_embeds=pooled_prompt_embeds1, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds1).images[0]

# use image1_raw as ctrlnet image and make another one with same prompt but controlnet_conditioning_scale=0.5 -> image1
torch.manual_seed(seed)
image1_ctrl = get_ctrl_img(image1_raw)
pipe, image1 = call_custom(pipe, image=image1_ctrl, latents=latents1, controlnet_conditioning_scale=ccs_begin, guidance_scale=0.0, num_inference_steps=num_inference_steps, prompt_embeds=prompt_embeds1, negative_prompt_embeds=negative_prompt_embeds1, pooled_prompt_embeds=pooled_prompt_embeds1, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds1)
image1 = image1.images[0]
mid_block1 = pipe.last_mid_block_res_sample.copy()
down_block1 = pipe.last_down_block_res_samples.copy()

# get img2
torch.manual_seed(seed)
pipe, image2 = call_custom(pipe, image=image1_ctrl, latents=latents1, controlnet_conditioning_scale=ccs_begin, guidance_scale=0.0, num_inference_steps=num_inference_steps, prompt_embeds=prompt_embeds2, negative_prompt_embeds=negative_prompt_embeds2, pooled_prompt_embeds=pooled_prompt_embeds2, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds2)
image2 = image2.images[0]
mid_block2 = pipe.last_mid_block_res_sample.copy()
down_block2 = pipe.last_down_block_res_samples.copy()



#% get imgs between
list_imgs = [get_img_mixed(pipe, 0.0), get_img_mixed(pipe, 1.0)]
list_fracts = [0.0, 1.0]


for i in tqdm(range(nmb_total_images-2)):
    list_imgs, list_fracts = find_highest_similarity_and_inject_new_image(lpipsnet, list_imgs, list_fracts, pipe)

ms = lt.MovieSaver(f"test.mp4")
for imageblend in tqdm(list_imgs):
    ms.write_frame(np.asarray(imageblend))
    
ms.write_frame(np.asarray(image2))
ms.finalize()
    

