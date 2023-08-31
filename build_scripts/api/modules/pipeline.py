import json
import logging
import math
import os
import sys
import hashlib

import torch
import numpy as np
from PIL import Image, ImageOps
import random
import cv2
from skimage import exposure
from typing import Any, Dict, List
import copy

# import modules.sd_hijack
from modules import devices, prompt_parser, masking, sd_samplers, lowvram, generation_parameters_copypaste, extra_networks, sd_vae_approx, scripts, sd_samplers_common, sd_unet, errors
# from modules.sd_hijack import model_hijack
from modules.shared import opts, cmd_opts, state
import modules.shared as shared
import modules.paths as paths
import modules.face_restoration
import modules.images as images
import modules.styles
import modules.sd_models as sd_models
import modules.sd_vae as sd_vae
# from ldm.data.util import AddMiDaS
# from ldm.models.diffusion.ddpm import LatentDepth2ImageDiffusion
from tqdm import tqdm

from einops import repeat, rearrange
from blendmodes.blend import blendLayers, BlendType

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# some of those options should not be changed at all because they would break the model, so I removed them from options.
opt_C = 4
opt_f = 8
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionPipeline, StableDiffusionControlNetPipeline, StableDiffusionXLControlNetPipeline
# pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32)
# test_pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
# test_pipeline.to("cuda")

def setup_color_correction(image):
    logging.info("Calibrating color correction.")
    correction_target = cv2.cvtColor(np.asarray(image.copy()), cv2.COLOR_RGB2LAB)
    return correction_target


def apply_color_correction(correction, original_image):
    logging.info("Applying color correction.")
    image = Image.fromarray(cv2.cvtColor(exposure.match_histograms(
        cv2.cvtColor(
            np.asarray(original_image),
            cv2.COLOR_RGB2LAB
        ),
        correction,
        channel_axis=2
    ), cv2.COLOR_LAB2RGB).astype("uint8"))

    image = blendLayers(image, original_image, BlendType.LUMINOSITY)

    return image


def apply_overlay(image, paste_loc, index, overlays):
    if overlays is None or index >= len(overlays):
        return image

    overlay = overlays[index]

    if paste_loc is not None:
        x, y, w, h = paste_loc
        base_image = Image.new('RGBA', (overlay.width, overlay.height))
        image = images.resize_image(1, image, w, h)
        base_image.paste(image, (x, y))
        image = base_image

    image = image.convert('RGBA')
    image.alpha_composite(overlay)
    image = image.convert('RGB')

    return image


def txt2img_image_conditioning(sd_model, x, width, height):
    if sd_model.model.conditioning_key in {'hybrid', 'concat'}: # Inpainting models

        # The "masked-image" in this case will just be all zeros since the entire image is masked.
        image_conditioning = torch.zeros(x.shape[0], 3, height, width, device=x.device)
        image_conditioning = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(image_conditioning))

        # Add the fake full 1s mask to the first dimension.
        image_conditioning = torch.nn.functional.pad(image_conditioning, (0, 0, 0, 0, 1, 0), value=1.0)
        image_conditioning = image_conditioning.to(x.dtype)

        return image_conditioning

    elif sd_model.model.conditioning_key == "crossattn-adm": # UnCLIP models

        return x.new_zeros(x.shape[0], 2*sd_model.noise_augmentor.time_embed.dim, dtype=x.dtype, device=x.device)

    else:
        # Dummy zero conditioning if we're not using inpainting or unclip models.
        # Still takes up a bit of memory, but no encoder call.
        # Pretty sure we can just make this a 1x1 image since its not going to be used besides its batch size.
        return x.new_zeros(x.shape[0], 5, 1, 1, dtype=x.dtype, device=x.device)


class StableDiffusionProcessing:
    """
    The first set of paramaters: sd_models -> do_not_reload_embeddings represent the minimum required to create a StableDiffusionProcessing
    """
    cached_uc = [None, None]
    cached_c = [None, None]

    def __init__(
        self, 
        sd_model=None, 
        outpath_samples=None, 
        outpath_grids=None, 
        prompt: str = "", 
        styles: List[str] = None, 
        seed: int = -1, 
        subseed: int = -1, 
        subseed_strength: float = 0, 
        seed_resize_from_h: int = -1, 
        seed_resize_from_w: int = -1, 
        seed_enable_extras: bool = True, 
        sampler_name: str = None, 
        batch_size: int = 1, 
        n_iter: int = 1, 
        steps: int = 50, 
        cfg_scale: float = 7.0, 
        width: int = 512, 
        height: int = 512, 
        restore_faces: bool = False, 
        tiling: bool = False, 
        do_not_save_samples: bool = False, 
        do_not_save_grid: bool = False, 
        extra_generation_params: Dict[Any, Any] = None, 
        overlay_images: Any = None, 
        negative_prompt: str = None, 
        eta: float = None, 
        do_not_reload_embeddings: bool = False, 
        denoising_strength: float = 0, 
        ddim_discretize: str = None, 
        s_min_uncond: float = 0.0, 
        s_churn: float = 0.0, 
        s_tmax: float = None, 
        s_tmin: float = 0.0, 
        s_noise: float = 1.0, 
        override_settings: Dict[str, Any] = None, 
        override_settings_restore_afterwards: bool = True, 
        sampler_index: int = None, 
        script_args: list = None, 
        pipeline_name: str = "StableDiffusionPipeline", 
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "latent",
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        denoising_start: Optional[float] = None,
        denoising_end: Optional[float] = None,
        use_refiner: bool = False,
        strength: float = 0.3,
        aesthetic_score: float = 6.0,
        negative_aesthetic_score: float = 2.5,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        aws_dus: bool = True,
    ):
        # self.test_pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", force_download=True, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
        # # self.test_pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32)
        # self.test_pipeline.to("cuda")
        if sampler_index is not None:
            print("sampler_index argument for StableDiffusionProcessing does not do anything; use sampler_name", file=sys.stderr)

        self.outpath_samples: str = outpath_samples
        self.outpath_grids: str = outpath_grids
        self.prompt: str = prompt
        self.prompt_2: str = None
        self.prompt_for_display: str = None
        self.negative_prompt: str = (negative_prompt or "")
        self.negative_prompt_2: str = None
        self.styles: list = styles or []
        self.seed: int = seed
        self.subseed: int = subseed
        self.subseed_strength: float = subseed_strength
        self.seed_resize_from_h: int = seed_resize_from_h
        self.seed_resize_from_w: int = seed_resize_from_w
        self.sampler_name: str = sampler_name
        self.batch_size: int = batch_size
        self.n_iter: int = n_iter
        self.steps: int = steps
        self.cfg_scale: float = cfg_scale
        self.width: int = width
        self.height: int = height
        self.restore_faces: bool = restore_faces
        self.tiling: bool = tiling
        self.do_not_save_samples: bool = do_not_save_samples
        self.do_not_save_grid: bool = do_not_save_grid
        self.extra_generation_params: dict = extra_generation_params or {}
        self.overlay_images = overlay_images
        self.eta = eta
        self.do_not_reload_embeddings = do_not_reload_embeddings
        self.paste_to = None
        self.color_corrections = None
        self.denoising_strength: float = denoising_strength
        self.sampler_noise_scheduler_override = None
        self.ddim_discretize = ddim_discretize or opts.ddim_discretize
        self.s_min_uncond = s_min_uncond or opts.s_min_uncond
        self.s_churn = s_churn or opts.s_churn
        self.s_tmin = s_tmin or opts.s_tmin
        self.s_tmax = s_tmax or float('inf')  # not representable as a standard ui option
        self.s_noise = s_noise or opts.s_noise
        self.override_settings = {k: v for k, v in (override_settings or {}).items() if k not in shared.restricted_opts}
        self.override_settings_restore_afterwards = override_settings_restore_afterwards
        self.is_using_inpainting_conditioning = False
        self.disable_extra_networks = False
        self.token_merging_ratio = 0
        self.token_merging_ratio_hr = 0

        if not seed_enable_extras:
            self.subseed = -1
            self.subseed_strength = 0
            self.seed_resize_from_h = 0
            self.seed_resize_from_w = 0

        self.scripts = None
        self.script_args = script_args
        self.all_prompts = None
        self.all_negative_prompts = None
        self.all_seeds = None
        self.all_subseeds = None
        self.iteration = 0
        self.is_hr_pass = False
        self.sampler = None

        self.prompts = None
        self.negative_prompts = None
        self.extra_network_data = None
        self.seeds = None
        self.subseeds = None

        self.step_multiplier = 1
        self.cached_uc = StableDiffusionProcessing.cached_uc
        self.cached_c = StableDiffusionProcessing.cached_c
        self.uc = None
        self.c = None

        self.user = None
        # control diffuser output
        self.output_type = output_type
        self.pipeline_name = pipeline_name
        self.generator = generator
        self.prompt_embeds = prompt_embeds
        self.negative_prompt_embeds = negative_prompt_embeds
        self.output_type = output_type
        self.callback = callback
        self.callback_steps = callback_steps
        self.cross_attention_kwargs = cross_attention_kwargs

        # parameters for sdxl
        self.denoising_end = denoising_end
        self.pooled_prompt_embeds = pooled_prompt_embeds
        self.negative_pooled_prompt_embeds = negative_pooled_prompt_embeds
        self.guidance_rescale = guidance_rescale
        self.original_size = original_size
        self.crops_coords_top_left = crops_coords_top_left
        self.target_size = target_size
        self.use_refiner = use_refiner

        # parameters for refiner
        self.strength = strength
        self.denoising_start = denoising_start
        self.aesthetic_score = aesthetic_score
        self.negative_aesthetic_score = negative_aesthetic_score

        # parameters for controlnet
        self.controlnet_conditioning_scale = controlnet_conditioning_scale
        self.guess_mode = guess_mode
        self.control_guidance_start = control_guidance_start
        self.control_guidance_end = control_guidance_end

        # signal for implementation based on diffusers
        self.aws_dus = aws_dus

    @property
    def sd_model(self):
        return shared.sd_model

    @property
    def sd_pipeline(self):
        return shared.sd_pipeline
        # return self.test_pipeline

    def txt2img_image_conditioning(self, x, width=None, height=None):
        self.is_using_inpainting_conditioning = self.sd_model.model.conditioning_key in {'hybrid', 'concat'}

        return txt2img_image_conditioning(self.sd_model, x, width or self.width, height or self.height)

    def depth2img_image_conditioning(self, source_image):
        # Use the AddMiDaS helper to Format our source image to suit the MiDaS model
        transformer = AddMiDaS(model_type="dpt_hybrid")
        transformed = transformer({"jpg": rearrange(source_image[0], "c h w -> h w c")})
        midas_in = torch.from_numpy(transformed["midas_in"][None, ...]).to(device=shared.device)
        midas_in = repeat(midas_in, "1 ... -> n ...", n=self.batch_size)

        conditioning_image = self.sd_model.get_first_stage_encoding(self.sd_model.encode_first_stage(source_image))
        conditioning = torch.nn.functional.interpolate(
            self.sd_model.depth_model(midas_in),
            size=conditioning_image.shape[2:],
            mode="bicubic",
            align_corners=False,
        )

        (depth_min, depth_max) = torch.aminmax(conditioning)
        conditioning = 2. * (conditioning - depth_min) / (depth_max - depth_min) - 1.
        return conditioning

    def edit_image_conditioning(self, source_image):
        conditioning_image = self.sd_model.encode_first_stage(source_image).mode()

        return conditioning_image

    def unclip_image_conditioning(self, source_image):
        c_adm = self.sd_model.embedder(source_image)
        if self.sd_model.noise_augmentor is not None:
            noise_level = 0 # TODO: Allow other noise levels?
            c_adm, noise_level_emb = self.sd_model.noise_augmentor(c_adm, noise_level=repeat(torch.tensor([noise_level]).to(c_adm.device), '1 -> b', b=c_adm.shape[0]))
            c_adm = torch.cat((c_adm, noise_level_emb), 1)
        return c_adm

    def inpainting_image_conditioning(self, source_image, latent_image, image_mask=None):
        self.is_using_inpainting_conditioning = True

        # Handle the different mask inputs
        if image_mask is not None:
            if torch.is_tensor(image_mask):
                conditioning_mask = image_mask
            else:
                conditioning_mask = np.array(image_mask.convert("L"))
                conditioning_mask = conditioning_mask.astype(np.float32) / 255.0
                conditioning_mask = torch.from_numpy(conditioning_mask[None, None])

                # Inpainting model uses a discretized mask as input, so we round to either 1.0 or 0.0
                conditioning_mask = torch.round(conditioning_mask)
        else:
            conditioning_mask = source_image.new_ones(1, 1, *source_image.shape[-2:])

        # Create another latent image, this time with a masked version of the original input.
        # Smoothly interpolate between the masked and unmasked latent conditioning image using a parameter.
        conditioning_mask = conditioning_mask.to(device=source_image.device, dtype=source_image.dtype)
        conditioning_image = torch.lerp(
            source_image,
            source_image * (1.0 - conditioning_mask),
            getattr(self, "inpainting_mask_weight", shared.opts.inpainting_mask_weight)
        )

        # Encode the new masked image using first stage of network.
        conditioning_image = self.sd_model.get_first_stage_encoding(self.sd_model.encode_first_stage(conditioning_image))

        # Create the concatenated conditioning tensor to be fed to `c_concat`
        conditioning_mask = torch.nn.functional.interpolate(conditioning_mask, size=latent_image.shape[-2:])
        conditioning_mask = conditioning_mask.expand(conditioning_image.shape[0], -1, -1, -1)
        image_conditioning = torch.cat([conditioning_mask, conditioning_image], dim=1)
        image_conditioning = image_conditioning.to(shared.device).type(self.sd_model.dtype)

        return image_conditioning

    def img2img_image_conditioning(self, source_image, latent_image, image_mask=None):
        source_image = devices.cond_cast_float(source_image)

        # HACK: Using introspection as the Depth2Image model doesn't appear to uniquely
        # identify itself with a field common to all models. The conditioning_key is also hybrid.
        if isinstance(self.sd_model, LatentDepth2ImageDiffusion):
            return self.depth2img_image_conditioning(source_image)

        if self.sd_model.cond_stage_key == "edit":
            return self.edit_image_conditioning(source_image)

        if self.sampler.conditioning_key in {'hybrid', 'concat'}:
            return self.inpainting_image_conditioning(source_image, latent_image, image_mask=image_mask)

        if self.sampler.conditioning_key == "crossattn-adm":
            return self.unclip_image_conditioning(source_image)

        # Dummy zero conditioning if we're not using inpainting or depth model.
        return latent_image.new_zeros(latent_image.shape[0], 5, 1, 1)

    def init(self, all_prompts, all_seeds, all_subseeds):
        pass

    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
        raise NotImplementedError()

    def close(self):
        self.sampler = None
        self.c = None
        self.uc = None
        if not opts.experimental_persistent_cond_cache:
            StableDiffusionProcessing.cached_c = [None, None]
            StableDiffusionProcessing.cached_uc = [None, None]

    def get_token_merging_ratio(self, for_hr=False):
        if for_hr:
            return self.token_merging_ratio_hr or opts.token_merging_ratio_hr or self.token_merging_ratio or opts.token_merging_ratio

        return self.token_merging_ratio or opts.token_merging_ratio

    def setup_prompts(self):
        if type(self.prompt) == list:
            self.all_prompts = self.prompt
        else:
            self.all_prompts = self.batch_size * self.n_iter * [self.prompt]

        if type(self.negative_prompt) == list:
            self.all_negative_prompts = self.negative_prompt
        else:
            self.all_negative_prompts = self.batch_size * self.n_iter * [self.negative_prompt]

        self.all_prompts = [shared.prompt_styles.apply_styles_to_prompt(x, self.styles) for x in self.all_prompts]
        self.all_negative_prompts = [shared.prompt_styles.apply_negative_styles_to_prompt(x, self.styles) for x in self.all_negative_prompts]

    def get_conds_with_caching(self, function, required_prompts, steps, caches, extra_network_data):
        """
        Returns the result of calling function(shared.sd_model, required_prompts, steps)
        using a cache to store the result if the same arguments have been used before.

        cache is an array containing two elements. The first element is a tuple
        representing the previously used arguments, or None if no arguments
        have been used before. The second element is where the previously
        computed result is stored.

        caches is a list with items described above.
        """

        cached_params = (
            required_prompts,
            steps,
            opts.CLIP_stop_at_last_layers,
            shared.sd_model.sd_checkpoint_info,
            extra_network_data,
            opts.sdxl_crop_left,
            opts.sdxl_crop_top,
            self.width,
            self.height,
        )

        for cache in caches:
            if cache[0] is not None and cached_params == cache[0]:
                return cache[1]

        cache = caches[0]

        with devices.autocast():
            cache[1] = function(shared.sd_model, required_prompts, steps)

        cache[0] = cached_params
        return cache[1]

    def setup_conds(self):
        prompts = prompt_parser.SdConditioning(self.prompts, width=self.width, height=self.height)
        negative_prompts = prompt_parser.SdConditioning(self.negative_prompts, width=self.width, height=self.height, is_negative_prompt=True)

        sampler_config = sd_samplers.find_sampler_config(self.sampler_name)
        self.step_multiplier = 2 if sampler_config and sampler_config.options.get("second_order", False) else 1
        self.uc = self.get_conds_with_caching(prompt_parser.get_learned_conditioning, negative_prompts, self.steps * self.step_multiplier, [self.cached_uc], self.extra_network_data)
        self.c = self.get_conds_with_caching(prompt_parser.get_multicond_learned_conditioning, prompts, self.steps * self.step_multiplier, [self.cached_c], self.extra_network_data)

    def parse_extra_network_prompts(self):
        self.prompts, self.extra_network_data = extra_networks.parse_prompts(self.prompts)


class Processed:
    def __init__(self, p: StableDiffusionProcessing, images_list, seed=-1, info="", subseed=None, all_prompts=None, all_negative_prompts=None, all_seeds=None, all_subseeds=None, index_of_first_image=0, infotexts=None, comments=""):
        self.images = images_list
        self.prompt = p.prompt
        self.negative_prompt = p.negative_prompt
        self.seed = seed
        self.subseed = subseed
        self.subseed_strength = p.subseed_strength
        self.info = info
        self.comments = comments
        self.width = p.width
        self.height = p.height
        self.sampler_name = p.sampler_name
        self.cfg_scale = p.cfg_scale
        self.image_cfg_scale = getattr(p, 'image_cfg_scale', None)
        self.steps = p.steps
        self.batch_size = p.batch_size
        self.restore_faces = p.restore_faces
        self.face_restoration_model = opts.face_restoration_model if p.restore_faces else None
        # self.sd_model_hash = shared.sd_model.sd_model_hash
        self.sd_model_hash = None
        self.seed_resize_from_w = p.seed_resize_from_w
        self.seed_resize_from_h = p.seed_resize_from_h
        self.denoising_strength = getattr(p, 'denoising_strength', None)
        self.extra_generation_params = p.extra_generation_params
        self.index_of_first_image = index_of_first_image
        self.styles = p.styles
        self.job_timestamp = state.job_timestamp
        self.clip_skip = opts.CLIP_stop_at_last_layers
        self.token_merging_ratio = p.token_merging_ratio
        self.token_merging_ratio_hr = p.token_merging_ratio_hr

        self.eta = p.eta
        self.ddim_discretize = p.ddim_discretize
        self.s_churn = p.s_churn
        self.s_tmin = p.s_tmin
        self.s_tmax = p.s_tmax
        self.s_noise = p.s_noise
        self.s_min_uncond = p.s_min_uncond
        self.sampler_noise_scheduler_override = p.sampler_noise_scheduler_override
        self.prompt = self.prompt if type(self.prompt) != list else self.prompt[0]
        self.negative_prompt = self.negative_prompt if type(self.negative_prompt) != list else self.negative_prompt[0]
        self.seed = int(self.seed if type(self.seed) != list else self.seed[0]) if self.seed is not None else -1
        self.subseed = int(self.subseed if type(self.subseed) != list else self.subseed[0]) if self.subseed is not None else -1
        self.is_using_inpainting_conditioning = p.is_using_inpainting_conditioning

        self.all_prompts = all_prompts or p.all_prompts or [self.prompt]
        self.all_negative_prompts = all_negative_prompts or p.all_negative_prompts or [self.negative_prompt]
        self.all_seeds = all_seeds or p.all_seeds or [self.seed]
        self.all_subseeds = all_subseeds or p.all_subseeds or [self.subseed]
        self.infotexts = infotexts or [info]

    def js(self):
        obj = {
            "prompt": self.all_prompts[0],
            "all_prompts": self.all_prompts,
            "negative_prompt": self.all_negative_prompts[0],
            "all_negative_prompts": self.all_negative_prompts,
            "seed": self.seed,
            "all_seeds": self.all_seeds,
            "subseed": self.subseed,
            "all_subseeds": self.all_subseeds,
            "subseed_strength": self.subseed_strength,
            "width": self.width,
            "height": self.height,
            "sampler_name": self.sampler_name,
            "cfg_scale": self.cfg_scale,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "restore_faces": self.restore_faces,
            "face_restoration_model": self.face_restoration_model,
            "sd_model_hash": self.sd_model_hash,
            "seed_resize_from_w": self.seed_resize_from_w,
            "seed_resize_from_h": self.seed_resize_from_h,
            "denoising_strength": self.denoising_strength,
            "extra_generation_params": self.extra_generation_params,
            "index_of_first_image": self.index_of_first_image,
            "infotexts": self.infotexts,
            "styles": self.styles,
            "job_timestamp": self.job_timestamp,
            "clip_skip": self.clip_skip,
            "is_using_inpainting_conditioning": self.is_using_inpainting_conditioning,
        }

        return json.dumps(obj)

    def infotext(self, p: StableDiffusionProcessing, index):
        return create_infotext(p, self.all_prompts, self.all_seeds, self.all_subseeds, comments=[], position_in_batch=index % self.batch_size, iteration=index // self.batch_size)

    def get_token_merging_ratio(self, for_hr=False):
        return self.token_merging_ratio_hr if for_hr else self.token_merging_ratio


# from https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/3
def slerp(val, low, high):
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    dot = (low_norm*high_norm).sum(1)

    if dot.mean() > 0.9995:
        return low * val + high * (1 - val)

    omega = torch.acos(dot)
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res


def create_random_tensors(shape, seeds, subseeds=None, subseed_strength=0.0, seed_resize_from_h=0, seed_resize_from_w=0, p=None):
    eta_noise_seed_delta = opts.eta_noise_seed_delta or 0
    xs = []

    # if we have multiple seeds, this means we are working with batch size>1; this then
    # enables the generation of additional tensors with noise that the sampler will use during its processing.
    # Using those pre-generated tensors instead of simple torch.randn allows a batch with seeds [100, 101] to
    # produce the same images as with two batches [100], [101].
    if p is not None and p.sampler is not None and (len(seeds) > 1 and opts.enable_batch_seeds or eta_noise_seed_delta > 0):
        sampler_noises = [[] for _ in range(p.sampler.number_of_needed_noises(p))]
    else:
        sampler_noises = None

    for i, seed in enumerate(seeds):
        noise_shape = shape if seed_resize_from_h <= 0 or seed_resize_from_w <= 0 else (shape[0], seed_resize_from_h//8, seed_resize_from_w//8)

        subnoise = None
        if subseeds is not None:
            subseed = 0 if i >= len(subseeds) else subseeds[i]

            subnoise = devices.randn(subseed, noise_shape)

        # randn results depend on device; gpu and cpu get different results for same seed;
        # the way I see it, it's better to do this on CPU, so that everyone gets same result;
        # but the original script had it like this, so I do not dare change it for now because
        # it will break everyone's seeds.
        noise = devices.randn(seed, noise_shape)

        if subnoise is not None:
            noise = slerp(subseed_strength, noise, subnoise)

        if noise_shape != shape:
            x = devices.randn(seed, shape)
            dx = (shape[2] - noise_shape[2]) // 2
            dy = (shape[1] - noise_shape[1]) // 2
            w = noise_shape[2] if dx >= 0 else noise_shape[2] + 2 * dx
            h = noise_shape[1] if dy >= 0 else noise_shape[1] + 2 * dy
            tx = 0 if dx < 0 else dx
            ty = 0 if dy < 0 else dy
            dx = max(-dx, 0)
            dy = max(-dy, 0)

            x[:, ty:ty+h, tx:tx+w] = noise[:, dy:dy+h, dx:dx+w]
            noise = x

        if sampler_noises is not None:
            cnt = p.sampler.number_of_needed_noises(p)

            if eta_noise_seed_delta > 0:
                torch.manual_seed(seed + eta_noise_seed_delta)

            for j in range(cnt):
                sampler_noises[j].append(devices.randn_without_seed(tuple(noise_shape)))

        xs.append(noise)

    if sampler_noises is not None:
        p.sampler.sampler_noises = [torch.stack(n).to(shared.device) for n in sampler_noises]

    x = torch.stack(xs).to(shared.device)
    return x


def decode_latent_batch(model, batch, target_device=None, check_for_nans=False):
    samples = []

    for i in range(batch.shape[0]):
        sample = decode_first_stage(model, batch[i:i + 1])[0]

        if check_for_nans:
            try:
                devices.test_for_nans(sample, "vae")
            except devices.NansException as e:
                if devices.dtype_vae == torch.float32 or not shared.opts.auto_vae_precision:
                    raise e

                errors.print_error_explanation(
                    "A tensor with all NaNs was produced in VAE.\n"
                    "Web UI will now convert VAE into 32-bit float and retry.\n"
                    "To disable this behavior, disable the 'Automaticlly revert VAE to 32-bit floats' setting.\n"
                    "To always start with 32-bit VAE, use --no-half-vae commandline flag."
                )

                devices.dtype_vae = torch.float32
                model.first_stage_model.to(devices.dtype_vae)
                batch = batch.to(devices.dtype_vae)

                sample = decode_first_stage(model, batch[i:i + 1])[0]

        if target_device is not None:
            sample = sample.to(target_device)

        samples.append(sample)

    return samples


def decode_first_stage(model, x):
    x = model.decode_first_stage(x.to(devices.dtype_vae))

    return x


def get_fixed_seed(seed):
    if seed is None or seed == '' or seed == -1:
        return int(random.randrange(4294967294))

    return seed


def fix_seed(p):
    p.seed = get_fixed_seed(p.seed)
    p.subseed = get_fixed_seed(p.subseed)


def program_version():
    import launch

    res = launch.git_tag()
    if res == "<none>":
        res = None

    return res


def create_infotext(p, all_prompts, all_seeds, all_subseeds, comments=None, iteration=0, position_in_batch=0, use_main_prompt=False):
    index = position_in_batch + iteration * p.batch_size

    clip_skip = getattr(p, 'clip_skip', opts.CLIP_stop_at_last_layers)
    enable_hr = getattr(p, 'enable_hr', False)
    token_merging_ratio = p.get_token_merging_ratio()
    token_merging_ratio_hr = p.get_token_merging_ratio(for_hr=True)

    uses_ensd = opts.eta_noise_seed_delta != 0
    if uses_ensd:
        uses_ensd = sd_samplers_common.is_sampler_using_eta_noise_seed_delta(p)

    generation_params = {
        "Steps": p.steps,
        "Sampler": p.sampler_name,
        "CFG scale": p.cfg_scale,
        "Image CFG scale": getattr(p, 'image_cfg_scale', None),
        "Seed": all_seeds[index],
        "Face restoration": (opts.face_restoration_model if p.restore_faces else None),
        "Size": f"{p.width}x{p.height}",
        # "Model hash": getattr(p, 'sd_model_hash', None if not opts.add_model_hash_to_info or not shared.sd_model.sd_model_hash else shared.sd_model.sd_model_hash),
        "Model hash": None,
        # "Model": (None if not opts.add_model_name_to_info else shared.sd_model.sd_checkpoint_info.name_for_extra),
        "Model": None,
        # (None if not opts.add_model_name_to_info else shared.sd_model.sd_checkpoint_info.name_for_extra),
        "Variation seed": (None if p.subseed_strength == 0 else all_subseeds[index]),
        "Variation seed strength": (None if p.subseed_strength == 0 else p.subseed_strength),
        "Seed resize from": (None if p.seed_resize_from_w <= 0 or p.seed_resize_from_h <= 0 else f"{p.seed_resize_from_w}x{p.seed_resize_from_h}"),
        "Denoising strength": getattr(p, 'denoising_strength', None),
        "Conditional mask weight": getattr(p, "inpainting_mask_weight", shared.opts.inpainting_mask_weight) if p.is_using_inpainting_conditioning else None,
        "Clip skip": None if clip_skip <= 1 else clip_skip,
        "ENSD": opts.eta_noise_seed_delta if uses_ensd else None,
        "Token merging ratio": None if token_merging_ratio == 0 else token_merging_ratio,
        "Token merging ratio hr": None if not enable_hr or token_merging_ratio_hr == 0 else token_merging_ratio_hr,
        "Init image hash": getattr(p, 'init_img_hash', None),
        "RNG": opts.randn_source if opts.randn_source != "GPU" else None,
        "NGMS": None if p.s_min_uncond == 0 else p.s_min_uncond,
        **p.extra_generation_params,
        "Version": program_version() if opts.add_version_to_infotext else None,
        "User": p.user if opts.add_user_name_to_info else None,
    }

    generation_params_text = ", ".join([k if k == v else f'{k}: {generation_parameters_copypaste.quote(v)}' for k, v in generation_params.items() if v is not None])

    prompt_text = p.prompt if use_main_prompt else all_prompts[index]
    negative_prompt_text = f"\nNegative prompt: {p.all_negative_prompts[index]}" if p.all_negative_prompts[index] else ""

    return f"{prompt_text}{negative_prompt_text}\n{generation_params_text}".strip()


def process_images(p: StableDiffusionProcessing) -> Processed:
    if p.scripts is not None:
        p.scripts.before_process(p)

    stored_opts = {k: opts.data[k] for k in p.override_settings.keys()}

    try:
        # # if no checkpoint override or the override checkpoint can't be found, remove override entry and load opts checkpoint
        # if sd_models.checkpoint_aliases.get(p.override_settings.get('sd_model_checkpoint')) is None:
        #     p.override_settings.pop('sd_model_checkpoint', None)
        #     sd_models.reload_model_weights()

        # for k, v in p.override_settings.items():
        #     setattr(opts, k, v)

        #     if k == 'sd_model_checkpoint':
        #         sd_models.reload_model_weights()

        #     if k == 'sd_vae':
        #         sd_vae.reload_vae_weights()

        # sd_models.apply_token_merging(p.sd_model, p.get_token_merging_ratio())

        res = process_images_inner(p)
        # pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", force_download=True, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
        # pipe.to("cuda")
        # prompt = "An astronaut riding a green horse"
        # images = pipe(prompt=prompt).images[0]

    finally:
        # sd_models.apply_token_merging(p.sd_model, 0)

        # restore opts to original state
        if p.override_settings_restore_afterwards:
            for k, v in stored_opts.items():
                setattr(opts, k, v)

                if k == 'sd_vae':
                    sd_vae.reload_vae_weights()

    return res

def check_controlnet(p: StableDiffusionProcessing):
    controlnet_state = False
    valid_script = None
    for script in p.scripts.alwayson_scripts:
        api_info_name = script.api_info.name
        if api_info_name == 'controlnet':
            enabled_units_len = len(script.get_enabled_units(p))
            if enabled_units_len > 0:
                controlnet_state = True
                valid_script = script
            break
    return controlnet_state, valid_script

def process_images_inner(p: StableDiffusionProcessing) -> Processed:
    """this is the main loop that both txt2img and img2img use; it calls func_init once inside all the scopes and func_sample once per batch"""

    if type(p.prompt) == list:
        assert(len(p.prompt) > 0)
    else:
        assert p.prompt is not None

    devices.torch_gc()

    seed = get_fixed_seed(p.seed)
    subseed = get_fixed_seed(p.subseed)

    # modules.sd_hijack.model_hijack.apply_circular(p.tiling)
    # modules.sd_hijack.model_hijack.clear_comments()

    comments = {}

    p.setup_prompts()

    if type(seed) == list:
        p.all_seeds = seed
    else:
        p.all_seeds = [int(seed) + (x if p.subseed_strength == 0 else 0) for x in range(len(p.all_prompts))]

    if type(subseed) == list:
        p.all_subseeds = subseed
    else:
        p.all_subseeds = [int(subseed) + x for x in range(len(p.all_prompts))]

    def infotext(iteration=0, position_in_batch=0, use_main_prompt=False):
        return create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, comments, iteration, position_in_batch, use_main_prompt)

    # if os.path.exists(cmd_opts.embeddings_dir) and not p.do_not_reload_embeddings:
    #     model_hijack.embedding_db.load_textual_inversion_embeddings()

    if p.scripts is not None:
        p.scripts.process(p)

    infotexts = []
    output_images = []

    with torch.no_grad():
        with devices.autocast():
            p.init(p.all_prompts, p.all_seeds, p.all_subseeds)

            # for OSX, loading the model during sampling changes the generated picture, so it is loaded here
            # if shared.opts.live_previews_enable and opts.show_progress_type == "Approx NN":
            #     sd_vae_approx.model()

            # sd_unet.apply_unet()

        if state.job_count == -1:
            state.job_count = p.n_iter

        for n in range(p.n_iter):
            p.iteration = n

            if state.skipped:
                state.skipped = False

            if state.interrupted:
                break

            p.prompts = p.all_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            p.negative_prompts = p.all_negative_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            p.seeds = p.all_seeds[n * p.batch_size:(n + 1) * p.batch_size]
            p.subseeds = p.all_subseeds[n * p.batch_size:(n + 1) * p.batch_size]

            if p.scripts is not None:
                p.scripts.before_process_batch(p, batch_number=n, prompts=p.prompts, seeds=p.seeds, subseeds=p.subseeds)

            if len(p.prompts) == 0:
                break

            p.parse_extra_network_prompts()

            if not p.disable_extra_networks:
                with devices.autocast():
                    extra_networks.activate(p, p.extra_network_data)

            if p.scripts is not None:
                p.scripts.process_batch(p, batch_number=n, prompts=p.prompts, seeds=p.seeds, subseeds=p.subseeds)

            # params.txt should be saved after scripts.process_batch, since the
            # infotext could be modified by that callback
            # Example: a wildcard processed by process_batch sets an extra model
            # strength, which is saved as "Model Strength: 1.0" in the infotext
            if n == 0:
                with open(os.path.join(paths.data_path, "params.txt"), "w", encoding="utf8") as file:
                    processed = Processed(p, [], p.seed, "")
                    file.write(processed.infotext(p, 0))

            # p.setup_conds()

            # for comment in model_hijack.comments:
            #     comments[comment] = 1

            # p.extra_generation_params.update(model_hijack.extra_generation_params)

            if p.n_iter > 1:
                shared.state.job = f"Batch {n+1} out of {p.n_iter}"

            # with devices.without_autocast() if devices.unet_needs_upcast else devices.autocast():
    #             samples_ddim = p.sample(conditioning=p.c, unconditional_conditioning=p.uc, seeds=p.seeds, subseeds=p.subseeds, subseed_strength=p.subseed_strength, prompts=p.prompts)
    #             latents = 1 / p.sd_pipeline.vae.config.scaling_factor * samples_ddim
    #             image = p.sd_pipeline.vae.decode(latents).sample
    #             image = (image / 2 + 0.5).clamp(0, 1)
    #             x_samples_ddim = image
            # with devices.without_autocast() if devices.unet_needs_upcast else devices.autocast():
            # with torch.autocast("cuda"):
            # check whether controlnet is needed
            # controlnet_state, controlnet_script = check_controlnet(p)
            controlnet_state = False
            controlnet_images = []

            if controlnet_state == True:
                # TODO XY: update pipeline
                # get controled image
                pipeline_name = shared.sd_pipeline.pipeline_name
                controlnet_images = []
                for detected_map in controlnet_script.detected_map:
                    controlnet_image = detected_map[0]
                    controlnet_images.append(controlnet_image)
                if pipeline_name != 'StableDiffusionXLControlNetPipeline' and pipeline_name != 'StableDiffusionControlNetPipeline':
                    if pipeline_name == 'StableDiffusionXLPipeline':
                        shared.sd_pipeline = StableDiffusionXLControlNetPipeline(**p.sd_pipeline.components, controlnet=controlnet_script.control_networks[0])
                        shared.sd_pipeline.pipeline_name = 'StableDiffusionXLControlNetPipeline'
                    else:
                        shared.sd_pipeline = StableDiffusionControlNetPipeline(**p.sd_pipeline.components, controlnet=controlnet_script.control_networks[0])
                        shared.sd_pipeline.pipeline_name = 'StableDiffusionControlNetPipeline'
                else:
                    shared.sd_pipeline.controlnet = controlnet_script.control_networks[0]
            else:
                controlnet_image = None

            samples_ddim = p.sample(conditioning=p.c, unconditional_conditioning=p.uc, seeds=p.seeds, subseeds=p.subseeds, subseed_strength=p.subseed_strength, prompts=p.prompts, controlnet_image=controlnet_images)
            latents = 1 / p.sd_pipeline.vae.config.scaling_factor * samples_ddim
            image = p.sd_pipeline.vae.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            x_samples_ddim = image

            # x_samples_ddim = decode_latent_batch(p.sd_model, samples_ddim, target_device=devices.cpu, check_for_nans=True)
            # x_samples_ddim = torch.stack(x_samples_ddim).float()
            # x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

            del samples_ddim

            # if lowvram.is_enabled(shared.sd_model):
            #     lowvram.send_everything_to_cpu()

            devices.torch_gc()

            if p.scripts is not None:
                p.scripts.postprocess_batch(p, x_samples_ddim, batch_number=n)

            for i, x_sample in enumerate(x_samples_ddim):
                p.batch_index = i

                x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
                x_sample = x_sample.astype(np.uint8)

                if p.restore_faces:
                    if opts.save and not p.do_not_save_samples and opts.save_images_before_face_restoration:
                        images.save_image(Image.fromarray(x_sample), p.outpath_samples, "", p.seeds[i], p.prompts[i], opts.samples_format, info=infotext(n, i), p=p, suffix="-before-face-restoration")

                    devices.torch_gc()

                    x_sample = modules.face_restoration.restore_faces(x_sample)
                    devices.torch_gc()

                image = Image.fromarray(x_sample)

                if p.scripts is not None:
                    pp = scripts.PostprocessImageArgs(image)
                    p.scripts.postprocess_image(p, pp)
                    image = pp.image

                if p.color_corrections is not None and i < len(p.color_corrections):
                    if opts.save and not p.do_not_save_samples and opts.save_images_before_color_correction:
                        image_without_cc = apply_overlay(image, p.paste_to, i, p.overlay_images)
                        images.save_image(image_without_cc, p.outpath_samples, "", p.seeds[i], p.prompts[i], opts.samples_format, info=infotext(n, i), p=p, suffix="-before-color-correction")
                    image = apply_color_correction(p.color_corrections[i], image)

                image = apply_overlay(image, p.paste_to, i, p.overlay_images)

                # if opts.samples_save and not p.do_not_save_samples:
                #     images.save_image(image, p.outpath_samples, "", p.seeds[i], p.prompts[i], opts.samples_format, info=infotext(n, i), p=p)

                text = infotext(n, i)
                infotexts.append(text)
                if opts.enable_pnginfo:
                    image.info["parameters"] = text
                output_images.append(image)

                if hasattr(p, 'mask_for_overlay') and p.mask_for_overlay and any([opts.save_mask, opts.save_mask_composite, opts.return_mask, opts.return_mask_composite]):
                    image_mask = p.mask_for_overlay.convert('RGB')
                    image_mask_composite = Image.composite(image.convert('RGBA').convert('RGBa'), Image.new('RGBa', image.size), images.resize_image(2, p.mask_for_overlay, image.width, image.height).convert('L')).convert('RGBA')

                    if opts.save_mask:
                        images.save_image(image_mask, p.outpath_samples, "", p.seeds[i], p.prompts[i], opts.samples_format, info=infotext(n, i), p=p, suffix="-mask")

                    if opts.save_mask_composite:
                        images.save_image(image_mask_composite, p.outpath_samples, "", p.seeds[i], p.prompts[i], opts.samples_format, info=infotext(n, i), p=p, suffix="-mask-composite")

                    if opts.return_mask:
                        output_images.append(image_mask)

                    if opts.return_mask_composite:
                        output_images.append(image_mask_composite)

            del x_samples_ddim

            devices.torch_gc()

            state.nextjob()

        p.color_corrections = None

        index_of_first_image = 0
        unwanted_grid_because_of_img_count = len(output_images) < 2 and opts.grid_only_if_multiple
        if (opts.return_grid or opts.grid_save) and not p.do_not_save_grid and not unwanted_grid_because_of_img_count:
            grid = images.image_grid(output_images, p.batch_size)

            if opts.return_grid:
                text = infotext(use_main_prompt=True)
                infotexts.insert(0, text)
                if opts.enable_pnginfo:
                    grid.info["parameters"] = text
                output_images.insert(0, grid)
                index_of_first_image = 1

            if opts.grid_save:
                images.save_image(grid, p.outpath_grids, "grid", p.all_seeds[0], p.all_prompts[0], opts.grid_format, info=infotext(use_main_prompt=True), short_filename=not opts.grid_extended_filename, p=p, grid=True)

    if not p.disable_extra_networks and p.extra_network_data:
        extra_networks.deactivate(p, p.extra_network_data)

    devices.torch_gc()


    res = Processed(
        p,
        images_list=output_images,
        seed=p.all_seeds[0],
        info=infotext(),
        comments="".join(f"{comment}\n" for comment in comments),
        subseed=p.all_subseeds[0],
        index_of_first_image=index_of_first_image,
        infotexts=infotexts,
    )

    if p.scripts is not None:
        p.scripts.postprocess(p, res)

    return res


def old_hires_fix_first_pass_dimensions(width, height):
    """old algorithm for auto-calculating first pass size"""

    desired_pixel_count = 512 * 512
    actual_pixel_count = width * height
    scale = math.sqrt(desired_pixel_count / actual_pixel_count)
    width = math.ceil(scale * width / 64) * 64
    height = math.ceil(scale * height / 64) * 64

    return width, height

class StableDiffusionPipelineTxt2Img(StableDiffusionProcessing):
    sampler = None
    cached_hr_uc = [None, None]
    cached_hr_c = [None, None]

    def __init__(self, enable_hr: bool = False, denoising_strength: float = 0.75, firstphase_width: int = 0, firstphase_height: int = 0, hr_scale: float = 2.0, hr_upscaler: str = None, hr_second_pass_steps: int = 0, hr_resize_x: int = 0, hr_resize_y: int = 0, hr_sampler_name: str = None, hr_prompt: str = '', hr_negative_prompt: str = '', **kwargs):
        super().__init__(**kwargs)
        self.enable_hr = enable_hr
        self.denoising_strength = denoising_strength
        self.hr_scale = hr_scale
        self.hr_upscaler = hr_upscaler
        self.hr_second_pass_steps = hr_second_pass_steps
        self.hr_resize_x = hr_resize_x
        self.hr_resize_y = hr_resize_y
        self.hr_upscale_to_x = hr_resize_x
        self.hr_upscale_to_y = hr_resize_y
        self.hr_sampler_name = hr_sampler_name
        self.hr_prompt = hr_prompt
        self.hr_negative_prompt = hr_negative_prompt
        self.all_hr_prompts = None
        self.all_hr_negative_prompts = None

        if firstphase_width != 0 or firstphase_height != 0:
            self.hr_upscale_to_x = self.width
            self.hr_upscale_to_y = self.height
            self.width = firstphase_width
            self.height = firstphase_height

        self.truncate_x = 0
        self.truncate_y = 0
        self.applied_old_hires_behavior_to = None

        self.hr_prompts = None
        self.hr_negative_prompts = None
        self.hr_extra_network_data = None

        self.cached_hr_uc = StableDiffusionPipelineTxt2Img.cached_hr_uc
        self.cached_hr_c = StableDiffusionPipelineTxt2Img.cached_hr_c
        self.hr_c = None
        self.hr_uc = None
        # self.tokenizer = self.sd_pipeline.tokenizer
        # self.tokenizer_2 = None ##for SDXL
        # self.unet = self.sd_pipeline.unet
        # self.vae = self.sd_pipeline.vae
        # self.scheduler = self.sd_pipeline.scheduler
        # self.text_encoder = self.sd_pipeline.text_encoder
        # self.text_encoder_2 = None ##for SDXL
        # self.decode_latents = self.sd_pipeline.decode_latents
        # self.pipeline_name = self.sd_pipeline.pipeline_name

    def init(self, all_prompts, all_seeds, all_subseeds):
        if self.enable_hr:
            if self.hr_sampler_name is not None and self.hr_sampler_name != self.sampler_name:
                self.extra_generation_params["Hires sampler"] = self.hr_sampler_name

            if tuple(self.hr_prompt) != tuple(self.prompt):
                self.extra_generation_params["Hires prompt"] = self.hr_prompt

            if tuple(self.hr_negative_prompt) != tuple(self.negative_prompt):
                self.extra_generation_params["Hires negative prompt"] = self.hr_negative_prompt

            if opts.use_old_hires_fix_width_height and self.applied_old_hires_behavior_to != (self.width, self.height):
                self.hr_resize_x = self.width
                self.hr_resize_y = self.height
                self.hr_upscale_to_x = self.width
                self.hr_upscale_to_y = self.height

                self.width, self.height = old_hires_fix_first_pass_dimensions(self.width, self.height)
                self.applied_old_hires_behavior_to = (self.width, self.height)

            if self.hr_resize_x == 0 and self.hr_resize_y == 0:
                self.extra_generation_params["Hires upscale"] = self.hr_scale
                self.hr_upscale_to_x = int(self.width * self.hr_scale)
                self.hr_upscale_to_y = int(self.height * self.hr_scale)
            else:
                self.extra_generation_params["Hires resize"] = f"{self.hr_resize_x}x{self.hr_resize_y}"

                if self.hr_resize_y == 0:
                    self.hr_upscale_to_x = self.hr_resize_x
                    self.hr_upscale_to_y = self.hr_resize_x * self.height // self.width
                elif self.hr_resize_x == 0:
                    self.hr_upscale_to_x = self.hr_resize_y * self.width // self.height
                    self.hr_upscale_to_y = self.hr_resize_y
                else:
                    target_w = self.hr_resize_x
                    target_h = self.hr_resize_y
                    src_ratio = self.width / self.height
                    dst_ratio = self.hr_resize_x / self.hr_resize_y

                    if src_ratio < dst_ratio:
                        self.hr_upscale_to_x = self.hr_resize_x
                        self.hr_upscale_to_y = self.hr_resize_x * self.height // self.width
                    else:
                        self.hr_upscale_to_x = self.hr_resize_y * self.width // self.height
                        self.hr_upscale_to_y = self.hr_resize_y

                    self.truncate_x = (self.hr_upscale_to_x - target_w) // opt_f
                    self.truncate_y = (self.hr_upscale_to_y - target_h) // opt_f

            # special case: the user has chosen to do nothing
            if self.hr_upscale_to_x == self.width and self.hr_upscale_to_y == self.height:
                self.enable_hr = False
                self.denoising_strength = None
                self.extra_generation_params.pop("Hires upscale", None)
                self.extra_generation_params.pop("Hires resize", None)
                return

            if not state.processing_has_refined_job_count:
                if state.job_count == -1:
                    state.job_count = self.n_iter

                shared.total_tqdm.updateTotal((self.steps + (self.hr_second_pass_steps or self.steps)) * state.job_count)
                state.job_count = state.job_count * 2
                state.processing_has_refined_job_count = True

            if self.hr_second_pass_steps:
                self.extra_generation_params["Hires steps"] = self.hr_second_pass_steps

            if self.hr_upscaler is not None:
                self.extra_generation_params["Hires upscaler"] = self.hr_upscaler

    def decode_latents(self):
        return self.sd_pipeline.decode_latents

    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts, controlnet_image=None):
        # self.sampler = sd_samplers.create_sampler(self.sampler_name, self.sd_model)

        # update sampler
        sd_pipeline = sd_samplers.update_sampler(self.sampler_name, self.sd_pipeline, self.pipeline_name)
        # sd_pipeline = self.sd_pipeline

        latent_scale_mode = shared.latent_upscale_modes.get(self.hr_upscaler, None) if self.hr_upscaler is not None else shared.latent_upscale_modes.get(shared.latent_upscale_default_mode, "nearest")
        if self.enable_hr and latent_scale_mode is None:
            if not any(x.name == self.hr_upscaler for x in shared.sd_upscalers):
                raise Exception(f"could not find upscaler named {self.hr_upscaler}")

        # common parameters for sd
        prompt = self.prompt 
        height = self.width
        width = self.height
        num_inference_steps = self.steps
        guidance_scale = self.cfg_scale
        negative_prompt = self.negative_prompt or ""
        num_images_per_prompt = self.batch_size
        eta = self.eta
        generator = self.generator
        latents = create_random_tensors([opt_C, self.height // opt_f, self.width // opt_f], seeds=seeds, subseeds=subseeds, subseed_strength=self.subseed_strength, seed_resize_from_h=self.seed_resize_from_h, seed_resize_from_w=self.seed_resize_from_w, p=self)
        prompt_embeds = self.prompt_embeds
        negative_prompt_embeds = self.negative_prompt_embeds
        output_type = self.output_type
        callback = self.callback
        callback_steps = self.callback_steps
        cross_attention_kwargs = self.cross_attention_kwargs

        # parameters for sdxl
        prompt_2 = self.prompt_2 or self.prompt
        negative_prompt_2 = self.negative_prompt_2 or self.negative_prompt
        denoising_end = self.denoising_end
        pooled_prompt_embeds = self.pooled_prompt_embeds
        negative_pooled_prompt_embeds = self.negative_pooled_prompt_embeds
        guidance_rescale = self.guidance_rescale
        original_size = self.original_size
        crops_coords_top_left = self.crops_coords_top_left
        target_size = self.target_size
        use_refiner = self.use_refiner

        # parameters for refiner
        strength = self.strength
        denoising_start = self.denoising_start
        aesthetic_score = self.aesthetic_score
        negative_aesthetic_score = self.negative_aesthetic_score

        # parameters for controlnet
        controlnet_conditioning_scale = self.controlnet_conditioning_scale
        guess_mode = self.guess_mode
        control_guidance_start = self.control_guidance_start
        control_guidance_end = self.control_guidance_end

        pipeline_name = sd_pipeline.pipeline_name
        # default output: latents
        if pipeline_name == 'StableDiffusionPipeline':
            # images = sd_pipeline(
            #     prompt = prompt).images
            latents = latents.to(torch.float16)
            images = sd_pipeline(
                prompt = prompt,
                height = height,
                width = width,
                num_inference_steps = num_inference_steps,
                guidance_scale = guidance_scale,
                negative_prompt = negative_prompt,
                num_images_per_prompt= num_images_per_prompt,
                eta = eta,
                generator = generator,
                latents = latents,
                prompt_embeds = prompt_embeds,
                negative_prompt_embeds= negative_prompt_embeds,
                output_type = output_type,
                return_dict = True,
                callback = callback,
                callback_steps = callback_steps,
                cross_attention_kwargs = cross_attention_kwargs).images
        elif pipeline_name == 'StableDiffusionXLPipeline':
            latents = latents.to(torch.float16)
            images = sd_pipeline(
                prompt = prompt,
                prompt_2 = prompt_2,
                height = height,
                width = width,
                num_inference_steps = num_inference_steps,
                denoising_end = denoising_end,
                guidance_scale = guidance_scale,
                negative_prompt = negative_prompt,
                negative_prompt_2 = negative_prompt_2,
                num_images_per_prompt = num_images_per_prompt,
                eta = eta,
                generator = generator,
                latents = latents,
                prompt_embeds = prompt_embeds,
                negative_prompt_embeds = negative_prompt_embeds,
                pooled_prompt_embeds = pooled_prompt_embeds,
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds,
                output_type = output_type,
                return_dict = True,
                callback = callback,
                callback_steps = callback_steps,
                cross_attention_kwargs = cross_attention_kwargs,
                guidance_rescale = guidance_rescale,
                original_size = original_size,
                crops_coords_top_left = crops_coords_top_left,
                target_size = target_size).images
        elif pipeline_name == 'StableDiffusionControlNetPipeline':
            images = sd_pipeline(
                prompt = prompt,
                image = controlnet_image,
                height = height,
                width = width,
                num_inference_steps = num_inference_steps,
                guidance_scale = guidance_scale,
                negative_prompt = negative_prompt,
                num_images_per_prompt = num_images_per_prompt,
                eta = eta,
                generator = generator,
                latents = latents,
                prompt_embeds = prompt_embeds,
                negative_prompt_embeds = negative_prompt_embeds,
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds,
                output_type = output_type,
                return_dict = True,
                callback = callback,
                callback_steps = callback_steps,
                cross_attention_kwargs = cross_attention_kwargs,
                controlnet_conditioning_scale = controlnet_conditioning_scale,
                guess_mode = guess_mode,
                control_guidance_start = control_guidance_start,
                control_guidance_end = control_guidance_end).images
        elif pipeline_name == 'StableDiffusionXLControlNetPipeline':
            images = sd_pipeline(
                prompt = prompt,
                prompt_2 = prompt_2,
                image = controlnet_image,
                height = height,
                width = width,
                num_inference_steps = num_inference_steps,
                guidance_scale = guidance_scale,
                negative_prompt = negative_prompt,
                negative_prompt_2 = negative_prompt_2,
                num_images_per_prompt = num_images_per_prompt,
                eta = eta,
                generator = generator,
                latents = latents,
                prompt_embeds = prompt_embeds,
                negative_prompt_embeds = negative_prompt_embeds,
                #pooled_prompt_embeds = pooled_prompt_embeds,
                #negative_pooled_prompt_embeds = negative_pooled_prompt_embeds,
                output_type = output_type,
                return_dict = True,
                callback = callback,
                callback_steps = callback_steps,
                cross_attention_kwargs = cross_attention_kwargs,
                controlnet_conditioning_scale = controlnet_conditioning_scale,
                guess_mode = guess_mode,
                control_guidance_start = control_guidance_start,
                control_guidance_end = control_guidance_end,
                original_size = original_size).images

        if use_refiner:
            images = self.refiner_pipeline(
                prompt = prompt,
                prompt_2 = prompt_2,
                image = images,
                strength = strength,
                num_inference_steps = num_inference_steps,
                denoising_start = denoising_start,
                denoising_end = denoising_end,
                guidance_scale = guidance_scale,
                negative_prompt = negative_prompt,
                negative_prompt_2 = negative_prompt_2,
                num_images_per_prompt = num_images_per_prompt,
                eta = eta,
                generator = generator,
                prompt_embeds = prompt_embeds,
                negative_prompt_embeds = negative_prompt_embeds,
                pooled_prompt_embeds = pooled_prompt_embeds,
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds,
                output_type = output_type,
                return_dict = True,
                callback = callback,
                callback_steps = callback_steps,
                cross_attention_kwargs = cross_attention_kwargs,
                guidance_rescale = guidance_rescale,
                original_size = original_size,
                crops_coords_top_left = crops_coords_top_left,
                target_size = target_size,
                aesthetic_score = aesthetic_score,
                negative_aesthetic_score = negative_aesthetic_score).images
                
            # images = images.to(torch.float32)

        samples = images
        # do_classifier_free_guidance = self.cfg_scale > 1.0

        # if self.prompt is not None and isinstance(self.prompt, str):
        #     prompt_batch_size = 1
        # elif self.prompt is not None and isinstance(self.prompt, list):
        #     prompt_batch_size = len(self.prompt)

        # ## SDXL ###
        # model_type = 'SD' ### 'SDXL'
        # if model_type == 'SDXL':
        #     self.prompt_2 = self.prompt_2 or self.prompt
        #     self.negative_prompt = self.negative_prompt or ""
        #     self.negative_prompt_2 = self.negative_prompt_2 or self.negative_prompt
        
        # prompts = [self.prompt, self.prompt_2] if self.prompt_2 is not None else [self.prompt]
        # negative_prompts = [self.negative_prompt, self.negative_prompt_2] if self.negative_prompt_2 is not None else [self.negative_prompt]
        
        # # Define tokenizers and text encoders
        # tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        # text_encoders = (
        #     [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        # )

        # # prompt text embedding
        # text_embeddings_list = []
        # for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
        #     text_input = tokenizer(
        #         prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt"
        #     )
        #     with torch.no_grad():
        #         text_embeddings = text_encoder(text_input.input_ids.to(shared.device))[0]
        #     text_embeddings_list.append(text_embeddings)
        # text_embeddings = torch.concat(text_embeddings_list, dim=-1)


        # if do_classifier_free_guidance:
        #     negative_prompt_embeds_list = []
        #     max_length = text_embeddings.shape[1]
        #     for negative_prompt, tokenizer, text_encoder in zip(negative_prompts, tokenizers, text_encoders):
        #         uncond_input = tokenizer(
        #             negative_prompt,
        #             padding="max_length",
        #             max_length=max_length,
        #             truncation=True,
        #             return_tensors="pt",
        #         )

        #         negative_prompt_embeds = text_encoder(
        #             uncond_input.input_ids.to(shared.device),
        #         )[0]
        #         # # We are only ALWAYS interested in the pooled output of the final text encoder
        #         # negative_pooled_prompt_embeds = negative_prompt_embeds[0]
        #         # negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

        #         negative_prompt_embeds_list.append(negative_prompt_embeds)

        #     negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)
        
        # text_embeddings = text_embeddings.to(dtype=self.text_encoder.dtype, device=shared.device)
        # bs_embed, seq_len, _ = text_embeddings.shape
        # # duplicate text embeddings for each generation per prompt, using mps friendly method
        # text_embeddings = text_embeddings.repeat(1, self.n_iter, 1)
        # text_embeddings = text_embeddings.view(bs_embed * self.n_iter, seq_len, -1)

        # if do_classifier_free_guidance:
        #     seq_len = negative_prompt_embeds.shape[1]
        #     negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=shared.device)
        #     negative_prompt_embeds = negative_prompt_embeds.repeat(1, self.n_iter, 1)
        #     negative_prompt_embeds = negative_prompt_embeds.view(prompt_batch_size * self.n_iter, seq_len, -1)
        #     text_embeddings = torch.cat([negative_prompt_embeds, text_embeddings])
        
        # # prepare timesteps
        # self.scheduler = EulerAncestralDiscreteScheduler.from_config(self.scheduler.config)
        # self.scheduler.set_timesteps(self.steps)
        # latents = latents * self.scheduler.init_noise_sigma

        # for t in tqdm(self.scheduler.timesteps):
        #     # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        #     latent_model_input = torch.cat([latents] * 2)
        #     # latent_model_input = latents

        #     latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)

        #     # predict the noise residual
        #     with torch.no_grad():
        #         noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)
        #         noise_pred = noise_pred.sample

        #     # perform guidance
        #     if do_classifier_free_guidance:
        #         noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        #         noise_pred = noise_pred_uncond + self.cfg_scale * (noise_pred_text - noise_pred_uncond)

        #     # compute the previous noisy sample x_t -> x_t-1
        #     latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        # # latents = 1 / 0.18215 * latents
        # # samples = copy.deepcopy(latents)
        # samples = latents
        # images = self.decode_latents(latents)
        # # 9. Run safety checker
        # # image, has_nsfw_concept = self.run_safety_checker(image, shared.device, prompt_embeds.dtype)
        # # 10. Convert to PIL
        # def numpy_to_pil(images):
        #     """
        #     Convert a numpy image or a batch of images to a PIL image.
        #     """
        #     if images.ndim == 3:
        #         images = images[None, ...]
        #     images = (images * 255).round().astype("uint8")
        #     if images.shape[-1] == 1:
        #         # special case for grayscale (single channel) images
        #         pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        #     else:
        #         pil_images = [Image.fromarray(image) for image in images]
        #     return pil_images
        # # images = numpy_to_pil(images)
        # # images[0].save("test.png")


        if not self.enable_hr:
            return samples

        self.is_hr_pass = True

        target_width = self.hr_upscale_to_x
        target_height = self.hr_upscale_to_y

        def save_intermediate(image, index):
            """saves image before applying hires fix, if enabled in options; takes as an argument either an image or batch with latent space images"""

            if not opts.save or self.do_not_save_samples or not opts.save_images_before_highres_fix:
                return

            if not isinstance(image, Image.Image):
                image = sd_samplers.sample_to_image(image, index, approximation=0)

            info = create_infotext(self, self.all_prompts, self.all_seeds, self.all_subseeds, [], iteration=self.iteration, position_in_batch=index)
            images.save_image(image, self.outpath_samples, "", seeds[index], prompts[index], opts.samples_format, info=info, p=self, suffix="-before-highres-fix")

        if latent_scale_mode is not None:
            for i in range(samples.shape[0]):
                save_intermediate(samples, i)

            samples = torch.nn.functional.interpolate(samples, size=(target_height // opt_f, target_width // opt_f), mode=latent_scale_mode["mode"], antialias=latent_scale_mode["antialias"])

            # Avoid making the inpainting conditioning unless necessary as
            # this does need some extra compute to decode / encode the image again.
            if getattr(self, "inpainting_mask_weight", shared.opts.inpainting_mask_weight) < 1.0:
                image_conditioning = self.img2img_image_conditioning(decode_first_stage(self.sd_model, samples), samples)
            else:
                image_conditioning = self.txt2img_image_conditioning(samples)
        else:
            decoded_samples = decode_first_stage(self.sd_model, samples)
            lowres_samples = torch.clamp((decoded_samples + 1.0) / 2.0, min=0.0, max=1.0)

            batch_images = []
            for i, x_sample in enumerate(lowres_samples):
                x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
                x_sample = x_sample.astype(np.uint8)
                image = Image.fromarray(x_sample)

                save_intermediate(image, i)

                image = images.resize_image(0, image, target_width, target_height, upscaler_name=self.hr_upscaler)
                image = np.array(image).astype(np.float32) / 255.0
                image = np.moveaxis(image, 2, 0)
                batch_images.append(image)

            decoded_samples = torch.from_numpy(np.array(batch_images))
            decoded_samples = decoded_samples.to(shared.device)
            decoded_samples = 2. * decoded_samples - 1.

            samples = self.sd_model.get_first_stage_encoding(self.sd_model.encode_first_stage(decoded_samples))

            image_conditioning = self.img2img_image_conditioning(decoded_samples, samples)

        shared.state.nextjob()

        img2img_sampler_name = self.hr_sampler_name or self.sampler_name

        if self.sampler_name in ['PLMS', 'UniPC']:  # PLMS/UniPC do not support img2img so we just silently switch to DDIM
            img2img_sampler_name = 'DDIM'

        self.sampler = sd_samplers.create_sampler(img2img_sampler_name, self.sd_model)

        samples = samples[:, :, self.truncate_y//2:samples.shape[2]-(self.truncate_y+1)//2, self.truncate_x//2:samples.shape[3]-(self.truncate_x+1)//2]

        noise = create_random_tensors(samples.shape[1:], seeds=seeds, subseeds=subseeds, subseed_strength=subseed_strength, p=self)

        # GC now before running the next img2img to prevent running out of memory
        x = None
        devices.torch_gc()

        if not self.disable_extra_networks:
            with devices.autocast():
                extra_networks.activate(self, self.hr_extra_network_data)

        with devices.autocast():
            self.calculate_hr_conds()

        sd_models.apply_token_merging(self.sd_model, self.get_token_merging_ratio(for_hr=True))

        if self.scripts is not None:
            self.scripts.before_hr(self)

        samples = self.sampler.sample_img2img(self, samples, noise, self.hr_c, self.hr_uc, steps=self.hr_second_pass_steps or self.steps, image_conditioning=image_conditioning)

        sd_models.apply_token_merging(self.sd_model, self.get_token_merging_ratio())

        self.is_hr_pass = False

        return samples

    def close(self):
        super().close()
        self.hr_c = None
        self.hr_uc = None
        if not opts.experimental_persistent_cond_cache:
            StableDiffusionPipelineTxt2Img.cached_hr_uc = [None, None]
            StableDiffusionPipelineTxt2Img.cached_hr_c = [None, None]

    def setup_prompts(self):
        super().setup_prompts()

        if not self.enable_hr:
            return

        if self.hr_prompt == '':
            self.hr_prompt = self.prompt

        if self.hr_negative_prompt == '':
            self.hr_negative_prompt = self.negative_prompt

        if type(self.hr_prompt) == list:
            self.all_hr_prompts = self.hr_prompt
        else:
            self.all_hr_prompts = self.batch_size * self.n_iter * [self.hr_prompt]

        if type(self.hr_negative_prompt) == list:
            self.all_hr_negative_prompts = self.hr_negative_prompt
        else:
            self.all_hr_negative_prompts = self.batch_size * self.n_iter * [self.hr_negative_prompt]

        self.all_hr_prompts = [shared.prompt_styles.apply_styles_to_prompt(x, self.styles) for x in self.all_hr_prompts]
        self.all_hr_negative_prompts = [shared.prompt_styles.apply_negative_styles_to_prompt(x, self.styles) for x in self.all_hr_negative_prompts]

    def calculate_hr_conds(self):
        if self.hr_c is not None:
            return

        self.hr_uc = self.get_conds_with_caching(prompt_parser.get_learned_conditioning, self.hr_negative_prompts, self.steps * self.step_multiplier, [self.cached_hr_uc, self.cached_uc], self.hr_extra_network_data)
        self.hr_c = self.get_conds_with_caching(prompt_parser.get_multicond_learned_conditioning, self.hr_prompts, self.steps * self.step_multiplier, [self.cached_hr_c, self.cached_c], self.hr_extra_network_data)

    def setup_conds(self):
        super().setup_conds()

        self.hr_uc = None
        self.hr_c = None

        if self.enable_hr:
            if shared.opts.hires_fix_use_firstpass_conds:
                self.calculate_hr_conds()

            elif lowvram.is_enabled(shared.sd_model):  # if in lowvram mode, we need to calculate conds right away, before the cond NN is unloaded
                with devices.autocast():
                    extra_networks.activate(self, self.hr_extra_network_data)

                self.calculate_hr_conds()

                with devices.autocast():
                    extra_networks.activate(self, self.extra_network_data)

    def parse_extra_network_prompts(self):
        res = super().parse_extra_network_prompts()

        if self.enable_hr:
            self.hr_prompts = self.all_hr_prompts[self.iteration * self.batch_size:(self.iteration + 1) * self.batch_size]
            self.hr_negative_prompts = self.all_hr_negative_prompts[self.iteration * self.batch_size:(self.iteration + 1) * self.batch_size]

            self.hr_prompts, self.hr_extra_network_data = extra_networks.parse_prompts(self.hr_prompts)

        return res

    
class StableDiffusionPipelineImg2Img(StableDiffusionProcessing):
    sampler = None

    def __init__(self, init_images: list = None, resize_mode: int = 0, denoising_strength: float = 0.75, image_cfg_scale: float = None, mask: Any = None, mask_blur: int = None, mask_blur_x: int = 4, mask_blur_y: int = 4, inpainting_fill: int = 0, inpaint_full_res: bool = True, inpaint_full_res_padding: int = 0, inpainting_mask_invert: int = 0, initial_noise_multiplier: float = None, **kwargs):
        super().__init__(**kwargs)

        self.init_images = init_images
        self.resize_mode: int = resize_mode
        self.denoising_strength: float = denoising_strength
        self.image_cfg_scale: float = image_cfg_scale if shared.sd_model.cond_stage_key == "edit" else None
        self.init_latent = None
        self.image_mask = mask
        self.latent_mask = None
        self.mask_for_overlay = None
        if mask_blur is not None:
            mask_blur_x = mask_blur
            mask_blur_y = mask_blur
        self.mask_blur_x = mask_blur_x
        self.mask_blur_y = mask_blur_y
        self.inpainting_fill = inpainting_fill
        self.inpaint_full_res = inpaint_full_res
        self.inpaint_full_res_padding = inpaint_full_res_padding
        self.inpainting_mask_invert = inpainting_mask_invert
        self.initial_noise_multiplier = opts.initial_noise_multiplier if initial_noise_multiplier is None else initial_noise_multiplier
        self.mask = None
        self.nmask = None
        self.image_conditioning = None
        self.tokenizer = self.sd_pipeline.tokenizer
        self.unet = self.sd_pipeline.unet
        self.vae = self.sd_pipeline.vae
        self.scheduler = self.sd_pipeline.scheduler
        self.text_encoder = self.sd_pipeline.text_encoder
        self.decode_latents = self.sd_pipeline.decode_latents
        self.generator = torch.Generator(device=shared.device)

    def init(self, all_prompts, all_seeds, all_subseeds):
        self.sampler = sd_samplers.create_sampler(self.sampler_name, self.sd_model)
        crop_region = None

        image_mask = self.image_mask

        if image_mask is not None:
            image_mask = image_mask.convert('L')

            if self.inpainting_mask_invert:
                image_mask = ImageOps.invert(image_mask)

            if self.mask_blur_x > 0:
                np_mask = np.array(image_mask)
                kernel_size = 2 * int(4 * self.mask_blur_x + 0.5) + 1
                np_mask = cv2.GaussianBlur(np_mask, (kernel_size, 1), self.mask_blur_x)
                image_mask = Image.fromarray(np_mask)

            if self.mask_blur_y > 0:
                np_mask = np.array(image_mask)
                kernel_size = 2 * int(4 * self.mask_blur_y + 0.5) + 1
                np_mask = cv2.GaussianBlur(np_mask, (1, kernel_size), self.mask_blur_y)
                image_mask = Image.fromarray(np_mask)

            if self.inpaint_full_res:
                self.mask_for_overlay = image_mask
                mask = image_mask.convert('L')
                crop_region = masking.get_crop_region(np.array(mask), self.inpaint_full_res_padding)
                crop_region = masking.expand_crop_region(crop_region, self.width, self.height, mask.width, mask.height)
                x1, y1, x2, y2 = crop_region

                mask = mask.crop(crop_region)
                image_mask = images.resize_image(2, mask, self.width, self.height)
                self.paste_to = (x1, y1, x2-x1, y2-y1)
            else:
                image_mask = images.resize_image(self.resize_mode, image_mask, self.width, self.height)
                np_mask = np.array(image_mask)
                np_mask = np.clip((np_mask.astype(np.float32)) * 2, 0, 255).astype(np.uint8)
                self.mask_for_overlay = Image.fromarray(np_mask)

            self.overlay_images = []

        latent_mask = self.latent_mask if self.latent_mask is not None else image_mask

        add_color_corrections = opts.img2img_color_correction and self.color_corrections is None
        if add_color_corrections:
            self.color_corrections = []
        imgs = []
        for img in self.init_images:

            # Save init image
            if opts.save_init_img:
                self.init_img_hash = hashlib.md5(img.tobytes()).hexdigest()
                images.save_image(img, path=opts.outdir_init_images, basename=None, forced_filename=self.init_img_hash, save_to_dirs=False)

            image = images.flatten(img, opts.img2img_background_color)

            if crop_region is None and self.resize_mode != 3:
                image = images.resize_image(self.resize_mode, image, self.width, self.height)

            if image_mask is not None:
                image_masked = Image.new('RGBa', (image.width, image.height))
                image_masked.paste(image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(self.mask_for_overlay.convert('L')))

                self.overlay_images.append(image_masked.convert('RGBA'))

            # crop_region is not None if we are doing inpaint full res
            if crop_region is not None:
                image = image.crop(crop_region)
                image = images.resize_image(2, image, self.width, self.height)

            if image_mask is not None:
                if self.inpainting_fill != 1:
                    image = masking.fill(image, latent_mask)

            if add_color_corrections:
                self.color_corrections.append(setup_color_correction(image))

            image = np.array(image).astype(np.float32) / 255.0
            image = np.moveaxis(image, 2, 0)

            imgs.append(image)

        if len(imgs) == 1:
            batch_images = np.expand_dims(imgs[0], axis=0).repeat(self.batch_size, axis=0)
            if self.overlay_images is not None:
                self.overlay_images = self.overlay_images * self.batch_size

            if self.color_corrections is not None and len(self.color_corrections) == 1:
                self.color_corrections = self.color_corrections * self.batch_size

        elif len(imgs) <= self.batch_size:
            self.batch_size = len(imgs)
            batch_images = np.array(imgs)
        else:
            raise RuntimeError(f"bad number of images passed: {len(imgs)}; expecting {self.batch_size} or less")

        image = torch.from_numpy(batch_images)
        image = 2. * image - 1.
        image = image.to(shared.device, dtype=devices.dtype_vae)
        
        self.init_latent = self.vae.encode(image)
        self.init_latent = self.init_latent.latent_dist.sample(self.generator)
        self.init_latent = self.vae.config.scaling_factor * self.init_latent
        
        #old_latent = self.sd_model.get_first_stage_encoding(self.sd_model.encode_first_stage(image))
        
        if self.resize_mode == 3:
            self.init_latent = torch.nn.functional.interpolate(self.init_latent, size=(self.height // opt_f, self.width // opt_f), mode="bilinear")

        if image_mask is not None:
            init_mask = latent_mask
            latmask = init_mask.convert('RGB').resize((self.init_latent.shape[3], self.init_latent.shape[2]))
            latmask = np.moveaxis(np.array(latmask, dtype=np.float32), 2, 0) / 255
            latmask = latmask[0]
            latmask = np.around(latmask)
            latmask = np.tile(latmask[None], (4, 1, 1))

            self.mask = torch.asarray(1.0 - latmask).to(shared.device).type(self.sd_model.dtype)
            self.nmask = torch.asarray(latmask).to(shared.device).type(self.sd_model.dtype)

            # this needs to be fixed to be done in sample() using actual seeds for batches
            if self.inpainting_fill == 2:
                self.init_latent = self.init_latent * self.mask + create_random_tensors(self.init_latent.shape[1:], all_seeds[0:self.init_latent.shape[0]]) * self.nmask
            elif self.inpainting_fill == 3:
                self.init_latent = self.init_latent * self.mask

        self.image_conditioning = self.img2img_image_conditioning(image, self.init_latent, image_mask)
    
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep - 1, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start

    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
        
        # update sampler
        sd_pipeline = sd_samplers.update_sampler(self.sampler_name, self.sd_pipeline, self.pipeline_name)

        # common parameters for sd
        prompt = self.prompt 
        height = self.width
        width = self.height
        num_inference_steps = self.steps
        guidance_scale = self.cfg_scale
        negative_prompt = self.negative_prompt or ""
        num_images_per_prompt = self.batch_size
        eta = self.eta
        generator = self.generator
        prompt_embeds = self.prompt_embeds
        negative_prompt_embeds = self.negative_prompt_embeds
        output_type = self.output_type
        callback = self.callback
        callback_steps = self.callback_steps
        cross_attention_kwargs = self.cross_attention_kwargs

        # parameters for sdxl
        prompt_2 = self.prompt_2 or self.prompt
        negative_prompt_2 = self.negative_prompt_2 or self.negative_prompt
        denoising_end = self.denoising_end
        pooled_prompt_embeds = self.pooled_prompt_embeds
        negative_pooled_prompt_embeds = self.negative_pooled_prompt_embeds
        guidance_rescale = self.guidance_rescale
        original_size = self.original_size
        crops_coords_top_left = self.crops_coords_top_left
        target_size = self.target_size
        use_refiner = self.use_refiner

        # parameters for refiner
        strength = self.strength
        denoising_start = self.denoising_start
        aesthetic_score = self.aesthetic_score
        negative_aesthetic_score = self.negative_aesthetic_score

        noise = create_random_tensors([opt_C, self.height // opt_f, self.width // opt_f], seeds=seeds, subseeds=subseeds, subseed_strength=self.subseed_strength, seed_resize_from_h=self.seed_resize_from_h, seed_resize_from_w=self.seed_resize_from_w, p=self)

        if self.initial_noise_multiplier != 1.0:
            self.extra_generation_params["Noise multiplier"] = self.initial_noise_multiplier
            noise *= self.initial_noise_multiplier
        

        pipeline_name = self.pipeline_name
        # default output: latents
        if pipeline_name == 'StableDiffusionPipeline':
            generator = generator.manual_seed(seed=seeds)
            if self.image_mask is None:
                images = sd_pipeline(
                    prompt = prompt,
                    image = self.init_latent,
                    strength = self.denoising_strength,
                    num_inference_steps = num_inference_steps,
                    guidance_scale = guidance_scale,
                    negative_prompt = negative_prompt,
                    num_images_per_prompt= num_images_per_prompt,
                    eta = eta,
                    generator = generator,
                    prompt_embeds = prompt_embeds,
                    negative_prompt_embeds= negative_prompt_embeds,
                    output_type = output_type,
                    return_dict = True,
                    callback = callback,
                    callback_steps = callback_steps,
                    cross_attention_kwargs = cross_attention_kwargs).images[0]
            else:
                generator = generator.manual_seed(seed=seeds)
                images = sd_pipeline(
                    prompt = prompt,
                    image = self.init_images,
                    mask_image = self.image_mask,
                    height = height,
                    width = width,
                    strength = self.denoising_strength,
                    num_inference_steps = num_inference_steps,
                    guidance_scale = guidance_scale,
                    negative_prompt = negative_prompt,
                    num_images_per_prompt= num_images_per_prompt,
                    eta = eta,
                    generator = generator,
                    latents = None,
                    prompt_embeds = prompt_embeds,
                    negative_prompt_embeds= negative_prompt_embeds,
                    output_type = output_type,
                    return_dict = True,
                    callback = callback,
                    callback_steps = callback_steps,
                    cross_attention_kwargs = cross_attention_kwargs).images[0]
        elif pipeline_name == 'StableDiffusionXLPipeline':
            ### image = self.init_latent + noise -> need set denoising_start is not none
            generator = generator.manual_seed(seed=seeds)
            if self.image_mask is None:
                images = sd_pipeline(
                    prompt = prompt,
                    prompt_2 = prompt_2,
                    image = self.init_latent,
                    strength = self.denoising_strength,
                    num_inference_steps = num_inference_steps,
                    denoising_start = denoising_start,
                    denoising_end = denoising_end,
                    guidance_scale = guidance_scale,
                    negative_prompt = negative_prompt,
                    negative_prompt_2 = negative_prompt_2,
                    num_images_per_prompt = num_images_per_prompt,
                    eta = eta,
                    generator = generator,
                    latents = None,
                    prompt_embeds = prompt_embeds,
                    negative_prompt_embeds = negative_prompt_embeds,
                    pooled_prompt_embeds = pooled_prompt_embeds,
                    negative_pooled_prompt_embeds = negative_pooled_prompt_embeds,
                    output_type = output_type,
                    return_dict = True,
                    callback = callback,
                    callback_steps = callback_steps,
                    cross_attention_kwargs = cross_attention_kwargs,
                    guidance_rescale = guidance_rescale,
                    original_size = original_size,
                    crops_coords_top_left = crops_coords_top_left,
                    target_size = target_size,
                    aesthetic_score = aesthetic_score,
                    negative_aesthetic_score = negative_aesthetic_score).images[0],
            else:
                images = sd_pipeline(
                    prompt = prompt,
                    prompt_2 = prompt_2,
                    image = self.init_latent,
                    mask_image = self.image_mask,
                    height = height,
                    width = width,
                    strength = self.denoising_strength,
                    num_inference_steps = num_inference_steps,
                    denoising_start = denoising_start,
                    denoising_end = denoising_end,
                    guidance_scale = guidance_scale,
                    negative_prompt = negative_prompt,
                    negative_prompt_2 = negative_prompt_2,
                    num_images_per_prompt = num_images_per_prompt,
                    eta = eta,
                    generator = generator,
                    latents = None,
                    prompt_embeds = prompt_embeds,
                    negative_prompt_embeds = negative_prompt_embeds,
                    pooled_prompt_embeds = pooled_prompt_embeds,
                    negative_pooled_prompt_embeds = negative_pooled_prompt_embeds,
                    output_type = output_type,
                    return_dict = True,
                    callback = callback,
                    callback_steps = callback_steps,
                    cross_attention_kwargs = cross_attention_kwargs,
                    guidance_rescale = guidance_rescale,
                    original_size = original_size,
                    crops_coords_top_left = crops_coords_top_left,
                    target_size = target_size,
                    aesthetic_score = aesthetic_score,
                    negative_aesthetic_score = negative_aesthetic_score).images[0],

            
        samples = images[None,:,:,:]

        #samples = self.sampler.sample_img2img(self, self.init_latent, x, conditioning, unconditional_conditioning, image_conditioning=self.image_conditioning)

        # # diffuser pipeline
        # noise = x 
        
        # # text embedding
        # text_input = self.tokenizer(
        #     self.prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt"
        # )
        # with torch.no_grad():
        #     text_embeddings = self.text_encoder(text_input.input_ids.to(shared.device))[0]
        
        # do_classifier_free_guidance = self.cfg_scale > 1.0
        # # get unconditional embeddings for classifier free guidance
        # if do_classifier_free_guidance:
        #     max_length = text_input.input_ids.shape[-1]
        #     uncond_input = self.tokenizer(
        #         [""] * self.batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        #     )
        #     uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(shared.device))[0]

        #     # For classifier free guidance, we need to do two forward passes.
        #     # Here we concatenate the unconditional and text embeddings into a single batch
        #     # to avoid doing two forward passes
        #     text_embeddings = torch.cat([uncond_embeddings, text_embeddings])


        # # set timesteps
        # self.scheduler = EulerAncestralDiscreteScheduler.from_config(self.scheduler.config)
        # self.scheduler.set_timesteps(self.steps, device=shared.device)
        # timesteps, num_inference_steps = self.get_timesteps(self.steps, self.denoising_strength, shared.device)
        # latent_timestep = timesteps[:1].repeat(self.batch_size * self.n_iter)
        
        # init_latents = self.scheduler.add_noise(self.init_latent, noise, latent_timestep)
        # latents = init_latents

        # num_channels_unet = self.unet.config.in_channels
        
        # for i, t in tqdm(enumerate(timesteps)):
        #     # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        #     latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        #     #latent_model_input = torch.cat([latents] * 2)

        #     latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)

        #     if self.image_mask is not None and num_channels_unet == 9:
        #         mask_image_conditioning = torch.cat([self.image_conditioning] * 2) if do_classifier_free_guidance else self.image_conditioning
        #         latent_model_input = torch.cat([latent_model_input, mask_image_conditioning], dim=1)


        #     # predict the noise residual
        #     with torch.no_grad():
        #         noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)[0]

        #     # perform guidance
        #     if do_classifier_free_guidance:
        #         noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        #         noise_pred = noise_pred_uncond + self.cfg_scale * (noise_pred_text - noise_pred_uncond)
        #     #noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        #     #noise_pred = noise_pred_uncond + self.cfg_scale * (noise_pred_text - noise_pred_uncond)

        #     # compute the previous noisy sample x_t -> x_t-1
        #     latents = self.scheduler.step(noise_pred, t, latents)[0]

        #     if self.image_mask is not None and num_channels_unet == 4:
        #             init_latents_proper = self.init_latent
        #             init_mask = self.mask

        #             if i < len(timesteps) - 1:
        #                 noise_timestep = timesteps[i + 1]
        #                 init_latents_proper = self.scheduler.add_noise(
        #                     init_latents_proper, noise, torch.tensor([noise_timestep])
        #                 )

        #             latents = (1 - init_mask) * init_latents_proper + init_mask * latents

        # samples = latents
        
        ##### debug for show the results
        # images = pipeline.decode_latents(latents)
        # # 9. Run safety checker
        # # image, has_nsfw_concept = self.run_safety_checker(image, shared.device, prompt_embeds.dtype)
        # # 10. Convert to PIL
        # def numpy_to_pil(images):
        #     """
        #     Convert a numpy image or a batch of images to a PIL image.
        #     """
        #     if images.ndim == 3:
        #         images = images[None, ...]
        #     images = (images * 255).round().astype("uint8")
        #     if images.shape[-1] == 1:
        #         # special case for grayscale (single channel) images
        #         pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        #     else:
        #         pil_images = [Image.fromarray(image) for image in images]
        #     return pil_images
        # images = numpy_to_pil(images)
        # images[0].save("test_img2img_diffuser.png")
        ############################################


        del x
        devices.torch_gc()

        return samples

    def get_token_merging_ratio(self, for_hr=False):
        return self.token_merging_ratio or ("token_merging_ratio" in self.override_settings and opts.token_merging_ratio) or opts.token_merging_ratio_img2img or opts.token_merging_ratio