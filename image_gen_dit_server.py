import os
import time
import torch
import ray
import io
import logging
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import argparse

from xfuser import (
    xFuserArgs,
)
# from xfuser.model_executor.models.transformers.transformer_z_image import xFuserZImageTransformer2DWrapper
from diffusers import DiffusionPipeline
from xfuser.core.distributed import (
    get_world_group,
    #get_runtime_state,
    initialize_runtime_state,
    #is_dp_last_group,
)
from bailingmm_utils import process_ratio
from PIL import Image
import base64
import pickle
import numpy as np
from diffusers import AutoencoderKL

def dict_to_base64(obj) -> str:
    def encode(x):
        if isinstance(x, torch.Tensor):
            t = x.detach().cpu().contiguous()
            arr = t.numpy()
            return {
                "__type__": "torch_tensor",
                "dtype": str(arr.dtype),
                "shape": arr.shape,
                "data": arr.tobytes(),  # raw bytes
            }
        elif isinstance(x, dict):
            return {k: encode(v) for k, v in x.items()}
        elif isinstance(x, (list, tuple)):
            y = [encode(v) for v in x]
            return {"__type__": "tuple", "items": y} if isinstance(x, tuple) else y
        else:
            return x

    packed = encode(obj)
    blob = pickle.dumps(packed, protocol=pickle.HIGHEST_PROTOCOL)
    return base64.b64encode(blob).decode("utf-8")


def base64_to_dict(s: str):
    def decode(x):
        if isinstance(x, dict) and x.get("__type__") == "torch_tensor":
            arr = np.frombuffer(x["data"], dtype=np.dtype(x["dtype"])).reshape(x["shape"])
            return torch.from_numpy(arr)
        elif isinstance(x, dict) and x.get("__type__") == "tuple":
            return tuple(decode(v) for v in x["items"])
        elif isinstance(x, dict):
            return {k: decode(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [decode(v) for v in x]
        else:
            return x

    blob = base64.b64decode(s.encode("utf-8"))
    packed = pickle.loads(blob)
    return decode(packed)


def run_pipe(pipe: DiffusionPipeline, gen_param_b64, logger): #prompt, steps, seed):
    # Pipe implementation currently encodes the prompt in-place,
    # causing any subsequent calls to use the already encoded prompt as prompt,
    # causing cascading encodings unless we provide a new list each time.
    #prompt = str(input_config.prompt)

    print("run_pipe")

    #is_last_process =  get_world_group().rank == get_world_group().world_size - 1
    # if is_last_process:
    #     import datetime
    #     ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # 不含空格/冒号
    #     filename = os.path.join("personal_gen_param_b64", f"{ts}.txt")
    #     with open(filename, "a", encoding="utf-8") as f:
    #         f.write(f"{gen_param_b64}\n")

    gen_param = base64_to_dict(gen_param_b64)
    #"gen_param_b64": dict_to_base64(task),
    condition_embeds = gen_param["image_gen_condition_embeds"].to(pipe.transformer.device)
    negative_condition_embeds = gen_param["image_gen_negative_condition_embeds"].to(pipe.transformer.device)
    image_gen_pixel_values_reference = gen_param["image_gen_pixel_values_reference"] if "image_gen_pixel_values_reference" in gen_param else None
    image_gen_seed = gen_param["image_gen_seed"] if "image_gen_seed" in gen_param else None
    image_gen_cfg = gen_param["image_gen_cfg"] if "image_gen_cfg" in gen_param else 2.0

    image_gen_height = gen_param["image_gen_height"]
    image_gen_width = gen_param["image_gen_width"]
    image_gen_highres = gen_param["image_gen_highres"]
    if image_gen_height is None or image_gen_width is None:
        if isinstance(image_gen_highres, int):
            image_gen_height, image_gen_width = [image_gen_highres] * condition_embeds.shape[0], [image_gen_highres] * condition_embeds.shape[0]
        elif image_gen_highres is True:
            image_gen_height, image_gen_width = [1024] * condition_embeds.shape[0], [1024] * condition_embeds.shape[0]
        else:
            image_gen_height, image_gen_width = [512] * condition_embeds.shape[0], [512] * condition_embeds.shape[0]
    elif isinstance(image_gen_height, torch.Tensor) or isinstance(image_gen_width, torch.Tensor):
        assert isinstance(image_gen_height, torch.Tensor), image_gen_height
        assert isinstance(image_gen_width, torch.Tensor), image_gen_width
        image_gen_height = image_gen_height.cpu().tolist()
        image_gen_width = image_gen_width.cpu().tolist()
        assert len(image_gen_height) == condition_embeds.shape[0]
        assert len(image_gen_width)  == condition_embeds.shape[0]
    elif isinstance(image_gen_height, int) or isinstance(image_gen_width, int):
        assert isinstance(image_gen_height, int), image_gen_height
        assert isinstance(image_gen_width, int), image_gen_width
        image_gen_height = [image_gen_height] * condition_embeds.shape[0]
        image_gen_width = [image_gen_width] * condition_embeds.shape[0]
    else:
        assert isinstance(image_gen_height, list), image_gen_height
        assert isinstance(image_gen_width, list), image_gen_width
        assert len(image_gen_height) == condition_embeds.shape[0]
        assert len(image_gen_width)  == condition_embeds.shape[0]


    image_gen_height_diffusion_list = []
    image_gen_width_diffusion_list = []
    image_gen_output_resize_height = []
    image_gen_output_resize_width = []
    for height, width in zip(image_gen_height, image_gen_width):
        closest_size, resize_size = process_ratio(ori_h=height, ori_w=width, highres=image_gen_highres)
        height, width = closest_size
        image_gen_height_diffusion_list.append(height)
        image_gen_width_diffusion_list.append(width)
        height, width = resize_size
        image_gen_output_resize_height.append(height)
        image_gen_output_resize_width.append(width)

    image_gen_height = image_gen_height_diffusion_list[0]
    assert all([i == image_gen_height for i in image_gen_height_diffusion_list])
    image_gen_width = image_gen_width_diffusion_list[0]
    assert all([i == image_gen_width for i in image_gen_width_diffusion_list])

    if image_gen_pixel_values_reference is not None:
        assert (image_gen_height, image_gen_width) == (image_gen_pixel_values_reference.shape[-2], image_gen_pixel_values_reference.shape[-1])

    if image_gen_seed is None or image_gen_seed < 0:
        from datetime import datetime
        image_gen_seed = datetime.now().microsecond % 1000

    logger.info(f"condition_embeds.shape {condition_embeds.shape}")
    logger.info(f"negative_condition_embeds.shape {negative_condition_embeds.shape}")
    logger.info(f"height {image_gen_height}")
    logger.info(f"height {image_gen_width}")
    logger.info(f"guidance_scale {image_gen_cfg}")
    logger.info(f"seed {image_gen_seed}")

    image = pipe(
        prompt_embeds=list(condition_embeds.unbind(0)),
        negative_prompt_embeds=list(negative_condition_embeds.unbind(0)),
        height=image_gen_height,
        width=image_gen_width,
        num_inference_steps=30, # Recommended value
        guidance_scale=image_gen_cfg, # Recommended value
        generator=torch.manual_seed(image_gen_seed),
        max_sequence_length=512,
        ref_hidden_states=image_gen_pixel_values_reference,
    ).images

    image = [i.resize((w, h)) for i, w, h in zip(image, image_gen_output_resize_width, image_gen_output_resize_height)]

    return image

    # prompt_embeds=encoder_hidden_states,
    #         negative_prompt_embeds=[en*0 for en in encoder_hidden_states],
    #         guidance_scale=cfg,
    #         #image_guidance_scale=image_cfg,
    #         #guidance_scale_mode=cfg_mode,
    #         generator=torch.manual_seed(seed),
    #         num_inference_steps=steps,
    #         height=height,
    #         width=width,
    #         max_sequence_length=512,
    #         device=self.device,
    #         #extra_vit_input=extra_vit_input,
    #         ref_hidden_states=ref_x,
    #         #use_dynamic_shifting=use_dynamic_shifting


# Define request model
class GenerateRequest(BaseModel):
    # prompt: str
    # num_inference_steps: Optional[int] = 50
    # seed: Optional[int] = 42
    # cfg: Optional[float] = 7.5
    # save_disk_path: Optional[str] = None
    # height: Optional[int] = 1024
    # width: Optional[int] = 1024
    gen_param_b64: str

    # # Add input validation
    # class Config:
    #     json_schema_extra = {
    #         "example": {
    #             "prompt": "a beautiful landscape",
    #             "num_inference_steps": 50,
    #             "seed": 42,
    #             "cfg": 7.5,
    #             "height": 1024,
    #             "width": 1024
    #         }
    #     }

app = FastAPI()

@ray.remote(num_gpus=1)
class ImageGenerator:
    def __init__(self, xfuser_args: xFuserArgs, rank: int, world_size: int, use_taylor_cache=False):
        # Set PyTorch distributed environment variables
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        
        self.rank = rank
        self.setup_logger()
        self.initialize_model(xfuser_args, use_taylor_cache=use_taylor_cache)

    def setup_logger(self):
        self.logger = logging.getLogger(__name__)
        # Add console handler if not already present
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            self.logger.setLevel(logging.INFO)

    def initialize_model(self, xfuser_args : xFuserArgs, use_taylor_cache=False):

        # init distributed environment in create_config
        self.engine_config, self.input_config = xfuser_args.create_config()
        print(self.engine_config)

        local_rank = get_world_group().local_rank

        print("self.engine_config.model_config.model", self.engine_config.model_config.model)
        model_name_or_path = self.engine_config.model_config.model

        import sys
        sys.path.insert(0, model_name_or_path)

        from diffusion.transformer_z_image_xfuser import xFuserZImageTransformer2DWrapper
        from diffusion.pipeline_z_image import ZImagePipeline
        from diffusers import FlowMatchEulerDiscreteScheduler

        
        #"/nativemm/share/cpfs/weilong.cwl/checkpoints/flash_v2_xpo_final_20260205_hf_metax_ais16893664"
        #"/nativemm/share/cpfs/weilong.cwl/checkpoints/bailing_native_moe_ming_flash_v2.0_xpo_final_20260205_vllm_new"

        zimage_model_path = os.path.join(model_name_or_path, "pipeline")
        #zimage_model_path = "/nativemm/share/cpfs/weilong.cwl/checkpoints/Z-Image-Turbo"
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_name_or_path, subfolder="scheduler")
        noise_scheduler.config['use_dynamic_shifting'] = True


        transformer = xFuserZImageTransformer2DWrapper.from_pretrained(
            model_name_or_path, subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )

        vae = AutoencoderKL.from_pretrained(
            model_name_or_path,
            subfolder="vae",
            torch_dtype=torch.bfloat16,
        )

        self.pipe = ZImagePipeline.from_pretrained(
            zimage_model_path,
            transformer=transformer,  
            text_encoder=None, 
            tokenizer=None,                          
            scheduler=noise_scheduler,
            vae=vae,
        )
        if use_taylor_cache:
            print("use_taylor_cache")
            self.pipe.set_taylor_cache()

        # from modeling_bailingmm2 import BailingMM2NativeForConditionalGeneration
        # import time
        # model = BailingMM2NativeForConditionalGeneration.from_pretrained(
        #     self.engine_config.model_config.model,
        #     torch_dtype=torch.bfloat16,
        #     attn_implementation="flash_attention_2",
        #     load_image_gen=True,
        #     load_image_gen_others=False,
        #     load_vlm=False,
        #     device_map=local_rank,
        #     image_gen_seq_parallel=True,
        # ).to(dtype=torch.bfloat16)
        # model.eval()
        # model.diffusion_loss.pipelines.transformer.config.num_attention_heads = model.diffusion_loss.pipelines.transformer.config.n_heads
        # model.diffusion_loss.pipelines.transformer.config.patch_size = model.diffusion_loss.pipelines.transformer.config.all_patch_size
        # model.diffusion_loss.pipelines.transformer.config.attention_head_dim = model.diffusion_loss.pipelines.transformer.config.axes_dims[-1]


        # #print(pipeline.transformer.config)
        # self.pipe = model.diffusion_loss.pipelines

        
        #is_last_process =  get_world_group().rank == get_world_group().world_size - 1

        # transformer = xFuserZImageTransformer2DWrapper.from_pretrained(
        #     self.engine_config.model_config.model,
        #     torch_dtype=torch.bfloat16,
        #     subfolder="transformer",
        # )
        # self.pipe = ZImagePipeline.from_pretrained(
        #     pretrained_model_name_or_path=self.engine_config.model_config.model,
        #     engine_config=self.engine_config,
        #     transformer=transformer,
        #     torch_dtype=torch.bfloat16,
        # )
        
        # self.pipe = self.pipe.to(f"cuda:{local_rank}")
        #parameter_peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

        self.pipe.transformer.config.num_attention_heads = self.pipe.transformer.config.n_heads
        self.pipe.transformer.config.patch_size = self.pipe.transformer.config.all_patch_size
        self.pipe.transformer.config.attention_head_dim = self.pipe.transformer.config.axes_dims[-1]

        print(self.pipe._execution_device)
        local_rank = get_world_group().local_rank
        self.pipe = self.pipe.to(f"cuda:{local_rank}")
        print(self.pipe._execution_device)

        initialize_runtime_state(self.pipe, self.engine_config)
        
        # model_name = self.engine_config.model_config.model.split("/")[-1]
        # pipeline_map = {
        #     "PixArt-XL-2-1024-MS": xFuserPixArtAlphaPipeline,
        #     "PixArt-Sigma-XL-2-2K-MS": xFuserPixArtSigmaPipeline,
        #     "stable-diffusion-3-medium-diffusers": xFuserStableDiffusion3Pipeline,
        #     "stabilityai__stable-diffusion-3-medium-diffusers": xFuserStableDiffusion3Pipeline,
        #     "HunyuanDiT-v1.2-Diffusers": xFuserHunyuanDiTPipeline,
        #     "FLUX.1-schnell": xFuserFluxPipeline,
        #     "FLUX.1-dev": xFuserFluxPipeline,
        # }
        
        # PipelineClass = pipeline_map.get(model_name)
        # if PipelineClass is None:
        #     raise NotImplementedError(f"{model_name} is currently not supported!")

        # self.logger.info(f"Initializing model {model_name} from {xfuser_args.model}")

        # self.pipe = PipelineClass.from_pretrained(
        #     pretrained_model_name_or_path=xfuser_args.model,
        #     engine_config=self.engine_config,
        #     torch_dtype=torch.float16,
        # ).to("cuda")
        
        # self.pipe.prepare_run(self.input_config)
        self.logger.info("Model initialization completed")

    def generate(self, request: GenerateRequest):
        # try:
        #     start_time = time.time()
        #     print("generate", len(request.gen_param_b64))
        #     output = run_pipe(self.pipe, request.gen_param_b64)
        #     #, request.num_inference_steps, request.seed)

        #     # output = self.pipe(
        #     #     height=request.height,
        #     #     width=request.width,
        #     #     prompt=request.prompt,
        #     #     num_inference_steps=request.num_inference_steps,
        #     #     output_type="pil",
        #     #     generator=torch.Generator(device="cuda").manual_seed(request.seed),
        #     #     guidance_scale=request.cfg,
        #     #     max_sequence_length=self.input_config.max_sequence_length
        #     # )
        #     elapsed_time = time.time() - start_time

        #     is_last_process =  get_world_group().rank == get_world_group().world_size - 1

        #     #if self.pipe.is_dp_last_group():
        #     if is_last_process:
        #         # if request.save_disk_path:
        #         #     timestamp = time.strftime("%Y%m%d-%H%M%S")
        #         #     filename = f"generated_image_{timestamp}.png"
        #         #     file_path = os.path.join(request.save_disk_path, filename)
        #         #     os.makedirs(request.save_disk_path, exist_ok=True)
        #         #     output[0].save(file_path)
        #         #     return {
        #         #         "message": "Image generated successfully",
        #         #         "elapsed_time": f"{elapsed_time:.2f} sec",
        #         #         "output": file_path,
        #         #         "save_to_disk": True
        #         #     }
        #         # else:
        #         # Convert to base64
        #         buffered = io.BytesIO()
        #         output[0].save(buffered, format="PNG")
        #         img_str = base64.b64encode(buffered.getvalue()).decode()
        #         return {
        #             "message": "Image generated successfully",
        #             "elapsed_time": f"{elapsed_time:.2f} sec",
        #             "output": img_str,
        #             "save_to_disk": False
        #         }
        #     return None

        # except Exception as e:
        #     self.logger.error(f"Error generating image: {str(e)}")
        #     raise HTTPException(status_code=500, detail=str(e))

        start_time = time.time()
        print("generate", len(request.gen_param_b64))
        output = run_pipe(self.pipe, request.gen_param_b64, logger=self.logger)
        #, request.num_inference_steps, request.seed)

        # output = self.pipe(
        #     height=request.height,
        #     width=request.width,
        #     prompt=request.prompt,
        #     num_inference_steps=request.num_inference_steps,
        #     output_type="pil",
        #     generator=torch.Generator(device="cuda").manual_seed(request.seed),
        #     guidance_scale=request.cfg,
        #     max_sequence_length=self.input_config.max_sequence_length
        # )
        elapsed_time = time.time() - start_time

        is_last_process =  get_world_group().rank == get_world_group().world_size - 1

        #if self.pipe.is_dp_last_group():
        if is_last_process:
            # if request.save_disk_path:
            #     timestamp = time.strftime("%Y%m%d-%H%M%S")
            #     filename = f"generated_image_{timestamp}.png"
            #     file_path = os.path.join(request.save_disk_path, filename)
            #     os.makedirs(request.save_disk_path, exist_ok=True)
            #     output[0].save(file_path)
            #     return {
            #         "message": "Image generated successfully",
            #         "elapsed_time": f"{elapsed_time:.2f} sec",
            #         "output": file_path,
            #         "save_to_disk": True
            #     }
            # else:
            # Convert to base64
            buffered = io.BytesIO()
            output[0].save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return {
                "message": "Image generated successfully",
                "elapsed_time": f"{elapsed_time:.2f} sec",
                "output": img_str,
                "save_to_disk": False
            }
        return None

class Engine:
    def __init__(self, world_size: int, xfuser_args: xFuserArgs, use_taylor_cache=False):
        # Ensure Ray is initialized
        if not ray.is_initialized():
            ray.init()
        
        num_workers = world_size
        self.workers = [
            ImageGenerator.remote(xfuser_args, rank=rank, world_size=world_size, use_taylor_cache=use_taylor_cache)
            for rank in range(num_workers)
        ]
        
    async def generate(self, request: GenerateRequest):
        results = ray.get([
            worker.generate.remote(request)
            for worker in self.workers
        ])

        return next(path for path in results if path is not None) 

@app.post("/generate")
async def generate_image(request: GenerateRequest):
    try:
        # Add input validation
        # if not request.prompt:
        #     raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        # if request.height <= 0 or request.width <= 0:
        #     raise HTTPException(status_code=400, detail="Height and width must be positive")
        # if request.num_inference_steps <= 0:
        #     raise HTTPException(status_code=400, detail="num_inference_steps must be positive")
        print(len(request.gen_param_b64))
            
        result = await engine.generate(request)
        return result
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='xDiT HTTP Service')
    parser.add_argument('--model_path', type=str, help='Path to the model', required=True)
    parser.add_argument('--world_size', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--pipefusion_parallel_degree', type=int, default=1, help='Degree of pipeline fusion parallelism')
    parser.add_argument('--ulysses_parallel_degree', type=int, default=1, help='Degree of Ulysses parallelism')
    parser.add_argument('--ring_degree', type=int, default=1, help='Degree of ring parallelism')
    parser.add_argument('--save_disk_path', type=str, default='output', help='Path to save generated images')
    parser.add_argument('--use_cfg_parallel', action='store_true', help='Whether to use CFG parallel')
    parser.add_argument('--use_taylor_cache', action='store_true', help='Whether to use taylor cache')
    args = parser.parse_args()

    xfuser_args = xFuserArgs(
        model=args.model_path,
        trust_remote_code=True,
        warmup_steps=1,
        use_parallel_vae=False,
        use_torch_compile=False,
        ulysses_degree=args.ulysses_parallel_degree,
        pipefusion_parallel_degree=args.pipefusion_parallel_degree,
        use_cfg_parallel=args.use_cfg_parallel,
        dit_parallel_size=0,
    )
    
    engine = Engine(
        world_size=args.world_size,
        xfuser_args=xfuser_args,
        use_taylor_cache=args.use_taylor_cache,
    )
    
    # Start the server
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6000)
    