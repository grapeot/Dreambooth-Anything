import argparse
from PIL import Image
import torch
from accelerate.utils import set_seed
from diffusers import StableDiffusionDepth2ImgPipeline, DiffusionPipeline, DDPMScheduler, DDIMScheduler, EulerAncestralDiscreteScheduler
from glob import glob
from tqdm import tqdm
from os import mkdir
from os.path import join, basename, exists
from sys import exit, stderr
from random import random

def predict(model: StableDiffusionDepth2ImgPipeline,
    imgfn: str,
    outfn: str,
    prompt: str,
    generator: torch.Generator = None,
    seed: int = 0):
    """
    Use the model to make one prediction
    """
    set_seed(seed)
    if generator is not None:
        generator.manual_seed(seed)
    init_image = Image.open(imgfn).convert("RGB")
    init_image = init_image.resize((640, 360))
    images = model(prompt=prompt,
        image=init_image,
        strength=0.4,
        guidance_scale=8,
        negative_prompt="disformed, extra limb, extra fingers",
        num_inference_steps=100).images
    images[0].save(outfn)

def loadCheckpoint(ckptDir: str):
    model_id = "stabilityai/stable-diffusion-2-depth"
    pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    state_dict = torch.load(join(ckptDir, 'pytorch_model.bin'))
    pipeline.unet.load_state_dict(state_dict)
    text_encoder_path = join(ckptDir, 'pytorch_model_1.bin') 
    if exists(text_encoder_path):
        state_dict = torch.load(text_encoder_path)
        pipeline.text_encoder.load_state_dict(state_dict)
    return pipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load a given checkpoint or model, and make predictions')
    parser.add_argument('--checkpoint', help='Path to the intermediate training checkpoint.', default=None)
    parser.add_argument('--model', help='Path to the final diffuser model.', default=None)
    parser.add_argument('--startFrame', help='Number of the start frame.', default=-1, type=int)
    parser.add_argument('--endFrame', help='Number of the end frame.', default=-1, type=int)
    parser.add_argument('--inputDir', help='Path to the input dir storing the input frames', required=True)
    parser.add_argument('--outputDir', help='Path to the output dir to store the generated frames', required=True)
    parser.add_argument('--prompt', help='Prompt used to generate the image', required=True)
    parser.add_argument('--seed', help='Seed to used generate each frame. The default value -1 will randomly generate one, and use it for all the frames. -2 will try out 50 seeds for seed tuning.', type=int, default=-1)
    args = parser.parse_args()
    
    if args.checkpoint is not None and args.model is not None:
        stderr.write('Error: have to specify either a checkpoint or a model, not both of them.\n')
        exit(1)
    if args.checkpoint is None and args.model is None:
        stderr.write('Error: have to specify either a checkpoint or a model.\n')
        exit(2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.checkpoint is not None:
        model = loadCheckpoint(args.checkpoint).to(device)
    if args.model is not None:
        model = StableDiffusionDepth2ImgPipeline.from_pretrained(
            args.model,
            torch_dtype=torch.float16).to(device)
    fns = glob(join(args.inputDir, '*.*'))
    fns = sorted(fns)
    startFrame = 0 if args.startFrame == -1 else args.startFrame
    endFrame = len(fns) if args.endFrame == -1 else args.endFrame
    if args.seed == -1:
        seed = int(random() * 99999999)
    else:
        seed = args.seed

    if not exists(args.outputDir):
        mkdir(args.outputDir)

    generator = torch.Generator(device=device)
    for fn in tqdm(fns[startFrame: endFrame]):
        if seed == -2:
            for seed in range(100):
                newfn = join(args.outputDir, f'{basename(fn)}_{seed}.jpg')
                predict(model, fn, newfn, args.prompt, generator, seed)
        else:
            newfn = join(args.outputDir, basename(fn))
            predict(model, fn, newfn, args.prompt, generator, seed)