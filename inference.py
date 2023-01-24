import PIL
import torch
from torchvision import transforms
from accelerate.utils import set_seed
import diffusers
import transformers
from diffusers import StableDiffusionDepth2ImgPipeline
from glob import glob
from tqdm import tqdm
import os
from os.path import join, basename
from sys import argv

if __name__ == '__main__':
    frameStart = int(argv[1])
    frameEnd = int(argv[2])
    modelPath = './sd21_dpeth_Lycoris500'
    print(f'Getting model from {modelPath}.')
    pipeline = StableDiffusionDepth2ImgPipeline.from_pretrained(
        modelPath,
        torch_dtype=torch.float16).to('cuda')


#image_path = "../diffusers/examples/dreambooth/Nichijou/46127259_1_0.flv_000065.jpg" # replace with whatever you want
#image = PIL.Image.open(image_path)

#result = pipeline("an anime in LYCORISANIME style", image, strength=0.9)
#result[0][0].save('output.jpg')

    OUT_DIR = 'NichijouVideo/resultFrames'
    fns = glob("NichijouVideo/frames/*.png")
    fns = sorted(fns)
    device = "cuda"
    model_id_or_path = "./sd21_dpeth_Lycoris500"

    for fn in tqdm(fns[frameStart:frameEnd]):
        set_seed(12345)
        init_image = PIL.Image.open(fn).convert("RGB")
        init_image = init_image.resize((640, 360))
        prompt = "an anime in LYCORISANIME style"
        images = pipeline(prompt=prompt, image=init_image, strength=0.6, guidance_scale=7, negative_prompt="disformed, extra limb, extra fingers", num_inference_steps=50).images
        newfn = join(OUT_DIR, basename(fn))
        images[0].save(newfn)
