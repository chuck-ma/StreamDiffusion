from utils.wrapper import StreamDiffusionWrapper
import os
import sys
from typing import Literal, Dict, Optional

import fire


sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

default_negative_prompt = "black and white, blurry, low resolution, pixelated,  pixel art, low quality, low fidelity"


def main(
    output: str = os.path.join(
        CURRENT_DIR, "..", "..", "images", "outputs", "output.png"),
    model_id_or_path: str = "RunDiffusion/Juggernaut-XL-Lightning",
    lora_dict: Optional[Dict[str, float]] = None,
    prompt: str = "A woman wearing a hat poses for a picture, in the style of oshare kei, black, wide lens, shiny/ glossy, solapunk, dark silver, rim light",
    width: int = 1024,
    height: int = 512,
    acceleration: Literal["none", "xformers", "tensorrt"] = "none",
    use_denoising_batch: bool = True,
    seed: int = 11,
):
    """
    Process for generating images based on a prompt using a specified model.

    Parameters
    ----------
    output : str, optional
        The output image file to save images to.
    model_id_or_path : str
        The name of the model to use for image generation.
    lora_dict : Optional[Dict[str, float]], optional
        The lora_dict to load, by default None.
        Keys are the LoRA names and values are the LoRA scales.
        Example: {'LoRA_1' : 0.5 , 'LoRA_2' : 0.7 ,...}
    prompt : str
        The prompt to generate images from.
    width : int, optional
        The width of the image, by default 512.
    height : int, optional
        The height of the image, by default 512.
    acceleration : Literal["none", "xformers", "tensorrt"]
        The type of acceleration to use for image generation.
    use_denoising_batch : bool, optional
        Whether to use denoising batch or not, by default False.
    seed : int, optional
        The seed, by default 2. if -1, use random seed.
    """
    # model_id_or_path = "stabilityai/sdxl-turbo"

    stream = StreamDiffusionWrapper(
        model_id_or_path=model_id_or_path,
        lora_dict=lora_dict,
        t_index_list=[0],
        frame_buffer_size=1,
        width=width,
        height=height,
        warmup=10,
        use_lcm_lora=False,
        vae_id=None,
        acceleration=acceleration,
        mode="txt2img",
        use_denoising_batch=use_denoising_batch,
        cfg_type="none",
        seed=seed,
        sdxl=True,
        # do_add_noise=False,
    )
    import time
    start_time = time.time()
    print("start painting...")

    stream.prepare(
        prompt=prompt,
        negative_prompt=default_negative_prompt,
        num_inference_steps=4,
        guidance_scale=2.0,
    )

    output_image = stream()
    output_image.save(output)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Code execution time: {elapsed_time} seconds")


if __name__ == "__main__":
    fire.Fire(main)
