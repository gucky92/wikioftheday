import os
from pathlib import Path
import json
from typing import Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import moviepy.editor as mpy
import pandas as pd
import numpy as np
import requests
import scipy
import torch
import tqdm
import wikipediaapi as wiki
from bs4 import BeautifulSoup
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoProcessor
from diffusers.utils import export_to_video
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

folderpath = Path(__file__).parent / "results"
folderpath.mkdir(exist_ok=True, parents=True)
# device = "mps"
device = "cuda"

api = wiki.Wikipedia("autowiki/0.1 (tomatoplay@protonmail.com)", "en")
openai_api_key = ""
os.environ["OPENAI_API_KEY"] = openai_api_key

llm = OpenAI(temperature=0, model_name="gpt-4", openai_api_key=openai_api_key)
llm_chat = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=openai_api_key)

class YoutubeScriptSnippet(BaseModel):
    narrator: Optional[str] = Field(description="The narrator text for the snippet.")
    image: Optional[str] = Field(description="The image text for the snippet.")
    video: Optional[str] = Field(description="The video text for the snippet.")


class YoutubeScript(BaseModel):
    snippets: List[YoutubeScriptSnippet]
    youtube_title: str = Field(description="The title of the youtube video.")
    youtube_description: str = Field(
        description="The description of the youtube video."
    )


output_parser = PydanticOutputParser(pydantic_object=YoutubeScript)


narrator_voice = "David Attenborough"

wiki_prompt = PromptTemplate(
    input_variables=["title", "text"],
    template=(
        "# Task instructions for Youtube video script generation \n"
        "You are a Youtube content creator and writer who loves history and knowledge. "
        "Your job is to take the wikipedia article of the day in order to write a script for your next video. "
        "You are given the title and the text of the article. "
        "The text that you will write should be three minutes long with room for jokes, dramatic pauses, and other things that make a video interesting. "
        "Also you need to put the article into your own words and put it into the greater historical context of the time period. "
        "Make sure to keep a light tone and allow the script to be read aloud by articial transcription services. "
        "Please also indicate visual elements that you want to include in the video. "
        "The description of the visual elements should be very specific and contain the appropriate information to recreate the image (e.g. include the relevant dates, names, etc.). "
        # "The description of the visual elements should also be easily processed by a text-to-video model."
        "Separate visual elements into two categories: 'image element' and 'video element'. "
        "The image element should be processed by a text-to-image model and the video element should be processed by a text-to-video model. Both image and video prompts must be safe for work and merely descriptive. "
        "Give your narrators voice a funny and light-hearted but also accurate and knowledgable personality and give the narrator a voice that fits into the time period."
        f"The narrator should sound like {narrator_voice}. "
        "Also provide and mark a snappy and funny and attention-grabbing youtube title and youtube description."
        "# Wikipedia Data \n"
        "Here is the article of the day:"
        "\n\n"
        "##Title\n{title}"
        "\n\n"
        "##Text\n{text}"
    ),
)
format_prompt = PromptTemplate(
    input_variables=["script"],
    template=(
        "This is a Youtube script for a video about the wikipedia article of the day: "
        "\n\n"
        "{script}"
        "\n\n"
        "Please format this script into a JSON serializable format. "
        "Include the youtube title and youtube description and divide the script into script snippets. "
        "Each script snippet contains the narrator, image, and video elements. "
        "Keep the whole text for each snippet and if it is too short then add more text for the narrator to say. "
        "# More detailed format instructions \n"
        "{format_instructions}"
    ),
    partial_variables={
        "format_instructions": output_parser.get_format_instructions()
    }
)

visual_prompt = PromptTemplate(
    input_variables=["old_prompt", "visual_type"],
    template=(
        "Shorten the length of this text to a text-to-{visual_type} prompt:"
        "{old_prompt}"
    )
)

# image and video translation
pipe_image = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
)
pipe_image = pipe_image.to(device)
# Recommended if your computer has < 64 GB of RAM
pipe_image.enable_attention_slicing()
pipe_image.enable_xformers_memory_efficient_attention()

base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
base.to("cuda")
# base.enable_model_cpu_offload()
refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to("cuda")

pipe_video = DiffusionPipeline.from_pretrained(
    "damo-vilab/text-to-video-ms-1.7b",
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe_video.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe_video.scheduler.config
)
pipe_video.to(device)
pipe_video.enable_model_cpu_offload()
pipe_video.enable_attention_slicing()
pipe_video.enable_vae_slicing()
pipe_video.enable_xformers_memory_efficient_attention()


processor_audio = AutoProcessor.from_pretrained("suno/bark-small")
model_audio = AutoModel.from_pretrained("suno/bark-small")
model_audio = model_audio.to(device)


chain = LLMChain(llm=llm_chat, prompt=wiki_prompt)
tovisual_chain = LLMChain(llm=llm, prompt=visual_prompt)
format_chain = LLMChain(
    llm=llm, prompt=format_prompt, output_parser=output_parser
)


def get_article_of_the_day():
    # Get the main page HTML
    response = requests.get("https://en.wikipedia.org/wiki/Main_Page")
    # Parse the HTML with BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")
    # Find the "Today's featured article" div
    tfa_div = soup.find("div", id="mp-tfa")
    # Find the first link in the div, which will be the article title
    link = tfa_div.find("p").find("b").find("a")
    return get_article(link.text)


def get_article(article_name):
    page = api.page(article_name)
    if not page.exists():
        raise ValueError(f'Article "{article_name}" does not exist.')
    return page.title, page.text


def generate_youtube_movie(filename, images, audio):
    audio_clips = [mpy.AudioFileClip(str(a)) for a in audio]
    audio_durations = [c.duration for c in audio_clips]
    frame_rate = np.mean([c.fps for c in audio_clips])

    clips = []
    for img, audio, duration in zip(images, audio_clips, audio_durations):
        clip = mpy.ImageClip(str(img)).set_duration(duration)
        clip = clip.set_audio(audio)
        clips.append(clip)

    video = mpy.concatenate_videoclips(clips)
    video.write_videofile(str(filename), fps=frame_rate, threads=64)
    
    
def generate_audio(prompt: str, idx: int = 0):
    inputs = processor_audio(
        text=[prompt],
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    
    speech_values = model_audio.generate(**inputs, do_sample=True)
    sampling_rate = model_audio.generation_config.sample_rate
    
    filepath = folderpath / f"audio_{idx}.wav"
    scipy.io.wavfile.write(
        filepath,
        rate=sampling_rate,
        data=speech_values.cpu().numpy().squeeze(),
    )
    return filepath


def generate_image(prompt: str, idx: int = 0, n_steps=40, high_noise_frac=0.8):
    image = base(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images
    image = refiner(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
    ).images[0]
    # # warmuup
    # _ = pipe_image(prompt, num_inference_steps=1)
    # # results
    # image = pipe_image(prompt).images[0]
    filepath = folderpath / f"image_{idx}.png"
    image.save(filepath, "PNG")
    return filepath


def generate_video(
    prompt: str, idx: int = 0, num_inference_steps: int = 50, 
    height=320, 
    width=576, 
    num_frames=24
):
    # warmuup
    filepath = folderpath / f"video_{idx}.mp4"
    video = pipe_video(
        prompt, 
        num_inference_steps=num_inference_steps, 
        # height=height, 
        # width=width, 
        # num_frames=num_frames,
        
    ).frames
    export_to_video(video, str(filepath))
    return filepath


def generate_content(script: YoutubeScript, title):
    filepaths = {
        "image": [],
        "video": [],
        "audio": [],
    }
    for idx, snippet in tqdm.tqdm(
        enumerate(script.snippets), total=len(script.snippets)
    ):
        # narrator
        prompt = snippet.narrator
        narrator_prompt = prompt
        prompt = f"[{narrator_voice}]: {prompt} ... [background noise]"
        filepath = generate_audio(prompt, idx)
        filepaths["audio"].append(filepath)
        
        audio_clip = mpy.AudioFileClip(str(filepath))
        duration = audio_clip.duration / 24
        
        if not snippet.image:
            prompt = narrator_prompt
            prompt = tovisual_chain.run({"old_prompt": prompt, "visual_type": "image"})
        else:
            prompt = snippet.image
            
        print(prompt)
        prompt = f"A cartoon bear dressed as a news reporter reporting on the {title} and in front of this scene: {prompt}; cartoon, animation"
        filepaths["image"].append(generate_image(prompt, idx))
        
        if snippet.video:
            prompt = snippet.video
            prompt = tovisual_chain.run({"old_prompt": prompt, "visual_type": "video"})
        else:
            prompt = narrator_prompt
        
        print(prompt)
        prompt = "{prompt}; animation-style, cartoon"
        print(duration)
        filepaths["video"].append(generate_video(
            prompt, idx, 
            # num_frames=int(duration*12)
        ))

    return filepaths


if __name__ == "__main__":
    redo = True
    if redo:
        title, text = get_article_of_the_day()
        print(title)
        print("generating script")
        youtube: YoutubeScript = chain.run({"title": title, "text": text})
        if not isinstance(youtube, YoutubeScript):
            youtube : YoutubeScript = format_chain.run({"script": youtube})
            # youtube = output_parser.parse(youtube)
        print("generating images and videos")
        filepaths = generate_content(youtube, title)

        print("save pydantic youtube script")
        with open(folderpath / "youtube.json", "w") as f:
            json.dump(youtube.dict(), f, indent=4)
            
    else:
        filepaths = {
            "image": [], 
            "video": [],
            "audio": [],
        }
        paths = folderpath.glob("*.png")
        paths = sorted(paths)
        for filepath in paths:
            filepaths["image"].append(filepath)
        paths = folderpath.glob("*.mp4")
        paths = sorted(paths)
        for filepath in paths:
            filepaths["video"].append(filepath)
        paths = folderpath.glob("*.wav")
        paths = sorted(paths)
        for filepath in paths:
            filepaths["audio"].append(filepath)
        
    print("generate movie")
    filename = folderpath / "youtube_movie.mp4"
    generate_youtube_movie(filename, filepaths["image"], filepaths["audio"])
