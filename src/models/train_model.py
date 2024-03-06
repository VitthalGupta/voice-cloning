import gradio as gr
from bark.generation import load_codec_model, SAMPLE_RATE, preload_models, codec_decode, generate_text_semantic, generate_coarse, generate_fine
from bark.api import generate_audio, semantic_to_waveform, text_to_semantic
from encodec.utils import convert_audio
import torchaudio
import torch
import nltk  # we'll use this to split into sentences
from hubert.hubert_manager import HuBERTManager
from hubert.pre_kmeans_hubert import CustomHubert
from hubert.customtokenizer import CustomTokenizer
import numpy as np
import os
from tqdm import tqdm
from scipy.io.wavfile import write as write_wav

parent_dir = './data'

# Create folders if they don't exist
folder_names = ["external", "processed", "models", "interim", "raw"]

for folder_name in folder_names:
    folder_path = os.path.join(parent_dir, folder_name)
    try:
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")
    except FileExistsError:
        pass
    except Exception as e:
        print(f"Error creating folder {folder_path}: {str(e)}")


def process_audio(input_path, voice_name, input_compute_device='mps'):
    # Device declaration
    device = input_compute_device
    model = load_codec_model(use_gpu=False if device == 'cpu' else True)

    print('Loading HuBERT model')
    hubert_manager = HuBERTManager()
    hubert_manager.make_sure_hubert_installed()
    hubert_manager.make_sure_tokenizer_installed()

    hubert_model = CustomHubert(
        checkpoint_path='data/models/hubert/hubert.pt').to(device)
    tokenizer = CustomTokenizer.load_from_checkpoint(
        'data/models/hubert/tokenizer.pth').to(device)

    # Load and pre-process the input audio waveform
    print('Loading audio')
    wav, sr = torchaudio.load(input_path)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.to(device)

    # Extract semantic vectors from HuBERT
    semantic_vectors = hubert_model.forward(
        wav, input_sample_hz=model.sample_rate)
    semantic_tokens = tokenizer.get_token(semantic_vectors)

    # Extract discrete codes from EnCodec
    with torch.no_grad():
        encoded_frames = model.encode(wav.unsqueeze(0))
    codes = torch.cat([encoded[0]
                      for encoded in encoded_frames], dim=-1).squeeze()

    # move codes to cpu
    codes = codes.cpu().numpy()
    # move semantic tokens to cpu
    semantic_tokens = semantic_tokens.cpu().numpy()

    # Creating path using os
    if not os.path.exists('data/processed/' + voice_name):
        os.makedirs('data/processed/' + voice_name)
        print(
            f"The directory '{voice_name}' has been created successfully.")
    else:
        print(f"The directory '{voice_name}' already exists.")
    # Save the prompts as a .npz file
    output_path = f'data/processed/{voice_name}/' + voice_name + '.npz'
    np.savez(output_path, fine_prompt=codes, coarse_prompt=codes[:2, :]
    , semantic_prompt=semantic_tokens)

    # Copy the npz file to the src/models/bark/assets folder
    os.system(f"cp {output_path} src/models/bark/assets/{voice_name}.npz")

    return output_path, f"Voice prompt for {voice_name} generated successfully."


def generate_voice(text, voice_name, input_slider_generate_voice_gen_temp, input_slider_generate_voice_min_eos_p, input_slider_generate_voice_gen_temp_semantic2wave):
    
    # Local variable declaration    
    # set to None if you don't want to use finetuned semantic
    semantic_path = "data/models/semantic_output/pytorch_model.bin"
    # set to None if you don't want to use finetuned coarse
    coarse_path = "data/models/coarse_output/pytorch_model.bin"
    # set to None if you don't want to use finetuned fine
    fine_path = "data/models/fine_output/pytorch_model.bin"
    use_rvc = False  # Set to False to use bark without RVC
    rvc_name = 'mi-test'
    rvc_path = f"Retrieval-based-Voice-Conversion-WebUI/weights/{rvc_name}.pth"
    index_path = f"Retrieval-based-Voice-Conversion-WebUI/logs/{rvc_name}/added_IVF256_Flat_nprobe_1_{rvc_name}_v2.index"
    device = "cuda:0"
    is_half = True

    # download and load models
    preload_models(
        text_use_gpu=True,
        text_use_small=False,
        text_model_path=semantic_path,
        coarse_use_gpu=True,
        coarse_use_small=False,
        coarse_model_path=coarse_path,
        fine_use_gpu=True,
        fine_use_small=False,
        fine_model_path=fine_path,
        codec_use_gpu=True,
        force_reload=False,
        path="data/models"
    )

    if use_rvc:
        from rvc_infer import get_vc, vc_single
        get_vc(rvc_path, device, is_half)

    text = text.replace("\n", " ").strip()
    sentences = nltk.sent_tokenize(text)

    silence = np.zeros(int(0.25 * SAMPLE_RATE))

    pieces =[]
    gen_temp = input_slider_generate_voice_gen_temp
    min_eos_p = input_slider_generate_voice_min_eos_p
    gen_temp_semantic2wave = input_slider_generate_voice_gen_temp_semantic2wave
    for sentence in tqdm(sentences, desc="Generating audio"):
        # audio_array = generate_audio(sentence, history_prompt=voice_name, temp=gen_temp, min_eos_p=0.05)
        semantic_tokens = generate_text_semantic(
            text,
            history_prompt=voice_name,
            temp=gen_temp,
            min_eos_p=min_eos_p,
        )
        audio_array = semantic_to_waveform(
            semantic_tokens,
            history_prompt=voice_name,
            temp=gen_temp,
        )
        pieces+=[audio_array, silence.copy()]

    audio_array = np.concatenate(pieces)
    # Check if the directory exists
    if not os.path.exists('data/external/' + voice_name):
        os.makedirs('data/external/' + voice_name)
        print(
            f"The directory '{voice_name}' has been created successfully.")
    
    write_wav(f'data/external/{voice_name}/{voice_name}_audio.wav', SAMPLE_RATE, audio_array)
    print("Audio saved successfully.")
    return  f"Audio {voice_name}_audio.wav generated successfully."

# Sliders
# Generate voice
input_slider_generate_voice_gen_temp = gr.inputs.Slider(minimum=0, maximum=1, label="Text Semantic temp", step=0.01)

input_slider_generate_voice_min_eos_p = gr.inputs.Slider(minimum=0, maximum=5, label="Text Semantic min_eos_p", step=0.01)

input_slider_generate_voice_gen_temp_semantic2wave = gr.inputs.Slider(
    minimum=0, maximum=1, label="Semantic to Waveform generation temperature (1.0 more diverse, 0.0 more conservative)", step=0.01)

# Gradio Radio for device selection
input_compute_device = gr.inputs.Radio(["cpu", "cuda:0", "mps"], label="Compute device")

# Creating Tabs
# Generating tokens for voice cloning
create_voice = gr.Interface(fn=process_audio, inputs=[input_compute_device, "text", "text"],
                     outputs=["text", "text"],
                    capture_session=True,
                    live=False,
                    description="Upload an audio file to generate voice prompts from it."
)

# Generating audio from voice prompts using previously generated voice tokens
generate_audio_ = gr.Interface(fn=generate_voice, 
                               inputs=["text", "text",input_slider_generate_voice_gen_temp,
                                       input_slider_generate_voice_min_eos_p, input_slider_generate_voice_gen_temp_semantic2wave],
                            outputs="text",
                            live=False,
                            description="Upload a text file and select the name the speaker and then generate audio from it."
)

tabbed_layout = gr.TabbedInterface([create_voice, generate_audio_],
                    ['Voice Cloning','Audio Generation'],
                    title="Voice cloning using HuBERT and Bark",
                    analytics_enabled=True
)

# Launching the interface
if __name__ == "__main__":
    tabbed_layout.launch()