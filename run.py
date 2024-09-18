import ChatTTS
import torch
import torchaudio
import numpy as np
import tools.audio.av
import shutil


chat = ChatTTS.Chat()
chat.load(compile=True) # Set to True for better performance


# rand_spk = chat.sample_random_speaker()
# print(len(rand_spk))

# rand_spk = chat.sample_random_speaker()
# rand_spk = chat.sample_audio_speaker(torch.Tensor(tools.audio.av.load_audio('./yue.mp3', 22050)))
# rand_spk = chat.sample_audio_speaker(torch.Tensor())
print(len(rand_spk))
# print(rand_spk)
# rand_spk = torchaudio.load('./yue.mp3')
# print(rand_spk)
# rand_spk = torch.load('./speaker/5.pt')
# print(len(rand_spk))
# rand_spk = torch.load('./speaker/11.pt')
# print(len(rand_spk))
# rand_spk = torch.load('./yue.pt')
# print(len(rand_spk))

# print(rand_spk) # save it for later timbre recovery

params_infer_code = ChatTTS.Chat.InferCodeParams(
    prompt = '[speed_0]',
    spk_emb = rand_spk, # add sampled speaker 
    temperature = .3,   # using custom temperature
    top_P = 0.7,        # top P decode
    top_K = 20,         # top K decode
)

###################################
# For sentence level manual control.

# use oral_(0-9), laugh_(0-2), break_(0-7) 
# to generate special token in text to synthesize.
params_refine_text = ChatTTS.Chat.RefineTextParams(
    prompt='[oral_2][laugh_0][break_6]',
)

###################################
# For word level manual control.

text = '好久不见。你还记得咱们大学那会儿吗？你听到的是开源项目 T T S List。那可是风华正茂的岁月啊！还记得咱俩爬那个山顶看日出吗？当时许的愿望，我到现在还记得呢。'
wavs = chat.infer(text, skip_refine_text=False, params_refine_text=params_refine_text,  params_infer_code=params_infer_code)
"""
In some versions of torchaudio, the first line works but in other versions, so does the second line.
"""
try:
    torchaudio.save("word_level_output.wav", torch.from_numpy(wavs[0]).unsqueeze(0), 24000)
except:
    torchaudio.save("word_level_output.wav", torch.from_numpy(wavs[0]), 24000)