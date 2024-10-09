from model import StyleSpeech
import json
import torch
import torchaudio
from config import config

def m2a_gfifinlim(mel_spectrogram, sample_rate,n_fft=400,n_mels=128):
    n_stft = int((n_fft//2) + 1)
    invers_transform = torchaudio.transforms.InverseMelScale(sample_rate=sample_rate, n_stft=n_stft,n_mels=n_mels).to(device)
    grifflim_transform = torchaudio.transforms.GriffinLim(n_fft=n_fft).to(device)
    linear_spectrogram  = invers_transform(mel_spectrogram).to(device)
    pseudo_waveform = grifflim_transform(linear_spectrogram)
    return pseudo_waveform


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


phone = ["SPACE","l","ao","sh","i","n","i","h","ao","SPACE"]
tone = [0,0,3,0,1,0,3,0,3,0]
pho_list = json.loads(open("./phoneme.json","r").read())["normalize_pinyin"]
index_pho= [pho_list.index(p) for p in phone]
config["pho_config"]['word_num'] = 46

stylespeech = StyleSpeech(config,fuse_step=0)
stylespeech.load_state_dict(torch.load("./stylespeech.pth"))
stylespeech.to(device).eval()

phone_num = len(phone)
x = torch.tensor([index_pho]).to(device)                        #phoneme
s = torch.tensor([tone]).to(device)                             #tone
l = torch.tensor([[13 for i in range(phone_num)]]).to(device)   #duration

src_lens = torch.tensor([x.shape[1]]).to(device)
mel_lens = torch.tensor([l.sum(dim=1)]).to(device)
#without duration target
with torch.no_grad():
    melspec,log_l_pred,mel_masks = stylespeech(x,s,src_lens=src_lens,mel_lens=mel_lens,max_mel_len=30*phone_num)
    mel_masks = torch.logical_not(mel_masks.unsqueeze(-1).expand_as(melspec)).float()

melspec = melspec*mel_masks
melspec = melspec[0].T*5000
melspec = melspec
audio = m2a_gfifinlim(melspec, 48000, 1024, 80).unsqueeze(0).cpu().detach()
torchaudio.save(f'sample.wav', audio, 48000)

#with duration target
with torch.no_grad():
    melspec,log_l_pred,mel_masks = stylespeech(x,s,src_lens=src_lens,duration_target=l, mel_lens=mel_lens,max_mel_len=30*phone_num)
    mel_masks = torch.logical_not(mel_masks.unsqueeze(-1).expand_as(melspec)).float()

melspec = melspec*mel_masks
melspec = melspec[0].T*5000
melspec = melspec
audio = m2a_gfifinlim(melspec, 48000, 1024, 80).unsqueeze(0).cpu().detach()
torchaudio.save(f'sample_l.wav', audio, 48000)
