# StyleSpeech
This repository is the official PyTorch implementation of the ACM MM Asia paper: StyleSpeech: Parameter-efficient Fine Tuning for Pre-trained Controllable Text-to-Speech, authored by Haowei Lou, Hye-young Paik, Wen Hu, and Lina Yao.

**Abstract**: This paper presents StyleSpeech, a novel Text-to-Speech(TTS) system that enhances the naturalness and accuracy of synthesized speech. StyleSpeech incorporates a Style Decorator structure that enables deep learning models to learn style and phoneme features simultaneously. It improves adaptability and efficiency through the principles of Lower Rank Adaptation(LoRA). 
LoRA allows efficient adaptation of style features in pre-trained models. 
Additionally, we introduce a novel automatic evaluation metric, the LLM-Guided Mean Opinion Score (LLM-MOS), which employs large language models to offer an objective and robust protocol for automatically assessing TTS system performance.
Extensive testing on benchmark dataset shows that our approach markedly outperforms existing state-of-the-art baseline methods in producing natural, accurate, and high-quality speech.
These advancements not only push the boundaries of current TTS system capabilities,  but also facilitate the application of TTS system in more dynamic and specialized, such as interactive virtual assistants, adaptive audiobooks, and customized voice for gaming.

![Architecature diagram](/fig/overview.png)

# Dataset

In this research, we utilize the **Baker Dataset**, a Mandarin speech dataset, to evaluate our results. The methodology, however, is designed to be applicable to any speech dataset. To train the model, the dataset should contain the following columns:

- **D (Phoneme Dictionary)**: A dictionary that maps phonemes to their corresponding indices.
- **X (Phoneme Sequence)**: A sequence of phonemes (e.g., n, i, h, ao) represented as indices in the phoneme dictionary.
- **S (Style Sequence)**: A sequence of style indices (e.g., 0, 3, 0, 3), where each index corresponds to a style attribute for the phoneme.
- **L (Phoneme Duration)**: An integer value representing the duration of each phoneme. This can be calculated using a force aligner.
- **Y (Target Output - Mel Spectrogram)**: A 2D feature representation of the speech. This can be a Mel-Spectrogram or any other suitable acoustic feature.

# Environment Setup

To set up the environment, install the required dependencies by running the following command:

```bash
pip install -r requirements.txt
```

# Trained Model
[Download StyleSpeech Model](https://unsw-my.sharepoint.com/:u:/g/personal/z5258575_ad_unsw_edu_au/EcwTnfpEpMNMotoFqmNCVM8BcDBkVDpJDHQ7kG40REmMDw?e=wLaHBF)

To obtain the password for the download, please contact me via email at:  
[haowei.lou@unsw.edu.au](mailto:haowei.lou@unsw.edu.au)
