# StyleSpeech
This repository is the official PyTorch implementation of the ACM MM Asia paper: StyleSpeech: Parameter-efficient Fine Tuning for Pre-trained Controllable Text-to-Speech, authored by Haowei Lou, Hye-young Paik, Wen Hu, and Lina Yao.

**Abstract**: This paper introduces StyleSpeech, a novel Text-to-Speech (TTS) system that enhances the naturalness and accuracy of synthesized speech. Building upon existing TTS technologies, StyleSpeech incorporates a unique Style Decorator structure that enables deep learning models to simultaneously learn style and phoneme features, improving adaptability and efficiency through the principles of Lower Rank Adaptation (LoRA). 
LoRA allows efficient adaptation of style features in pre-trained models. 
Additionally, we introduce a novel automatic evaluation metric, the LLM-Guided Mean Opinion Score (LLM-MOS), which employs large language models to offer an objective and robust protocol for automatically assessing TTS system performance.
Extensive testing on benchmark datasets shows that our approach markedly outperforms existing state-of-the-art baseline methods in producing natural, accurate, and high-quality speech.
These advancements not only pushes the boundaries of current TTS system capabilities,  but also facilitate the application of TTS system in more dynamic and specialized, such as interactive virtual assistants, adaptive audiobooks, and customized voice for gaming.

![Architecature diagram](/fig/overview.png)
