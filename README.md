

``` 
Illustrations descriptor/
├── src/                          # Source code
│   ├── Action_Detection/         # Action detection pipeline
│   │   ├── benchmarking_pipeline.py      # Pipeline to benchmark the LLaVa models
│   │   ├── full_pipeline.py      # Main pipeline script with LlaVa
│   │   └── llava_utils.py        # Utilities for LLaVA interaction
│   ├── Entity_Detection_+_Image_Captioning/
│   │   ├── benchmarking.ipynb    # Benchmarking entity detection models
│   │   └── Entity_Detection&_Image_Captioning.ipynb # Entity detection and Image Captioning pipeline
│   ├── Evaluation/
│       └── BERTScore.ipynb       # Quantitative evaluation on Llava outputs
│   ├── Data_Analysis/
│       └── Entity_EDA.ipynb      # Data Analysis on the entity detection results
│   ├── Illustration_Detection/
│   │   ├── extractor.py          # Illustration extraction script
│   │   └── annotator.py          # Annotator for illustration detection
├── output/                       # Outputs
│   ├── Entity_Caption.csv        # Captions for entities
│   ├── LLaVA_Outputs.csv         # Action detection outputs
│   ├── Illustrations_Detection_Performances.csv  # Performances results of the Illustration detection process
│   ├── preprocessed_METADATA.txt  # Statistics of illustrations extraction for each pdf
|   └── llava_benchmarking/
|       └── ....out               # Results of 1 benchmark test for a llava model
│
├── 20_samples/                   # sample images
│
├── out.md                        # Examples of outputs from LLaVa
└── README.md                     # Project documentation
```
## Important Note

This is a ML4Science project so it is not feasible or necessary to reproduce our work. For `Action Detection`, we submitted the job to scitas. For `Entity Detection`, we mostly work on Google Drive. Due to the large size of our dataset, we didn't upload it here either. You can find our output in the `output` folder, in the files `Entity_Caption.csv`, `LlaVa_Outputs.csv` and `preprocessing_METADATA.txt`.


## Overview

This project focuses on advancing multi-modal narrative understanding by analyzing illustrations in children's books. By leveraging state-of-the-art Large Language Models (LLMs), particularly LLaVA, we extract key entities, detect actions, and analyze emotional tones conveyed through visual content. The ultimate goal is to establish a robust framework for understanding how illustrations enhance storytelling.

## Features

- **Illustration Detection**: Automatically detect and extract pages containing illustrations from scanned children’s books.
- **Entity Detection and Classification**: Identify and classify entities (Humans, Animals, Both, or Neither).
- **Action Detection**: Generate structured descriptions of actions performed by entities using LLaVA with various prompting strategies.
- **Emotional Tone Analysis**: Infer the emotional tone conveyed by visual elements and interactions in the illustrations.
- **Evaluation**: Benchmark the performance of entity and action detection models using qualitative and quantitative metrics.

## Workflow

### Data Collection and Preparation:
1. Extract scanned pages from a dataset of 2,800 children’s books using the PyMuPDF library.
2. Detect pages with illustrations by filtering out text-only and decorative pages.

### Illustration Detection:
- Apply simple statistical pixel analysis to identify mimetic illustrations (depicting real-world entities).

### Entity Detection and Classification:
- Use pre-trained object detection models (YOLOv10, Faster R-CNN, and Detr) to detect entities.
- Map detected entities into four classes: Human, Animal, Both, or Neither.

### Action Detection and Captioning:
- Apply the LLaVA (LLaVA-v1.6-vicuna-13b) model with:
  - **Zero-Shot prompting**: Directly describe actions in the scene.
  - **Chain-of-Thought prompting**: Break descriptions into structured reasoning steps.
- Extract key details: Setting, Entities, Actions, Interactions, and Emotional Tone.

### Evaluation:
1. Perform qualitative evaluation to assess the richness and accuracy of outputs.
2. Use BERTScore to quantify semantic consistency between prompting strategies.

## Example Outputs

**Input Illustration:**  
Illustration: *Cinderella attending the royal ball (page 10).*

**Zero-Shot Output:**  
- **Setting**: "A grand room or hall with many people in the background."
- **Entities**: "A man and a woman in the center."
- **Actions**: "The man is holding the woman's hand, and she is seated, looking up at him."
- **Emotional Tone**: "Warmth and intimacy amidst a public event."

**Chain-of-Thought Output:**  
- **Setting**: "A large room with a historical atmosphere, possibly illuminated by natural light."
- **Entities**: "A man and woman in historical attire."
- **Actions**: "The man appears to be engaged in conversation with the woman, who is seated."
- **Interactions**: "The two central figures share a private connection, while others observe."
- **Emotional Tone**: "A moment of intimacy and social engagement with historical context."

## Evaluation Metrics

### Illustration Detection:
- **Accuracy**: 96%
- **Precision**: 73%

### Entity Detection:
- Benchmarked using YOLOv10, Faster R-CNN, and Detr.
- **Best model**: YOLOv10 with 69% accuracy.

### Action Detection:
- **Qualitative Analysis**: Consistency and richness of descriptions.
- **Quantitative Analysis**: Semantic similarity evaluated using BERTScore across components (Setting, Entities, Actions, Interactions, Emotional Tone).

## Limitations

- **Entity Coverage**: Limited by predefined categories in the COCO framework (e.g., specific animals).
- **Image Quality**: Degraded historical book illustrations impact model accuracy.
- **Action Detection**: Fine-grained body posture detection remains challenging.

## Future Work

1. Fine-tune LLaVA for improved action grounding.
2. Expand annotations to include contemporary illustrations and broader categories.
3. Address hallucinations in Chain-of-Thought reasoning.
