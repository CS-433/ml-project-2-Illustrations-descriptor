
# Illustrations Descriptor

``` 
Illustrations descriptor/
├── src/                          # Source code
│   ├── Action_Detection/         # Action detection pipeline
│   │   ├── full_pipeline.py      # Main pipeline script
│   │   └── extractor.py          # Illustration extraction script
│   ├── llava_utils.py            # Utilities for LLaVA interaction
│
├── output/                       # Outputs
│   ├── Entity_Caption.csv        # Captions for entities
│   ├── LLaVA_Outputs.csv         # Action detection outputs
│
├── Data Analysis/                # Notebooks for evaluation
│   ├── Benchmarking.ipynb        # Model benchmarking
│   └── Entity_Detection.ipynb    # Entity detection analysis
│
├── annotations/                  # Annotated examples for validation
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

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
