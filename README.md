# CS566 Project -- Riichi Mahjong AI

A deep learning–based Riichi Mahjong decision system built from Tenhou game logs, designed to assist with real-time gameplay recommendations for discard, call, riichi, and win decisions.

The project focuses on training practical Mahjong agents using supervised learning from expert game logs, combining board-state encoding with action history modeling for stronger strategic understanding.

## Features

- Preprocessing pipeline for converting Tenhou/MJAI logs into training datasets
- Separate models for:
  - **Tsumo decisions** (self-draw actions: discard, riichi, kan, tsumo win, etc.)
  - **Dahai reactions** (responses to opponent discards: chi, pon, ron, pass, etc.)
- CNN-based board-state encoder using a lightweight **ResNet**
- Transformer-based **history encoder** for sequential action understanding
- Legal action masking to ensure only valid Mahjong actions are predicted
- Top-k recommendation output for live gameplay assistance
- MJAI-compatible bot integration for simulation and testing

## Model Design
<img width="1665" height="945" alt="model" src="https://github.com/user-attachments/assets/fb92eb3b-4fe6-461b-88a3-a0f70360d20a" />

### Board State Encoder

A residual CNN (ResNet-style) processes the current game state:

- hand tiles
- discards
- melds
- dora indicators
- riichi state
- round information
- player scores
- positional information

This provides strong local pattern recognition for tile efficiency and tactical board evaluation.

### History Encoder

A lightweight Transformer encoder processes recent game events:

- draw/discard order
- calls (chi / pon / kan)
- riichi declarations
- player interactions

This helps capture temporal information that static board state alone cannot represent, such as opponent intent and sequence-sensitive danger signals.

## Training

### Example Setup

- Batch size: 256
- Epochs: 30
- Optimizer: AdamW
- Learning rate: 3e-4
- Weight decay: 1e-4
- ReduceLROnPlateau scheduler
- CrossEntropyLoss with class weighting for imbalance handling

### Performance

Typical validation performance:

- **Discard Top-1 Accuracy:** ~74%
- **Discard Top-3 Accuracy:** ~95%

History encoding consistently improves decision quality compared to board-state-only models.

## Tech Stack

- Python
- PyTorch
- CUDA
- MJAI
- Tenhou log preprocessing
- Docker (for bot simulation)

## Project Goal

The goal is not to create a fully autonomous Mahjong bot, but to build a strong decision-support system that produces high-quality recommendations comparable to advanced Mahjong AI assistants like Mortal.

This project explores how supervised deep learning and sequential modeling can be applied to imperfect-information games with complex human strategy.
