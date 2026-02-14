# LLM Data Pipeline Lab 2

## Overview

This lab demonstrates a streaming language modeling (LM) data pipeline using Hugging Face Datasets and Transformers. The pipeline processes large text corpora in a memory-efficient streaming manner, tokenizing data on-the-fly, concatenating texts, and chunking into fixed-length blocks suitable for PyTorch-based LM training.

## Key Features

- **Streaming Processing**: Handles web-scale datasets without loading everything into RAM.
- **On-the-Fly Tokenization**: Uses Hugging Face tokenizers to convert text to tokens dynamically.
- **Rolling Buffer Chunking**: Concatenates and chunks text into fixed-size blocks for LM training.
- **PyTorch Integration**: Produces DataLoader-ready batches with input_ids, attention_mask, and labels.

## Changes and Enhancements Made

### Dataset and Tokenizer Updates

- **Original Setup**: Used Wikitext-2-raw-v1 dataset with GPT-2 tokenizer.
- **Updated Setup**: Retained Wikitext dataset for compatibility, but switched to BERT-base-uncased tokenizer for a different tokenization approach (WordPiece vs. BPE).

### New Features Added

- **Configuration Dictionary**: Centralized all hyperparameters (dataset name/config, tokenizer, block size, batch size, min text length) in a `config` dict for easy customization and maintenance.
- **Text Filtering**: Added filtering to exclude texts shorter than 10 characters, improving data quality by removing noise.
- **Progress Tracking**: Integrated `tqdm` progress bar for visual feedback during batch iteration.
- **Memory Optimization**: Used `remove_columns=["text"]` after tokenization to discard unused text data, reducing memory footprint.

### Optimizations Implemented

- **Efficient Buffer Management**: Replaced the list-based buffer with `collections.deque` for O(1) time complexity on pop operations when chunking tokens, improving performance for large streams.
- **Parameterized Code**: All key values now reference the config dict, making the pipeline more flexible and easier to modify.

## How to Run

1. Ensure dependencies are installed: `transformers`, `datasets`, `torch`, `tqdm`.
2. Configure the `config` dictionary as needed.
3. Run the notebook cells sequentially.
4. The sample section demonstrates iterating over batches with progress tracking.

## Output

The pipeline generates batches of shape `[batch_size, block_size]` (e.g., `[8, 128]`), ready for LM training. Note: BERT tokenizer may issue warnings about sequence length, but chunking to 128 tokens ensures compatibility.
