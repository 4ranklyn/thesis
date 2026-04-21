# Summary of `finaltest_refactored.ipynb`

This document provides a comprehensive report of the machine learning pipeline developed in `finaltest_refactored.ipynb`. The notebook focuses on **Indonesian Image Captioning**, setting up, training, and evaluating six different Vision-Encoder-Decoder models on the Flickr8k dataset translated into Indonesian.

## High-Level Overview

The notebook accomplishes the following major tasks:
1. **Setup & Initialization**: Global seeds are set for reproducibility. Directories for saving the various models are established. Common hyperparameters (Batch Size, Epochs, Learning Rate, etc.) are defined.
2. **Data Cleaning & Noise Reduction**: Uses a custom statistical spell checker based on Peter Norvig's algorithm to correct typos and anomalies in the Indonesian captions. Fixes data leakage by ensuring all captions corresponding to a single image stay in the same split (Train 70% / Val 15% / Test 15%).
3. **Data Loading & Preprocessing**: Sets up a custom PyTorch `Dataset` (`FlickrIndoDataset`) and `DataCollator` (`SmartDataCollator`). Uses Hugging Face's datasets, tokenizers, and image processors. A unified IndoNLG tokenizer is employed across all models for consistency.
4. **Model Training**: It fine-tunes 6 different combinations of Vision Encoder-Decoder models. Early stopping is applied to prevent overfitting.
   - **Models trained**:
     - ViT + IndoBERT
     - ViT + GPT2 (Indonesian)
     - Swin Transformer + IndoBARTv2
     - ViT + IndoBARTv2
     - Swin Transformer + GPT2 (Indonesian)
     - Swin Transformer + IndoBERT
5. **Evaluation**: Evaluates the trained models comprehensively using the `pycocoevalcap` suite, computing metrics like BLEU (1-4), METEOR, ROUGE_L, CIDEr, and SPICE.
6. **Inference & Visualization**: Generates captions for test images and displays them alongside ground truth. Evaluates model loss and BLEU scores, producing comparison charts saved as images.

---

## Detailed Cell-by-Cell Breakdown

### Cell 1: Global Setup
- Sets the global random seed for reproducibility (`random`, `numpy`, `torch`).
- Defines a `Config` class containing paths (`IMAGE_DIR`, `CSV_FILE`), global hyperparameters (`BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`, `MAX_LENGTH`, `FP16`, `WEIGHT_DECAY`), and output directories for 6 different models.
- Creates necessary subdirectories for saving model checkpoints.

### Cell 2: Data Cleaning and Leakage Fix
- Implements a statistical spell checker (Peter Norvig's approach) to correct typos in the dataset.
- Cleans symbols, lowercases text, and replaces misspelled words.
- Splits the dataset into Train (70%), Validation (15%), and Test (15%) splits based on **unique image names** rather than rows, which resolves data leakage issues (as Flickr8k has ~5 captions per image).

### Cell 3: Data Cleaning Visualizations
- Visualizes the top 15 words before and after cleaning using horizontal bar charts (`seaborn`).
- Compiles a Pandas DataFrame displaying the top 10 typos and their corrections, saving it as a CSV file.

### Cell 4: Custom Dataset and Collator
- Defines `FlickrIndoDataset` for loading image pixel values and tokenized caption labels.
- Defines `SmartDataCollator` for batching features and generating `decoder_input_ids`.
- Defines `compute_metrics_bleu` for evaluation during training using Hugging Face's `evaluate` library.
- Defines a plotting function `plot_training_history` to display Loss and BLEU score progression.

### Cell 5: Hyperparameters Display
- Displays a Pandas DataFrame summarizing the universal finetuning hyperparameters applied to all models.

### Cell 6: Experiment Configuration
- Defines `EXPERIMENT_CONFIGS`, an object storing model-specific hyperparameters.
- Defines `log_experiment()` to save the results and settings of each run into a JSON file.
- Initializes the `UNIFIED_TOKENIZER` (`MBartTokenizer` from `indobenchmark/indobart-v2`), which is shared across all models.

### Cells 7 to 12: Model Training
Each of these cells trains one specific model architecture. The workflow is largely identical across them:
1. Initializes `AutoImageProcessor` and `VisionEncoderDecoderModel` from pre-trained weights.
2. For GPT2 models, it applies `add_cross_attention=True` and `is_decoder=True`.
3. Resizes token embeddings to match the unified tokenizer.
4. Aligns special tokens (`pad_token_id`, `bos_token_id`, `eos_token_id`).
5. Implements **Smart Freezing**: Freezes the entire encoder except for the last layer to preserve pre-trained visual representations while allowing some adaptation.
6. Instantiates `Seq2SeqTrainingArguments` and `Seq2SeqTrainer` with `EarlyStoppingCallback`.
7. Calls `trainer.train()`, plots the history, logs the experiment, and saves the weights.

- **Cell 7**: Trains `ViT` + `IndoBERT`.
- **Cell 8**: Trains `ViT` + `GPT2` (small Indonesian).
- **Cell 9**: Trains `Swin Transformer` + `IndoBARTv2`.
- **Cell 10**: Trains `ViT` + `IndoBARTv2`.
- **Cell 11**: Trains `Swin Transformer` + `GPT2` (small Indonesian).
- **Cell 12**: Trains `Swin Transformer` + `IndoBERT`.

### Cell 13: Comprehensive Evaluation
- Evaluates all 6 models using the `pycocoevalcap` suite on the 15% unseen test data.
- Handles edge cases where model configurations fail to load properly by reconstructing the architecture and loading `.safetensors`/`.bin` weights manually.
- Uses Beam Search (`num_beams=4`) for caption generation.
- Computes BLEU-1 to BLEU-4, METEOR, ROUGE_L, CIDEr, and SPICE.
- Concatenates the results into a Pandas DataFrame and exports them to `comprehensive_metrics_comparison.csv`.

### Cell 14: Final Inference (Swin + IndoBARTv2)
- Performs bulk inference on the test dataset using the best-performing model (assumed to be `Swin + IndoBARTv2`).
- Uses Beam Search to generate captions and appends them to a new column (`prediksi_swin_indobart`) in the test DataFrame.
- Saves the inference results to a CSV file.

### Cell 15: Inference Visualization
- Randomly samples 2 images from the test set where captions were successfully generated.
- Plots the image alongside the predicted caption and the ground truth using `matplotlib`.

### Cells 16 & 17: Preprocessing Illustration
- Illustrates how the Swin Transformer processes an image.
- Plots 1) the original image, 2) the image resized to a 224x224 tensor, and 3) an illustration of the patching mechanism (32x32 grids, representing the 4x4 patches with a 7x7 window).

### Cell 18: Tokenizer Demonstration
- Demonstrates how the `IndoBERT` tokenizer breaks down 5 randomly sampled Indonesian captions into tokens and token IDs.

### Cell 19: Additional Inference Visualizations
- Another visualization block displaying 3 random examples of the `Swin + IndoBARTv2` model predicting captions alongside their ground truth.

### Cells 20 & 21: Model Metric Plots
- Hardcodes the loss and BLEU metrics for all 6 models over 7 epochs.
- Plots **Training vs Validation Loss** for each of the 6 models in a 3x2 grid.
- Plots the **Validation BLEU Score** for all 6 models on a single line chart for direct comparison.
- Saves these final plots as `grafik_loss_evaluasi.png` and `grafik_bleu_evaluasi.png`.
