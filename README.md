# Invasive Plant Detection

A PyTorch pipeline that classifies whether an aerial image contains invasive plant species, built on the Kaggle [Invasive Species Monitoring](https://www.kaggle.com/c/invasive-species-monitoring) dataset. Instead of feeding raw RGB images straight into a CNN, the pipeline first extracts a texture representation via Sobel-based edge/texture segmentation, then trains a convolutional classifier on that representation. Hyperparameters (optimizer, learning rate, batch size) are tuned automatically with Optuna. The trained CNN is also exported to ONNX and benchmarked against PyTorch for CPU inference.

## How it works

1. **Data prep:** Downloads and extracts the Kaggle dataset (images + labels) into Google Drive, for use in a Colab environment.
2. **Texture segmentation:** Converts each image to grayscale and applies Sobel edge detection on both axes to produce a texture/gradient magnitude map, which becomes the model's input channel.
3. **Dataset/loader:** A custom `TextureDataset` resizes, segments, and normalizes images on the fly, with a configurable train/test split.
4. **Model:** A 3-block CNN (Conv → ReLU → MaxPool, channels 32 → 64 → 128) followed by two fully connected layers and a sigmoid output for binary classification.
5. **Hyperparameter tuning:** An Optuna study searches over optimizer type, learning rate, and batch size, using F1 score as the objective and pruning underperforming trials.
6. **Training/evaluation:** Trains with the best-found hyperparameters and reports accuracy, precision, recall, and F1 on the held-out split. The trained model weights are saved and reloaded for evaluation.
7. **ONNX export and benchmark:** Exports the CNN to ONNX (opset 17, dynamic batch), verifies outputs match PyTorch, and benchmarks CPU latency for PyTorch vs ONNX Runtime, including an optional int8 quantized variant.

## Requirements

- Python 3, PyTorch
- OpenCV (`cv2`)
- pandas, NumPy, scikit-learn
- Optuna
- ONNX, ONNX Runtime, onnxscript (`pip install onnx onnxruntime onnxscript`)
- Kaggle API (`kaggle.json` credentials)
- Designed to run in Google Colab (uses `google.colab.drive` and `google.colab.files`)

## Usage

1. Open the notebook in Google Colab.
2. Upload your `kaggle.json` API credentials when prompted.
3. Mount Google Drive. The notebook downloads, extracts, and caches the dataset there.
4. Run the cells in order: data prep → segmentation → preprocessing → hyperparameter search → training/evaluation.
5. The trained model is saved to Drive and can be reloaded for inference/testing without retraining.
6. Run the ONNX section at the end of the notebook to export the model and reproduce the CPU benchmark. If `model.pth` is missing it falls back to random weights, which gives valid latency numbers but not meaningful prediction comparisons.

## Results

**Hyperparameter search (Optuna, 27 trials):** the study explored SGD, Adam, and RMSprop across learning rates (1e-4 to 2.5e-3) and batch sizes (32/64/128), pruning weak trials early based on intermediate F1. Many trials collapsed to a degenerate "predict everything as invasive" solution (recall 1.0, precision ~0.63), which pruning helped weed out.

| Trial | Optimizer | LR | Batch Size | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| **6 (best)** | SGD | 0.0025 | 32 | 0.7147 | 0.9653 | **0.8213** |
| 1 | SGD | 0.0005 | 64 | 0.7107 | 0.9722 | 0.8211 |
| 16 | SGD | 0.0005 | 64 | 0.7348 | 0.9236 | 0.8185 |
| 0 | SGD | 0.0001 | 64 | 0.6992 | 0.9688 | 0.8122 |

SGD consistently outperformed Adam and RMSprop, which tended toward the degenerate solution above.

**Final model performance:**

| Split | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Training | 0.9793 | 0.9794 | 0.9879 | 0.9836 |
| Testing | 0.8519 | 0.8641 | 0.9113 | 0.8870 |

The gap between training and testing metrics suggests some overfitting on the training split; the texture-based CNN still generalizes reasonably well, correctly identifying invasive plants in held-out images roughly 85% of the time.

**Qualitative results:** sample predictions below show the original image, its Sobel-based segmentation mask, and the mask overlaid on the image alongside the model's prediction. The model correctly classifies both invasive (label 1.0) and non-invasive (label 0.0) samples shown here, with the segmentation mask highlighting the plant regions the texture features are drawn from.

![Sample predictions with segmentation masks](sample_predictions.png)

## ONNX Export and CPU Inference Benchmark

The CNN was exported to ONNX and benchmarked against PyTorch on CPU.

### Export
- `torch.onnx.export` (TorchScript exporter), opset 17, input shape `1x1x224x224` with a dynamic batch axis
- Graph: Conv, Relu, MaxPool (x3), Flatten, Gemm, Relu, Gemm, Sigmoid (44.7 MB)

### Correctness
Same inputs through PyTorch and ONNX Runtime, 32 real test-split images:
- Max abs difference, probabilities: 2.4e-7
- Max abs difference, logits: 8.9e-7 (logits checked separately so a saturated sigmoid can't hide differences)
- Batch of 8 (dynamic axis): 6.0e-8
- Prediction agreement: 32/32

### Method
- PyTorch with `torch.inference_mode()` vs ONNX Runtime `CPUExecutionProvider` (all graph optimizations on)
- Batch size 1, identical input tensors, 2 intra-op threads for both, 1 inter-op thread
- 20 warmup runs excluded, then 300 timed runs per framework with `time.perf_counter`, interleaved in 3 rounds of 100
- JPEGs loaded into memory first, so disk and Drive I/O are excluded
- Model weights were randomly initialized for this run. Latency of a dense CNN does not depend on weight values; prediction-level comparisons for int8 do.

Environment: Intel Xeon @ 2.20GHz, 2 logical cores, torch 2.11.0+cpu, onnx 1.23.2, onnxruntime 1.31.0, opencv 5.0.0, Python 3.13.16 (Google Colab).

### Results (ms)

**Model inference only**

| | mean | p50 | p95 |
|---|---|---|---|
| PyTorch | 46.38 | 35.22 | 101.78 |
| ONNX Runtime | 17.42 | 16.84 | 19.93 |
| Speedup | 2.66x | 2.09x | 5.11x |

**End to end per image** (JPEG decode, resize, Sobel texture map, resize, normalize, inference)

| | mean | p50 | p95 |
|---|---|---|---|
| PyTorch | 75.13 | 58.92 | 159.09 |
| ONNX Runtime | 49.83 | 44.13 | 73.40 |
| Preprocessing only | 29.65 | 30.10 | 38.68 |
| Speedup | 1.51x | 1.34x | 2.17x |

**Dynamic int8 quantization (ONNX Runtime, uint8 weights)**

| | mean | p50 | p95 | size |
|---|---|---|---|---|
| ORT fp32 | 19.45 | 16.89 | 31.34 | 44.7 MB |
| ORT int8 | 33.12 | 23.31 | 55.46 | 11.2 MB |

Int8 vs fp32 max abs probability difference: 3.2e-3 (random weights, so agreement is not a meaningful accuracy measure).

### Takeaways
- ONNX Runtime roughly halves median model latency and removes most of PyTorch's tail latency, likely from whole-graph execution, Conv+ReLU fusion, CPU-tuned conv kernels and a preallocated memory arena.
- End to end gains are smaller because Sobel preprocessing is about two thirds of the ONNX Runtime pipeline. Preprocessing is the next optimization target.
- Dynamic int8 made the model 4x smaller but slower: runtime quantization ops and ConvInteger kernels cost more than they save at batch 1. Static quantization with calibration is the next thing to try.

## Notes

- This is a research/experimentation notebook (Colab-oriented), not a packaged library or CLI.
- The segmentation step is a simple Sobel-based texture extractor, not a learned segmentation model. It's used as a lightweight preprocessing/feature-engineering step ahead of the CNN.
- Benchmark numbers are from a 2-vCPU Google Colab runtime and will differ on other hardware. The relative speedup is more transferable than the absolute milliseconds.
