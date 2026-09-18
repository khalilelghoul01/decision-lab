# Modified model artifact

This is Decision Lab, a derivative of LiquidAI/LFM2.5-350M.
The checkpoint adds supervised LoRA updates and a small trained answer-token head.
LoRA updates are merged for inference. The original full vocabulary output is replaced
with constrained choice scores. Weights are converted to fp16 and an ONNX graph
with explicit attention and convolution caches. Calibration changes the probability readout.
It is not an unmodified Liquid AI release or the Jev model.
See BASE_MODEL_LICENSE and BASE_MODEL_README.md for upstream terms and attribution.

This candidate additionally quantizes constant matrix multiplications to signed
4-bit weights in blocks of 32. Embeddings, activations, and caches remain FP16;
the answer-token head remains FP32. Its calibration is fitted separately.
