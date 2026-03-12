# OpenVINO INT8 Inference Bug on Apple M4 Max

> **Bug:** INT8 quantized models produce ~10% accuracy (random guessing) on Apple M4 Max,
> while the same models run correctly on Apple M1 and Intel x86.
> FP32 inference works correctly on all platforms.

This repository contains a reproduction of the bug using the **official OpenVINO notebook**
[image-classification-quantization](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/image-classification-quantization)
with minimal modifications to collect cross-platform results.

---

## Environment

| Property | Value |
|---|---|
| OpenVINO | `2026.0.0-20965-c6d6a13a886-releases/2026/0` |
| NNCF | `3.0.0` |
| macOS | `15.6.1` (Sequoia) - same on both M1 and M4 Max |

| Platform | Python |
|---|---|
| M1 | `3.9.6` |
| M4 Max | `3.13.5` |
| Colab x86 | `3.12.12` |

Bug reproduces across Python 3.9, 3.12, and 3.13 so Python version is not a factor.

To check your own environment:

```python
import platform
import openvino as ov

core = ov.Core()
print(platform.mac_ver())
print(platform.machine())
print(platform.processor())
print(core.get_property("CPU", "FULL_DEVICE_NAME"))
print(core.get_property("CPU", "OPTIMIZATION_CAPABILITIES"))
```

| Device | `FULL_DEVICE_NAME` | `OPTIMIZATION_CAPABILITIES` |
|---|---|---|
| Apple M1 | `Apple M1` | `['FP32', 'FP16', 'INT8', 'BIN', 'EXPORT_IMPORT']` |
| Apple M4 Max | `Apple M4 Max` | `['FP32', 'FP16', 'INT8', 'BIN', 'EXPORT_IMPORT']` |
| Colab x86 | `Intel(R) Xeon(R) CPU @ 2.20GHz` | `['FP32', 'INT8', 'BIN', 'EXPORT_IMPORT']` |

Both Apple Silicon chips report identical optimization capabilities. Hence, the bug is not reflected in OpenVINO's capability detection.

---

## Results

### Cross-platform accuracy matrix

Each INT8 model was quantized on one platform and then executed on all three platforms.
Model: **MobileNetV2 on CIFAR-10** (10,000 validation images, batch size 1).

| Quantized on → Run on | M1 | M4 Max | Colab x86 |
|---|---|---|---|
| **M1** | 93.57% | 10.00% | 93.26% |
| **M4 Max** | 93.56% | 10.00% | 93.27% |
| **Colab x86** | 93.57% | 10.00% | 93.28% |
| **FP32 (baseline)** | 93.57% | 93.57% | 93.61% |

**Every INT8 model, regardless of where it was quantized, produces ~10% accuracy on M4 Max.
Every INT8 model runs correctly on M1 and x86.**

### Additional diagnostic: forced FP32 precision hint

To check whether the bug was in INT8 kernel execution, we forced FP32 precision on the
compiled model using `INFERENCE_PRECISION_HINT`:

```python
core = ov.Core()
compiled = core.compile_model(
    "model/quantized_mobilenet_v2.xml",
    "CPU",
    {"INFERENCE_PRECISION_HINT": "f32"}
)
int8_acc_forced_fp32 = test_accuracy(compiled, val_loader)
print(f"INT8 model with forced FP32 precision: {int8_acc_forced_fp32[0] * 100:.2f}%")
```

| Platform | Normal INT8 | Forced `INFERENCE_PRECISION_HINT=f32` |
|---|---|---|
| M1 | 93.57% | 93.47% |
| M4 Max | 10.00% | 10.00% |
| Colab x86 | 93.28% | 93.28% |

The hint works correctly on M1 and Colab but has **no effect on M4 Max**, showing that the broken code path is invoked regardless of the precision override.

### Output inspection

Raw output vector for a single sample on M4 Max (M1-quantized model):

```
Output  : [[ 1.13  -1.74   2.94  -0.53  -1.80  -1.37   2.71  -1.09   1.37  -1.63]]
Max     : 2.94
Min     : -1.80
Any NaN : False
Any Inf : False
Top label: 2 | True label: 3
```

The output is **not NaN or Inf** so it is numerically valid but wrong. The values resemble
a plausible probability distribution, but the argmax points to the wrong class.
This suggests the model is computing but producing systematically wrong results,
rather than failing due to numerical corruption.

---

## Hypothesis

OpenVINO documents that ARM platforms execute quantized models in **FP32 simulation mode**
(INT8 ops are executed in floating-point). This works correctly on M1.

Apple M4 Max uses a newer ARM architecture than M1. Based on the output inspection
(plausible but wrong values, no NaN/Inf) the issue appears to be in how OpenVINO's
ARM CPU plugin executes INT8 ops on M4 hardware — likely an incorrect code path
triggered by M4-specific CPU features. The `INFERENCE_PRECISION_HINT=f32` override
does not resolve the issue, suggesting the problematic path bypasses this hint.

The exact cause requires investigation in the ARM CPU plugin source.

**Summary of what each test ruled out:**

| Test | What it ruled out |
|---|---|
| FP32 works on all platforms | Dataset, pipeline, model architecture |
| INT8 breaks on M4 but not M1 (same macOS) | OS version as the cause |
| INT8 breaks on M4 but not M1 (both ARM) | ARM architecture in general |
| All quantized models break on M4 regardless of source | NNCF calibration as the cause |
| M4-quantized model works correctly on M1 and x86 | The model file itself being corrupt |
| No NaN/Inf in output | Memory corruption or numerical overflow |
| `INFERENCE_PRECISION_HINT=f32` has no effect on M4 | This hint as a viable workaround |

The bug is **isolated to M4 Max inference** in OpenVINO's ARM CPU plugin.

---

## Reproduction Steps

### Requirements

```
openvino>=2023.1.0
nncf>=2.6.0
torch
torchvision
tqdm
jupyter
```

### Steps

1. Clone this repository:
   ```bash
   git clone https://github.com/Passavee-Losripat/OpenVINO-ARM-Int-8-Bug.git
   cd OpenVINO-ARM-Int-8-Bug
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the notebook and connect to your kernel:

4. Run all cells in order.

The accuracy result of quantized model compared with original model will be reported in cell 14. The last section requires model quantized in m1 and colab, which is already included in this repository.

---

## Related

- Official notebook: [image-classification-quantization](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/image-classification-quantization)