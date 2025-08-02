# AI-BMT Platform — Python Submitter Interface
**Last Updated:** 2025-08-03

---

## 1. Environment

- ISA (Instruction Set Architecture): AMD64 (x86_64)
- OS: Windows 10
- Python Version: **3.8.X ~ 3.12.X supported**

---

## 2. Project Description

This version of the AI-BMT Platform allows you to implement your submitter in **Python** by inheriting the provided abstract interface `bmt.AI_BMT_Interface`.  
Once implemented, your model and preprocessing pipeline can be evaluated through the unified GUI interface, just like C++-based submitters.

You can directly modify the **`class SubmitterImplementation(bmt.AI_BMT_Interface)`** in `main.py`.  
We also provide ONNX Runtime-based example scripts for **Classification**, **Object Detection**, and **Semantic Segmentation** in the `example/` folder.

---

## 3. Submitter Development Guide

### Required Interface
submitter **must** subclass `bmt.AI_BMT_Interface` and implement the following methods:
```python
class SubmitterImplementation(bmt.AI_BMT_Interface):
    def Initialize(self, model_path: str) -> None:
        # Load and initialize your model here
        ...

    def convertToPreprocessedDataForInference(self, image_path: str) -> VariantType:
        # Perform image loading and preprocessing here
        ...

    def runInference(self, data: List[VariantType]) -> List[BMTResult]:
        # Perform inference and return results
        ...
```

### Optional Interface

submitter can optionally provide hardware/system metadata using:
```python
class SubmitterImplementation(bmt.AI_BMT_Interface):
    def getOptionalData(self) -> Optional_Data:
        data = Optional_Data()
        data.cpu_type = "Intel i7-9750HF"
        data.accelerator_type = "DeepX M1 (NPU)"
        data.submitter = "DeepX"
        data.cpu_core_count = "16"
        data.cpu_ram_capacity = "32GB"
        data.cooling = "Air"
        data.cooling_option = "Active"
        data.cpu_accelerator_interconnect_interface = "PCIe Gen5 x16"
        data.benchmark_model = "ResNet-50"
        data.operating_system = "Windows 10"
        return data
```
