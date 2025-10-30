# AI-BMT Platform — Python Submitter Interface (Added LLM Tasks(HellaSwag, MMLU))
**Last Updated:** 2025-10-31

---

## 1. Environment

- ISA (Instruction Set Architecture): AMD64 (x86_64)
- OS: Windows 10
- Python Version: **3.8.X ~ 3.13.X supported**

---

## 2. Project Description
1. Implement AI_BMT_Interface to operate with the intended AI Processing Unit (e.g., CPU, GPU, NPU).
2. Various task example codes are provided. Use these example codes as a reference to implement the interface for the AI Processing Unit.

---

## 3. Submitter Development Guide

### Required Interface
submitter **must** subclass `bmt.AI_BMT_Interface` and implement the following methods:
```python
class SubmitterImplementation(bmt.AI_BMT_Interface):

    # Load and initialize your model here
    def initialize(self, model_path: str) -> None:

    # return the implemented interface task type. 
    def getInterfaceType(self) -> InterfaceType:

    #  Vision tasks: preprocessing & inference
    #  - preprocessVisionData: convert raw image file into model input format
    #  - inferVision: run inference on preprocessed data and return vision model outputs
    #  - dataTransferVision : transfer vision model outputs to BMT result format
    def preprocessVisionData(self, image_path: str) -> VariantType:
    def inferVision(self, data: List[VariantType]) -> model_outputs:
    def dataTransferVision(self, model_outputs) -> List[BMTVisionResult]:

    # LLM tasks: preprocessing & inference
    # - preprocessLLMData: convert raw input into model input format
    # - inferLLM: run inference on preprocessed data and return LLM model outputs
    # - dataTransferLLM : transfer LLM model outputs to BMT result format
    def preprocessLLMData(self, llmData: LLMPreprocessedInput) -> VariantType:
    def inferLLM(self, data: List[VariantType]) -> model_outputs:
    def dataTransferLLM(self, model_outputs) -> List[BMTLLMResult]:  
        
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
