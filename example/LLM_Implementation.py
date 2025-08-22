import os
import numpy as np
import onnxruntime as ort
from GUI_Mananger import bmt

# LLM용 Submitter (ONNX Runtime)
class SubmitterLLMImplementation(bmt.AI_BMT_Interface):
    def __init__(self):
        super().__init__()
        self.session: ort.InferenceSession | None = None
        self.input_names: list[str] = []
        self.output_names: list[str] = []
        self.modelHasTokenType: bool = False
        self.modelHasAttnMask: bool = False

    def getOptionalData(self):
        optional = bmt.Optional_Data()
        optional.cpu_type = ""
        optional.accelerator_type = ""   # e.g., "DeepX M1(NPU)"
        optional.submitter = ""          # e.g., "DeepX"
        optional.cpu_core_count = ""
        optional.cpu_ram_capacity = ""   # e.g., "32GB"
        optional.cooling = ""            # e.g., "Air"
        optional.cooling_option = ""     # e.g., "Active"
        optional.cpu_accelerator_interconnect_interface = ""  # e.g., "PCIe Gen5 x16"
        optional.benchmark_model = ""
        optional.operating_system = ""
        return optional
    
    def getInterfaceType(self):
        return bmt.InterfaceType.LLM

    def initialize(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        # 필요시 CUDA 등으로 교체 가능: ['CUDAExecutionProvider','CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        # 입력/출력 이름 수집 (C++와 동일하게 전체 수집)
        self.input_names  = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

        # 입력 시그니처 검사
        self.modelHasTokenType = ("token_type_ids"  in self.input_names)
        self.modelHasAttnMask  = ("attention_mask" in self.input_names)
        return True

    def preprocessLLMData(self, llmData: bmt.LLMPreprocessedInput):
        """
        llmData: bmt.LLMPreprocessedInput (fields: input_ids, attention_mask, token_type_ids)
        """
        # pybind로 바인딩된 구조체이므로 필드에 직접 대입 가능
        S = len(llmData.input_ids)
        if self.modelHasAttnMask:
            if len(llmData.attention_mask) != S:
                llmData.attention_mask = [1] * S
        if self.modelHasTokenType:
            if len(llmData.token_type_ids) != S:
                llmData.token_type_ids = [0] * S
        return llmData  # VariantType으로 그대로 전달

    def inferLLM(self, preprocessed_data_list):
        results: list[bmt.BMTLLMResult] = []
        for preprocessed_data in preprocessed_data_list:
            S = len(preprocessed_data.input_ids)
            shape = (1, S)

            # feed 사전 구성 (모델 입력 이름 순회, 필요 없는 입력은 생략)
            feed: dict[str, np.ndarray] = {}
            for nm in self.input_names:
                if nm == "input_ids":
                    feed[nm] = np.asarray(preprocessed_data.input_ids, dtype=np.int64).reshape(shape)
                elif nm == "attention_mask" and self.modelHasAttnMask:
                    # preprocessLLMData에서 보정되었어야 함
                    feed[nm] = np.asarray(preprocessed_data.attention_mask, dtype=np.int64).reshape(shape)
                elif nm == "token_type_ids" and self.modelHasTokenType:
                    feed[nm] = np.asarray(preprocessed_data.token_type_ids, dtype=np.int64).reshape(shape)
                else:
                    pass

            # Fill the result structure
            outs = self.session.run(self.output_names if self.output_names else None, feed)
            r = bmt.BMTLLMResult()
            r.rawOutputShape = [int(x) for x in outs[0].shape]
            r.rawOutput = np.asarray(outs[0], dtype=np.float32).ravel().tolist()
            results.append(r)
        return results
