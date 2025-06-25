from pathlib import Path
import os
import sys
sys.path.append(str(Path(__file__).resolve().parent / "bin"))
import ai_bmt_inteface_python as bmt

# Python에서 AI_BMT_Interface 상속하여 구현
class PySegmentationInterface(bmt.AI_BMT_Interface):
    def Initialize(self, model_path: str):
        print(f"[Python] Initializing model from {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        # 모델 초기화 로직 추가 (예: ONNX 모델 로드 등)
        return True
    
    def convertToPreprocessedDataForInference(self, image_path: str):
        print(f"[Python] Preprocessing {image_path}")
        dummy = [0.1] * (3 * 520 * 520)  
        return dummy

    def runInference(self, preprocessed_data):
        print(f"[Python] Inference on data of size {len(preprocessed_data)}")
        result = bmt.BMTResult()
        result.segmentationResult = [0.1] * (520 * 520 * 21)  # 더미
        return [result]

global_interface = None
global_caller = None
def main():
    global global_interface, global_caller
    
    global_interface = PySegmentationInterface()
    current_dir = Path(__file__).resolve().parent
    model_path = current_dir / "Model" / "Segmentation" / "deeplabv3_mobilenet_v3_large_opset12.onnx"
    model_path = model_path.as_posix()

    # Do not touch the following lines
    os.environ["QT_PLUGIN_PATH"] = current_dir.as_posix()
    global_caller = bmt.AI_BMT_GUI_CALLER(global_interface, model_path)
    args = [sys.argv[0], "--current_dir", current_dir.as_posix()]
    global_caller.call_BMT_GUI(args)
    return 0

if __name__ == "__main__":
    main()