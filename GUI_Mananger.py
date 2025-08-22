# # Do not touch this GUI_Mananger.py file.
# import os
# import sys
# from pathlib import Path
# sys.path.append(str(Path(__file__).resolve().parent / "bin"))
# current_dir = Path(__file__).resolve().parent
# os.environ["QT_PLUGIN_PATH"] = current_dir.as_posix()
# import ai_bmt_inteface_python as bmt

# def ExecuteGUI(global_interface):
#     global_caller = bmt.AI_BMT_GUI_CALLER(global_interface)
#     args = [sys.argv[0], "--current_dir", current_dir.as_posix()]
#     global_caller.call_BMT_GUI(args)
#     return 0

# GUI_Mananger.py
import os, sys
from pathlib import Path
from collections.abc import Sequence
sys.path.append(str(Path(__file__).resolve().parent / "bin"))
import ai_bmt_interface_python as bmt
current_dir = Path(__file__).resolve().parent
os.environ["QT_PLUGIN_PATH"] = current_dir.as_posix()

def ExecuteGUI(global_interface):
    args = [sys.argv[0], "--current_dir", current_dir.as_posix()]
    # 1) Single interface
    if isinstance(global_interface, bmt.AI_BMT_Interface):
        return bmt.AI_BMT_GUI_CALLER.call_BMT_GUI_For_Single_Task(args, global_interface)

    # 2) Sequence (excluding strings)
    if isinstance(global_interface, Sequence) and not isinstance(global_interface, (str, bytes)):
        interfaces = list(global_interface)
        if len(interfaces) == 0:
            raise ValueError("ExecuteGUI: An empty interface list was provided.")
        if not all(isinstance(x, bmt.AI_BMT_Interface) for x in interfaces):
            raise TypeError("ExecuteGUI: All items in the list must be instances of bmt.AI_BMT_Interface.")
        if len(interfaces) == 1:
            return bmt.AI_BMT_GUI_CALLER.call_BMT_GUI_For_Single_Task(args, interfaces[0])
        else:
            return bmt.AI_BMT_GUI_CALLER.call_BMT_GUI_For_Multiple_Tasks(args, interfaces)
        
    # Type mismatch
    raise TypeError(
        "ExecuteGUI: Argument must be a bmt.AI_BMT_Interface or a sequence (list/tuple, etc.) of them."
    )
