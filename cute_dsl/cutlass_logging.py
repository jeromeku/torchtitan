import os

def set_logging(log_to_console: bool = False, log_level: int = 10, log_to_file: str = None):
    if log_to_console:
        os.environ["CUTE_DSL_LOG_TO_CONSOLE"] = "1"
    if log_to_file is not None:
        os.environ["CUTE_DSL_LOG_TO_FILE"] = log_to_file

    os.environ["CUTE_DSL_LOG_LEVEL"] = str(log_level)

def set_ir_dump(print_ir: bool = False, keep_ir: bool = True):
    if print_ir:
        os.environ["CUTE_DSL_PRINT_IR"] = "1"
    if keep_ir:
        os.environ["CUTE_DSL_KEEP_IR"] = "1"

def set_arch(arch: str):
    from cutlass.base_dsl import detect_gpu_arch
    os.environ.setdefault("CUTE_DSL_ARCH", detect_gpu_arch(None))