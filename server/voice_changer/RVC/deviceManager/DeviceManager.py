import torch
import onnxruntime


class DeviceManager(object):
    _instance = None
    forceTensor: bool = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.gpu_num = torch.cuda.device_count()
        self.mps_enabled: bool = (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        )

    def getDevice(self, id: int):
        if id < 0 or self.gpu_num == 0:
            if self.mps_enabled is False:
                dev = torch.device("cpu")
            else:
                dev = torch.device("mps")
        else:
            if id < self.gpu_num:
                dev = torch.device("cuda", index=id)
            else:
                print("[Voice Changer] device detection error, fallback to cpu")
                dev = torch.device("cpu")
        return dev

    def getOnnxExecutionProvider(self, gpu: int):
        availableProviders = onnxruntime.get_available_providers()
        devNum = torch.cuda.device_count()
        
        if gpu >= 0 and devNum > 0:
            if gpu < devNum:
                # Check for ROCMExecutionProvider first (AMD GPUs)
                if "ROCMExecutionProvider" in availableProviders:
                    return ["ROCMExecutionProvider"], [{"device_id": gpu}]
                # Fall back to CUDAExecutionProvider (works with both CUDA and ROCm)
                elif "CUDAExecutionProvider" in availableProviders:
                    return ["CUDAExecutionProvider"], [{"device_id": gpu}]
            else:
                print("[Voice Changer] device detection error, fallback to cpu")
                return ["CPUExecutionProvider"], [
                    {
                        "intra_op_num_threads": 8,
                        "execution_mode": onnxruntime.ExecutionMode.ORT_PARALLEL,
                        "inter_op_num_threads": 8,
                    }
                ]
        elif gpu >= 0 and "DmlExecutionProvider" in availableProviders:
            return ["DmlExecutionProvider"], [{"device_id": gpu}]
        
        return ["CPUExecutionProvider"], [
            {
                "intra_op_num_threads": 8,
                "execution_mode": onnxruntime.ExecutionMode.ORT_PARALLEL,
                "inter_op_num_threads": 8,
            }
        ]

    def setForceTensor(self, forceTensor: bool):
        self.forceTensor = forceTensor

    def halfPrecisionAvailable(self, id: int):
        if self.gpu_num == 0:
            return False
        if id < 0:
            return False
        if self.forceTensor:
            return False

        is_amd = False
        try:
            gpuName = torch.cuda.get_device_name(id).upper()
            
            # Check for AMD GPUs (ROCm support)
            # AMD GPUs with ROCm generally support half precision
            if "AMD" in gpuName or "RADEON" in gpuName:
                # AMD 6000 series and newer support half precision well
                # This includes 6800XT, 6900XT, 7000 series, etc.
                is_amd = True
                return True
            
            # NVIDIA GPU checks
            if (
                ("16" in gpuName and "V100" not in gpuName)
                or "P40" in gpuName.upper()
                or "1070" in gpuName
                or "1080" in gpuName
            ):
                return False
        except Exception as e:
            print(e)
            return False

        try:
            cap = torch.cuda.get_device_capability(id)
            if cap[0] < 7:  # コンピューティング機能が7以上の場合half precisionが使えるとされている（が例外がある？T500とか）
                return False
        except Exception as e:
            # ROCm may not support get_device_capability in the same way
            # For AMD GPUs, we already returned True above
            # For NVIDIA GPUs, if we can't get capability, assume it's not supported
            print(f"[Voice Changer] Could not get device capability: {e}")
            return False

        return True

    def getDeviceMemory(self, id: int):
        try:
            return torch.cuda.get_device_properties(id).total_memory
        except Exception as e:
            # except:
            print(e)
            return 0
