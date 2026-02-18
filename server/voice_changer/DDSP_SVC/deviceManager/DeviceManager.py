import torch


class DeviceManager(object):
    _instance = None

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
            dev = torch.device("cuda", index=id)
        return dev

    def halfPrecisionAvailable(self, id: int):
        if self.gpu_num == 0:
            return False
        if id < 0:
            return False

        try:
            gpuName = torch.cuda.get_device_name(id).upper()
            
            # Check for AMD GPUs (ROCm support)
            # AMD GPUs with ROCm generally support half precision
            if "AMD" in gpuName or "RADEON" in gpuName:
                # AMD 6000 series and newer support half precision well
                # This includes 6800XT, 6900XT, 7000 series, etc.
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
            if cap[0] < 7:
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
            # except Exception as e:
        except:  # NOQA
            # print(e)
            return 0
