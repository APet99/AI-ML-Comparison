import json
import platform
from datetime import datetime
from pathlib import Path

import psutil
from cpuinfo import get_cpu_info


def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


def get_system_cpu():
    uname = platform.uname();
    cpu = get_cpu_info()

    data = {}
    data['hostname'] = uname.node
    data['system'] = f"{uname.system} {uname.release}"
    data['processor'] = cpu['brand_raw']
    data['architecture'] = platform.architecture()[0]
    data['physical_cores'] = psutil.cpu_count(False)
    data['total_cores'] = psutil.cpu_count(True)
    data['logs'] = []

    return data


def get_system_information():
    uname = platform.uname()
    cpu = get_cpu_info()
    vir_mem = psutil.virtual_memory()

    data = {}
    data['time'] = datetime.today().strftime("%m/%d/%Y %H:%M:%S")
    data['hostname'] = uname.node
    data['system'] = f"{uname.system} {uname.release}"
    data['processor'] = cpu['brand_raw']
    data['architecture'] = platform.architecture()[0]
    data['physical_cores'] = psutil.cpu_count(False)
    data['total_cores'] = psutil.cpu_count(True)
    data['frequency'] = psutil.cpu_freq().current
    data['cores'] = []
    for i, percentage in enumerate(psutil.cpu_percent(True, 1)):
        data['cores'].append({
            'core': i + 1,
            'usage': str(percentage) + "%"
        })
    data['ram_available'] = get_size(vir_mem.available)
    data['ram_used'] = get_size(vir_mem.used)
    data['ram_total'] = get_size(vir_mem.total)

    return data


def create_system_info_json(results_path: Path):
    system_info = get_system_information()
    json_path = results_path / "device.json"
    json_path.write_text(json.dumps(system_info, indent=4))


def create_hardware_log_json(results_path: Path, data: dict):
    json_path = results_path / "hardware_monitor.json"
    json_path.write_text(json.dumps(data, indent=4))


class MemoryStatus:
    def __init__(self, free, used, total):
        self.free = free
        self.used = used
        self.total = total

    def __init__(self):
        vir_mem = psutil.virtual_memory()
        self.free = get_size(vir_mem.available)
        self.used = get_size(vir_mem.used)
        self.total = get_size(vir_mem.total)


class CPUStatus:
    def __init__(self, cores: list, frequency):
        self.cores = cores
        self.frequency = frequency

    def __init__(self):
        cpu = get_cpu_info()
        self.cores = []
        for i, percentage in enumerate(psutil.cpu_percent(True, 1)):
            new_core = Core(i + 1, str(percentage) + '%')
            self.cores.append(new_core)
        self.frequency = cpu["hz_actual"][0]


class Core:
    def __init__(self, number: int, usage: str):
        self.number = number
        self.usage = usage


class LogObject:
    def __init__(self, cpu: CPUStatus, memory: MemoryStatus):
        self.cpu = cpu
        self.memory = memory
        self.time = datetime.now().strftime("%m/%d/%Y %H:%M:%S")

    def created_at(self):
        return self.time

    # def toJson(self):
    #     return json.dumps(self, default=lambda o: o.__dict__)


class LogObjectEncoder(json.JSONEncoder):
    def default(self, o):
        return o.__dict__
