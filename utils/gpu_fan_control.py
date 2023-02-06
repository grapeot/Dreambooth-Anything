"""
This script controls the GPU fans smartly in a headless Linux box, pretty much like what Afterburner does in Windows.

It uses nvidia-smi to read in the temperature, and invokes nvidia-settings to control the fan speed.
The nvidia-settings magic on headless servers come from this post: https://u2pia.medium.com/ubuntu-20-04-nvidia-gpu-control-through-ssh-terminal-bb136f447e11

The script is expected to run with sudo privileges, which are required by the nvidia-settings.
"""
from time import sleep
from typing import List, Callable
from subprocess import Popen, PIPE
from os import environ
import os
import re

# The key magic number to make nvidia-settings in headless mode.
# Consult https://u2pia.medium.com/ubuntu-20-04-nvidia-gpu-control-through-ssh-terminal-bb136f447e11 to get this number properly filled.
GDM_UID = 127
environ['DISPLAY'] = ':0'
environ['XAUTHORITY'] = f'/run/user/{GDM_UID}/gdm/Xauthority'
# In some rare cases, the GPU id is not consecutive. Need to figure out a list.
gpu_ids = [0, 2, 3, 4]

def gpu_fan_function(temperature: int) -> int:
    """
    The core function determining the power of the GPU fan.
    The input is the temperature in Celcius.
    Returns the GPU fan power from 0 to 100.
    """
    if temperature < 40:
        return 30
    if temperature < 60:
        return int(30 + (temperature - 40) * 70 / (60 - 40))
    return 100

def get_gpu_temperature(print_nvidia_smi: bool = True) -> List[int]:
    """
    Get the temperatures of the GPUs, in order, in Celcius.

    When print_nvidia_smi is True, also print the result of nvidia-smi for monitoring.
    """
    proc = Popen('nvidia-smi', stdout=PIPE)
    results = [line.decode('utf-8').strip() for line in proc.stdout]
    temperatures = []
    if print_nvidia_smi:
        os.system('cls' if os.name == 'nt' else 'clear')
    for line in results:
        if print_nvidia_smi:
            print(line)
        match = re.search(' ([0-9]+)C', line)
        if match is not None:
            temperatures.append(int(match.group(1)))
        if line == '':
            # The next is the process list
            break
    return temperatures

def adjust_gpu_fan(gpu_id: int, temperature: int, func: Callable[[int], int]):
    fan_power = func(temperature)
    Popen(['nvidia-settings', '-a', f'[fan:{gpu_id}]/GPUTargetFanSpeed={fan_power}'], stdout=PIPE)

def recover_system_fan_control():
    Popen(['nvidia-settings', '-a', '/GPUFanControlState=0'])

if __name__ == '__main__':
    try:
        Popen(['nvidia-settings', '-a', f'/GPUFanControlState=1'], stdout=PIPE)
        while True:
            temperatures = get_gpu_temperature()
            for i, t in zip(gpu_ids, temperatures):
                adjust_gpu_fan(i, t, gpu_fan_function)
            sleep(2)
    except (SystemExit, KeyboardInterrupt) as ex:
        # graceful shutdown
        recover_system_fan_control()
