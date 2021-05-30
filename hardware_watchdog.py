import getopt
import sys
from pathlib import Path

import psutil as psutil

from utils.gdrive_utils.gdrive_utils import get_google_drive_results_folder, get_google_drive_file
from utils.hardware_specs import get_system_cpu, CPUStatus, MemoryStatus, LogObject, LogObjectEncoder, \
    create_hardware_log_json
from utils.results import ResultSaver


def main(argv):
    try:
        opts = getopt.getopt(argv, "p:")
    except getopt.GetoptError:
        print('ERROR! Missing Hardware PID. Usage: hardware_watchdog.py -p <PID>')
        sys.exit(2)

    for opt in opts:
        for item in opt:
            param = item[0]
            arg = item[1]
            if param == "-h":
                print("hardware_watchdog.py -p <PID>")
                sys.exit()
            elif param == '-p':
                execute_watchdog(int(arg))


def execute_watchdog(pid):
    hardware_data = get_system_cpu()

    while psutil.pid_exists(pid):
        log_array = hardware_data['logs']
        tmp_cpu = CPUStatus()
        tmp_memory = MemoryStatus()
        tmp_log_object = LogObject(tmp_cpu, tmp_memory)
        string_log = LogObjectEncoder().encode(tmp_log_object)
        fixed_log = eval(string_log)
        log_array.append(fixed_log)
        print("process exists...recording data...")
    print("process has finished...")
    results_path = ResultSaver.results_folder_path
    create_hardware_log_json(results_path, hardware_data)
    print("Woof! Woof! The Hardware Watchdog has completed result gathering.")
    append_results_gdrive(results_path)


def append_results_gdrive(path):
    dog_path = path.joinpath("results_path.dog")
    try:
        dog_file = open(str(dog_path))
    except Exception:
        print("ERROR! .dog file not found in results. Was it generated? (File: hardware_watchdog.py)")

    try:
        gdrive_time = dog_file.read()
        results = get_google_drive_results_folder()
        gdrive_user_folder = get_google_drive_file(Path.home().name, results, is_child_a_folder=True)
        gdrive_time_folder = get_google_drive_file(gdrive_time, gdrive_user_folder, is_child_a_folder=True)
        hardware_results = get_google_drive_file("hardware_monitor.json", gdrive_time_folder, is_child_a_folder=False)

        hardware_results.SetContentFile(str(path.joinpath('hardware_monitor.json')))

        hardware_results.Upload()
    except Exception as e:
        print(e.__traceback__)

    input()

    try:
        dog_file.close()
        os.remove(str(dog_path))
    except Exception as e:
        print(e.__traceback__)

    input()


if __name__ == '__main__':
    main(sys.argv[1:])
