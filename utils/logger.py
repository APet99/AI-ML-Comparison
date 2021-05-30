from datetime import datetime as dt
from pathlib import Path


def get_repository_root_folder_path() -> Path:
    """
    Will look for the '.git' directory to assuming that submodules are not used

    :return: Path object with the absolute path to the root of the repository
    """
    path = Path(__file__).parent
    while '.git' not in [str(x.name) for x in path.iterdir()]:
        path = path.parent
    return path


def log(message: str, dir: Path = None, filename: 'str' = 'log.txt', write_file=True, write_console=True):
    content = f'{dt.now().replace(microsecond=0)} \t {message}'

    if write_console:
        print(content)

    if write_file:
        if dir is None:
            dir = get_repository_root_folder_path().joinpath('results')

        if not dir.exists():
            dir.mkdir(parents=True, exist_ok=True)

        with open(dir / filename, 'a+') as file:
            file.writelines(content + '\n')
            file.close()
