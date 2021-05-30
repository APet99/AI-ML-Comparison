import os
from pathlib import Path

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from pydrive.files import GoogleDriveFile

from utils.logger import log

__cwd = os.getcwd()  # stores CWD because a change is about to occur to have the correct 'client_secrets.json'
os.chdir(str(Path(__file__).parent))
GoogleAuth.DEFAULT_SETTINGS['client_config_file'] = 'client_secrets.json'
__google_auth = GoogleAuth()  # initiates O-Auth
__google_auth.LocalWebserverAuth()  # Creates local webserver and auto handles O-Auth
__drive = GoogleDrive(__google_auth)
os.chdir(__cwd)  # Change back to CWD after O-Auth


class NotUniqueException(Exception):
    pass


def get_google_drive() -> GoogleDrive:
    """
    Singleton pattern used to get the Google Drive Object to access the ItsClassified Google Drive

    :return: Google Drive object with O-Auth access to Google Drive of ItsClassified Account
    """
    return __drive


def get_repository_root_folder_path() -> Path:
    """
    Will look for the '.git' directory to assuming that submodules are not used

    :return: Path object with the absolute path to the root of the repository
    """
    path = Path(__file__).parent
    while '.git' not in [str(x.name) for x in path.iterdir()]:
        path = path.parent
    return path.absolute()


def get_google_drive_results_folder() -> GoogleDriveFile:
    """
    Gets the results folder object in the ItsClassified Google Drive

    :return: GoogleDriveFile object of the results folder in the ItsClassified Google Drive
    """
    return get_google_drive_file(file_name='results', is_file_a_folder=True)


def get_google_drive_file(file_name: str, gdrive_parent_folder: GoogleDriveFile = None,
                          create_if_not_exists: bool = True,
                          is_file_a_folder: bool = False, ) -> GoogleDriveFile:
    """
    Retrieves a GoogleDriveFile object of the specified file or creates the specified file if it does not exist.

    :param file_name: the name of the specified file that is wished to be retrieved (str)
    :param gdrive_parent_folder: parent folder of the specified file, if the folder is located within root then the
    param should be None (GoogleDriveFile or None)
    :param create_if_not_exists: creat the specified file if the file does not exist (bool)
    :param is_file_a_folder: if the specified folder to be created is supposed to be a folder (bool)

    :return: GoogleDriveFile object of the specified file
    """
    mimetypes = 'mimeType=\'application/vnd.google-apps.folder\' and ' if is_file_a_folder else ''
    folder_list = __drive.ListFile({
        'q': f"title='{file_name}' and "
             f"'{gdrive_parent_folder['id'] if gdrive_parent_folder else 'root'}' "
             f"in parents and {mimetypes}"
             f"trashed=false"}).GetList()
    if len(folder_list) > 1:
        raise NotUniqueException(
            f'There should only be one folder in the google drive labeled {file_name}')
    elif len(folder_list) < 1 and create_if_not_exists:
        if is_file_a_folder:
            folder_list.append(
                __drive.CreateFile({'title': file_name,
                                    "mimeType": "application/vnd.google-apps.folder",
                                    'parents': [
                                        {'id': gdrive_parent_folder['id'] if gdrive_parent_folder else 'root'}]}))
        else:
            folder_list.append(
                __drive.CreateFile({'title': file_name,
                                    'parents': [
                                        {'id': gdrive_parent_folder['id'] if gdrive_parent_folder else 'root'}]}))

    return folder_list[0]


def download_gdrive_file(gdrive_file: GoogleDriveFile, path: Path):
    """
    Downloads a specific google drive file to a specified path given. The newly downloaded file will have the same name
    as the google drive file.

    :param gdrive_file: specified google drive file to download (GoogleDriveFile)
    :param path: absolute path of where to place the newly downloaded file (Path)
    :return: None but the newly downloaded file will be located at specified path
    """
    gdrive_file.GetContentFile(f'{path}/{gdrive_file["title"]}')


def download_gdrive_folder(gdrive_folder: GoogleDriveFile, local_path: Path = get_repository_root_folder_path()):
    """
    Downloads a specific google drive folder to a specified path given. The newly downloaded folder will have the same
    name as the google drive folder. Will download all the content of the folder, even empty folders.

    :param gdrive_folder: specified google drive folder to download (GoogleDriveFile)
    :param local_path: absolute path of where to place the newly downloaded folder (Path)
    :return: None but the newly downloaded folder will be located at specified path
    """
    local_path.mkdir(parents=True, exist_ok=True)
    file_list = __drive.ListFile(
        {'q': f"'{gdrive_folder['id']}' in parents and trashed=false"}).GetList()
    for i, file in enumerate(sorted(file_list, key=lambda x: x['title']), start=1):
        if file['mimeType'] != 'application/vnd.google-apps.folder':
            log(f"Downloading {gdrive_folder['title']}/{file['title']} from Google Drive ({i}/{len(file_list)})")
            download_gdrive_file(gdrive_file=file, path=local_path)
        else:
            download_gdrive_folder(file, local_path.joinpath(file['title']))
