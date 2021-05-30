import shutil
import unittest
from datetime import datetime
from pathlib import Path

from utils.gdrive_utils.gdrive_utils import get_google_drive, get_repository_root_folder_path, \
    get_google_drive_results_folder, get_google_drive_file, NotUniqueException, download_gdrive_file, \
    download_gdrive_folder


class TestGDriveUtils(unittest.TestCase):

    def test_get_google_drive(self):
        # given
        google_drive_account_name = 'CBU ItsClassified'

        # when
        gdrive_obj = get_google_drive()

        # then
        self.assertTrue(gdrive_obj.auth, msg="If O-Auth was given .auth property should have some value.")
        self.assertEqual(hex(id(gdrive_obj)), hex(id(get_google_drive())),
                         msg="If singleton should have same memory address")
        self.assertTrue(gdrive_obj.GetAbout()["name"] == google_drive_account_name,
                        msg=f"Should be signed in with the '{google_drive_account_name}' account.")

    def test_get_repository_root_folder_path(self):
        # given
        repo_name = "ItsClassified-Implementation"

        # when
        repo_root_path = get_repository_root_folder_path()

        # then
        self.assertTrue(repo_root_path.is_absolute(), msg='given path object should hold an absolute path.')
        self.assertEqual(repo_root_path.name, repo_name,
                         msg='path should be at the root of the repository, for this repo it should be'
                             f' "{repo_name}."')

    def test_get_google_drive_results_folder(self):
        # given
        folder_name = 'results'
        file_type = 'application/vnd.google-apps.folder'
        owner_name = 'CBU ItsClassified'

        # when
        result_folder = get_google_drive_results_folder()

        # then
        self.assertEqual(result_folder['title'], folder_name,
                         msg=f'the folder name should be the same as {folder_name}')
        self.assertEqual(result_folder['mimeType'], file_type, msg='file should be a folder')
        self.assertTrue(result_folder['parents'][0]['isRoot'], msg='folder should be a child of root')
        self.assertEqual(result_folder['ownerNames'][0], owner_name, msg=f'owner should be {owner_name}')

    def test_get_google_drive_file_normal_folder(self):
        # given
        folder_name = 'models'
        file_type = 'application/vnd.google-apps.folder'
        owner_name = 'CBU ItsClassified'

        # when
        model_folder = get_google_drive_file(file_name='models', is_file_a_folder=True)

        # then
        self.assertEqual(model_folder['title'], folder_name,
                         msg=f'the folder name should be the same as {folder_name}')
        self.assertEqual(model_folder['mimeType'], file_type, msg='file should be a folder')
        self.assertTrue(model_folder['parents'][0]['isRoot'], msg='folder should be a child of root')
        self.assertEqual(model_folder['ownerNames'][0], owner_name, msg=f'owner should be {owner_name}')

    def test_get_google_drive_file_exception(self):
        # given
        folder_name = 'testing'
        testing_folder = get_google_drive_file(file_name=folder_name, is_file_a_folder=True)

        # when
        try:
            get_google_drive_file(file_name='Test1', gdrive_parent_folder=testing_folder, is_file_a_folder=True)
            self.fail('Should have thrown a "NotUniqueException."')
        except NotUniqueException:
            # then
            pass

    def test_get_google_drive_file_create_folder(self):
        # given
        folder_name = f'{datetime.now().strftime("%c")}'
        file_type = 'application/vnd.google-apps.folder'
        owner_name = 'CBU ItsClassified'

        # when
        created_folder = get_google_drive_file(file_name=folder_name, is_file_a_folder=True, create_if_not_exists=True)

        # then
        created_folder.Upload()
        self.assertEqual(created_folder['title'], folder_name,
                         msg=f'the folder name should be the same as {folder_name}')
        self.assertEqual(created_folder['mimeType'], file_type, msg='file should be a folder')
        self.assertTrue(created_folder['parents'][0]['isRoot'], msg='folder should be a child of root')
        self.assertEqual(created_folder['ownerNames'][0], owner_name, msg=f'owner should be {owner_name}')

        # finally
        created_folder.Delete()

    def test_get_google_drive_file_create_file(self):
        # given
        file_name = f'{datetime.now().strftime("%c")}.txt'
        file_type = 'text/plain'
        owner_name = 'CBU ItsClassified'

        # when
        created_file = get_google_drive_file(file_name=file_name, is_file_a_folder=False, create_if_not_exists=True)

        # then
        created_file.Upload()
        self.assertEqual(created_file['title'], file_name,
                         msg=f'the folder name should be the same as {file_name}')
        self.assertEqual(created_file['mimeType'], file_type, msg='should be a plain text file: "text/plain"')
        self.assertTrue(created_file['parents'][0]['isRoot'], msg='folder should be a child of root')
        self.assertEqual(created_file['ownerNames'][0], owner_name, msg=f'owner should be {owner_name}')

        # finally
        created_file.Delete()

    def test_download_gdrive_file(self):
        # given
        file_name = 'test.txt'
        testing_folder = get_google_drive_file(file_name='testing', is_file_a_folder=True)
        created_file = get_google_drive_file(file_name=file_name, gdrive_parent_folder=testing_folder,
                                             is_file_a_folder=False, create_if_not_exists=False)
        local_parent_dir = Path('./test_resource')
        local_parent_dir.mkdir(parents=True, exist_ok=True)
        file_content = 'Hello World'

        # when
        download_gdrive_file(created_file, local_parent_dir)

        # then
        self.assertTrue(local_parent_dir.joinpath(file_name).exists())
        with open(local_parent_dir.joinpath(file_name)) as file:
            self.assertEqual(file.read().strip(), file_content, msg=f'Content of file should be "{file_content}"')

        # finally
        shutil.rmtree(local_parent_dir)

    def test_download_gdrive_folder(self):
        # given
        folder_name = 'testing'
        testing_folder = get_google_drive_file(file_name=folder_name, is_file_a_folder=True, create_if_not_exists=False)
        local_parent_dir = Path('./test_resource')
        local_parent_dir.mkdir(parents=True, exist_ok=False)
        local_parent_children = ['Test1', 'test.txt']

        # when
        download_gdrive_folder(testing_folder, local_parent_dir)

        # then
        self.assertTrue(local_parent_dir.exists())
        for i, file in enumerate(local_parent_dir.iterdir()):
            self.assertEqual(file.name, local_parent_children[i],
                             msg=f"children should be in order of {local_parent_dir}")

        # finally
        shutil.rmtree(local_parent_dir)


if __name__ == '__main__':
    unittest.main()
