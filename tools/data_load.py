import io
from pydoc_data.topics import topics
from sys import argv
import os
from typing import Tuple, DefaultDict, List, Dict, Set
from collections import defaultdict

from googleapiclient.discovery import build, Resource
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account


CREDENTIALS_PATH = os.path.realpath(__file__).rsplit('/', 1)[0] + '/gdrive_credentials.json'
DATASETS_PATH = os.path.realpath(__file__).rsplit('/', 2)[0] + '/datasets/'


def init_service() -> Resource:
    scope = ['https://www.googleapis.com/auth/drive']
    creds = service_account.Credentials.from_service_account_file(
        filename=CREDENTIALS_PATH,
        scopes=scope)
    return build("drive", "v3", credentials=creds)


def create_fs_tree(service: Resource) ->(
        Tuple)[DefaultDict[str, List[str]], DefaultDict[str, str], Dict[str, str]]:
    """
    for mapping filesystem on drive:
    :param service: google drive service
    :return: tuple containing:
    map from folder to subfolders,
    map from subfolder to parent,
    set of top level folder ids,
    map from folder ids to folder names
    """
    children = defaultdict(list)
    parents = defaultdict(str)
    folder_names = {}

    folders = (
        service.files()
        .list(pageSize=20, fields="nextPageToken, files(id, name, parents)", q=
        "mimeType = 'application/vnd.google-apps.folder'")
        .execute()
    )

    for folder in folders.get("files", []):
        name = folder['name']
        folder_id = folder['id']
        parent = folder['parents'][0] if 'parents' in folder else None
        folder_names[folder_id] = name

        if folder_id not in children.keys():
            children[folder_id] = []

        if parent is not None:
            parents[folder_id] = parent
            children[parent].append(folder_id)

    return children, parents, folder_names


def fetch_single_file(file_id: str, file_name: str, service: Resource, save_path):
    request = service.files().get_media(fileId=file_id)
    file = io.BytesIO()
    downloader = MediaIoBaseDownload(file, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
    with open(f"{save_path}/{file_name}", "wb") as f:
        f.write(file.getbuffer())


def fetch_ds(ds_internal_id: str, service, fs_tree: Dict[str, List[str]], name_mapping: Dict[str, str]):
    if not os.path.exists(f"{DATASETS_PATH}"):
        os.makedirs(f"{DATASETS_PATH}")
    path_mapping = {ds_internal_id: f"{DATASETS_PATH}{ds_id}"}
    to_visit = [ds_internal_id]

    while to_visit:
        folder_id = to_visit.pop()
        if not os.path.isdir(path_mapping[folder_id]):
            os.makedirs(path_mapping[folder_id])
        results = (
            service.files()
            .list(fields="nextPageToken, files(id, name)",
                  q=f"'{folder_id}' in parents and not mimeType='application/vnd.google-apps.folder'")
            .execute()
        )
        for item in results.get("files", []):
            fetch_single_file(item["id"], item["name"], service, path_mapping[folder_id])
        for child in fs_tree[folder_id]:
            path_mapping[child] = f"{path_mapping[folder_id]}/{name_mapping[child]}"
            to_visit.append(child)


if __name__ == "__main__":
    assert len(argv) == 2, f"usage: {argv[0]} <dataset_id>"
    ds_id = argv[1]
    service = init_service()
    children, parent, folder_names = create_fs_tree(service)
    root_folder_id = None
    for folder_id in folder_names.keys():
        if folder_id not in parent:
            root_folder_id = folder_id
    assert root_folder_id is not None, "could not find root folder on google drive - this is strange"
    top_lvl_dirs = children[root_folder_id]
    ds_id_internal = None
    for folder_id in top_lvl_dirs:
        if folder_names[folder_id] == ds_id:
            ds_id_internal = folder_id
    assert ds_id_internal is not None, f"could not find dataset named {ds_id}"
    fetch_ds(ds_id_internal, service, children, folder_names)
