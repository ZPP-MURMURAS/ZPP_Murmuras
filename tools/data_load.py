import io
from sys import argv
import os

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account


CREDENTIALS_PATH = ''
DATASETS_PATH = 'datasets/'


def init_service():
    scope = ['https://www.googleapis.com/auth/drive']
    creds = service_account.Credentials.from_service_account_file(
        filename=CREDENTIALS_PATH,
        scopes=scope)
    return build("drive", "v3", credentials=creds)


def map_ds_id(ds_id: str, service):
    results = (
        service.files()
        .list(pageSize=20, fields="nextPageToken, files(id, name)", q=
        "mimeType = 'application/vnd.google-apps.folder'")
        .execute()
    )
    items = results.get("files", [])
    for item in items:
        if item["name"] == ds_id:
            return item["id"]
    raise ValueError(f"no dataset with name {ds_id} found")


def fetch_single_file(file_id: str, file_name: str, service, save_path):
    request = service.files().get_media(fileId=file_id)
    file = io.BytesIO()
    downloader = MediaIoBaseDownload(file, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
    with open(f"{save_path}/{file_name}", "wb") as f:
        f.write(file.getbuffer())


def fetch_ds(ds_internal_id: str, ds_id: str, service):
    if not os.path.exists(f"{os.getcwd()}/{DATASETS_PATH}"):
        os.makedirs(f"{os.getcwd()}/{DATASETS_PATH}")
    ds_path = f"{os.getcwd()}/{DATASETS_PATH}{ds_id}"
    if not os.path.isdir(ds_path):
        os.mkdir(ds_path)
    results = (
        service.files()
        .list(pageSize=100, fields="nextPageToken, files(id, name)", q=f"'{ds_internal_id}' in parents")
        .execute()
    )
    for item in results.get("files", []):
        fetch_single_file(item["id"], item["name"], service, ds_path)


if __name__ == "__main__":
    assert len(argv) == 2, f"usage: {argv[0]} <dataset_id>"
    ds_id = argv[1]
    service = init_service()
    ds_id_internal = map_ds_id(ds_id, service)
    fetch_ds(ds_id_internal, ds_id, service)