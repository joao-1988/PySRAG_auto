import os
import json
import base64
from datetime import datetime, timedelta
from typing import List

import wget
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


# === CONFIGURATION ===
DRIVE_FOLDER_ID = "1siqO3xH1Q05qCIiLKz6zf3PmHPCqhqdK"
TEMP_DIR = "temp"
CREDENTIALS_ENV_VAR = "GITHUB_TOKEN"


def authenticate_google_drive() -> any:
    """
    Load credentials from a GitHub secret and build the Google Drive service.
    """
    creds_b64 = os.getenv(CREDENTIALS_ENV_VAR)
    if not creds_b64:
        raise EnvironmentError(f"Missing environment variable: {CREDENTIALS_ENV_VAR}")
    
    creds_dict = json.loads(base64.b64decode(creds_b64).decode('utf-8'))
    credentials = service_account.Credentials.from_service_account_info(creds_dict)
    return build('drive', 'v3', credentials=credentials)


def list_drive_csv_files(drive_service) -> List[str]:
    """
    List all CSV file names available in the Google Drive account.
    """
    response = drive_service.files().list(pageSize=1000).execute()
    items = response.get('files', [])
    return [item['name'] for item in items if item.get('mimeType') == 'text/csv']


def download_missing_files(existing_files: List[str]):
    """
    Download CSV files for the last 7 days, if not already uploaded to Drive.
    """
    today = datetime.now().date()
    one_week_ago = today - timedelta(days=7)
    os.makedirs(TEMP_DIR, exist_ok=True)

    for year in range(2022, today.year + 1):
        current_date = one_week_ago
        while current_date <= today:
            filename = f"INFLUD{str(year)[-2:]}-{current_date.strftime('%d-%m-%Y')}.csv"
            if filename not in existing_files:
                url = f"https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SRAG/{year}/{filename}"
                file_path = os.path.join(TEMP_DIR, filename)
                try:
                    wget.download(url, file_path)
                    print(f"\nDownloaded: {filename}")
                except Exception as e:
                    print(f"\nFile not found: {filename} | {str(e)}")
            else:
                print(f"Already exists on Drive: {filename}")
            current_date += timedelta(days=1)


def upload_files_to_drive(drive_service):
    """
    Upload all .csv files from TEMP_DIR to a specific folder on Google Drive.
    """
    for filename in os.listdir(TEMP_DIR):
        if filename.endswith('.csv'):
            filepath = os.path.join(TEMP_DIR, filename)
            print(f"Uploading {filename} to Google Drive...")

            file_metadata = {
                'name': filename,
                'parents': [DRIVE_FOLDER_ID]
            }
            media = MediaFileUpload(filepath, resumable=True)
            try:
                uploaded_file = drive_service.files().create(
                    body=file_metadata, media_body=media, fields='id'
                ).execute()
                print(f"✅ Uploaded {filename} | File ID: {uploaded_file.get('id')}")
            except Exception as e:
                print(f"❌ Failed to upload {filename}: {str(e)}")


def main():
    drive_service = authenticate_google_drive()
    existing_files = list_drive_csv_files(drive_service)
    download_missing_files(existing_files)
    upload_files_to_drive(drive_service)


if __name__ == "__main__":
    main()
