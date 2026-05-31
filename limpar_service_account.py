import os
import json
import base64
from google.oauth2 import service_account
from googleapiclient.discovery import build

def limpar_tudo():
    creds_b64 = os.getenv("GITHUB_TOKEN")
    creds_dict = json.loads(base64.b64decode(creds_b64).decode('utf-8'))
    credentials = service_account.Credentials.from_service_account_info(creds_dict)
    drive_service = build('drive', 'v3', credentials=credentials)

    print("Buscando arquivos na Service Account...")
    # Traz todos os arquivos que pertencem à Service Account
    response = drive_service.files().list(q="'me' in owners", pageSize=1000, fields="files(id, name)").execute()
    arquivos = response.get('files', [])

    if not arquivos:
        print("Nenhum arquivo encontrado. A cota já deve estar limpa.")
        return

    for arquivo in arquivos:
        try:
            drive_service.files().delete(fileId=arquivo['id']).execute()
            print(f"Deletado: {arquivo['name']}")
        except Exception as e:
            print(f"Erro ao deletar {arquivo['name']}: {e}")
            
    print("Limpeza concluída! A Service Account agora tem 15GB livres novamente.")

if __name__ == "__main__":
    limpar_tudo()