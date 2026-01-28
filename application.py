# Carregar pacotes
import os
import numpy as np
import pandas as pd
from pysrag.data import SRAG
from pysrag.model import GBMTrainer
from joblib import dump, load
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from datetime import datetime

import base64
import json

# Função para baixar um arquivo
def download_file(file_id, filepath, service):
    try:
        request = service.files().get_media(fileId=file_id)
        with open(filepath, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                print(f"Download {int(status.progress() * 100)}%.")
    except HttpError as error:
        print(f"An error occurred: {error}")

# Obter a data mais recente de cada ano
def get_latest_data(all_files,string_year):
  format_string = '%d-%m-%Y'
  list_files = [i.replace(f'{string_year}-','').split('.')[0] for i in all_files if i[:8] == string_year]
  if not list_files:
      return None
  latest_data = max(list_files,key=lambda x: datetime.strptime(x, format_string))
  string_latest_data = string_year+'-'+latest_data+'.csv'
  return string_latest_data

# Configurar o caminho para as credenciais
creds_env = os.getenv('GOOGLE_CREDS')
creds_json = json.loads(base64.b64decode(creds_env).decode('utf-8'))
creds = service_account.Credentials.from_service_account_info(creds_json)
drive_service = build('drive', 'v3', credentials=creds)

# Listar arquivos csv 
results = drive_service.files().list(pageSize=1000).execute()
items = results.get('files', [])
all_files = [item['name'] for item in items if item['mimeType'] == 'text/csv']

# Obter os nomes dos arquivos mais recentes
filepath = []
today = datetime.now().date()
for year in range(2022, today.year + 1):
    influd = get_latest_data(all_files,'INFLUD'+str(year)[2:])
    print(influd)
    if influd is not 'None':
        filepath.append(influd)

files_saved = pd.read_csv('https://raw.githubusercontent.com/joao-1988/PySRAG_auto/refs/heads/main/filename.csv').filename.to_list()

if set(files_saved) != set(filepath):
    
    # Obter os ids dos arquivos e baixa
    file_map = {item['name']: item['id'] for item in items} # Mapeia name-to-id
    for influd in filepath:
        influd_id = file_map.get(influd)
        if influd_id:
            print(influd, influd_id)
            download_file(influd_id, influd, drive_service)
        else:
            print(f"Warning: Arquivo '{influd}' não encontrado.")

    # Carregar os dados
    cols_X = ['REGIAO_LATITUDE', 'REGIAO_LONGITUDE', 'UF_LATITUDE'
            , 'UF_LONGITUDE', 'LATITUDE', 'LONGITUDE', 'POPULACAO', 'IDADE_ANO'
            , 'ANO_SEM_SIN_PRI']
    col_y = ['POS_SARS2', 'POS_FLUA', 'POS_FLUB', 'POS_VSR',
             'POS_PARA1', 'POS_PARA2', 'POS_PARA3', 'POS_PARA4',
             'POS_ADENO', 'POS_METAP', 'POS_BOCA', 'POS_RINO', 'POS_OUTROS']
    
    print(filepath)
    srag = SRAG(filepath)
    X, y = srag.generate_training_data('multiclass', cols_X, col_y)
    
    weeks = np.unique(X['ANO_SEM_SIN_PRI'])
    train = srag.get_start_day_of_week(0)
    df_training_weeks = srag.generate_training_weeks()
    
    for influd in filepath:
      if os.path.exists(influd):
        os.remove(influd)   
    
    # Treinar o modelo
    trainer = GBMTrainer(objective='multiclass', eval_metric='multi_logloss')
    trainer.fit(X, y)
    
    # Salvar o modelo
    model = {'filename': filepath,
             'weeks': weeks,
             'train': train,
              'virus': trainer.model.classes_,
              'df_training_weeks': df_training_weeks,
              'model': trainer,
              'train_size': len(y),
              'best_boost_iteration': trainer.best_iteration}
    
    dump(model,'./dict_model')
    
    # Exclui semanas com lag 0 e 1
    excl_weeks = []
    for lag in [0,1]:
      dict_lag = srag.get_start_day_of_week(lag)
      excl_weeks.append( dict_lag['year']*100 + dict_lag['week'] )
    
    ind_excl_weeks = df_training_weeks['ANO_SEM_SIN_PRI'].isin(excl_weeks)
    
    df_training_weeks_app = df_training_weeks[~ind_excl_weeks].query('SEM_SIN_PRI > 0')
    
    # Salvar os dados
    df_training_weeks_app.to_csv('./df_semanas.csv',index=False)
    SRAG.load_common_data().to_csv('./df_municipios.csv',index=False)
    feature_name = model['model'].model.feature_name_
    pd.Series(feature_name,name='feature_name').to_csv('./feature_name.csv',index=False)
    classes = model['virus']
    pd.Series(classes,name='virus').to_csv('./classes.csv',index=False)
    model['model'].model.booster_.save_model('./booster.txt')
    pd.Series(filepath,name='filename').to_csv('./filename.csv',index=False)


