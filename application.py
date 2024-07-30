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
  latest_data = max(list_files,key=lambda x: datetime.strptime(x, format_string))
  string_latest_data = string_year+'-'+latest_data+'.csv'
  return string_latest_data

# Configurar o caminho para as credenciais
#creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')
creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
creds = service_account.Credentials.from_service_account_file(creds_path)
drive_service = build('drive', 'v3', credentials=creds)

# Listar arquivos csv 
results = drive_service.files().list(pageSize=1000).execute()
items = results.get('files', [])
all_files = [item['name'] for item in items if item['mimeType'] == 'text/csv']

# Obter os arquivos mais recentes e seus respectivos ids
influd22 = get_latest_data(all_files,'INFLUD22')
influd23 = get_latest_data(all_files,'INFLUD23')
influd24 = get_latest_data(all_files,'INFLUD24')
print(influd22, influd23, influd24)
influd22_id = [item['id'] for item in items if item['name'] == influd22][0]
influd23_id = [item['id'] for item in items if item['name'] == influd23][0]
influd24_id = [item['id'] for item in items if item['name'] == influd24][0]
print(influd22_id, influd23_id, influd24_id)

# Baixar os arquivos
download_file(influd22_id,influd22,drive_service)
download_file(influd23_id,influd23,drive_service)
download_file(influd24_id,influd24,drive_service)

# Carregar os dados
cols_X = ['REGIAO_LATITUDE', 'REGIAO_LONGITUDE', 'UF_LATITUDE'
        , 'UF_LONGITUDE', 'LATITUDE', 'LONGITUDE', 'POPULACAO', 'IDADE_ANO'
        , 'ANO_SEM_SIN_PRI']
col_y = ['POS_SARS2', 'POS_FLUA', 'POS_FLUB', 'POS_VSR',
         'POS_PARA1', 'POS_PARA2', 'POS_PARA3', 'POS_PARA4',
         'POS_ADENO', 'POS_METAP', 'POS_BOCA', 'POS_RINO', 'POS_OUTROS']
demais_virus = ''

list_filepath = [influd22, influd23, influd24]

list_X = []
list_y = []
list_weeks = []
list_train = []
list_training_weeks = []

for filepath in list_filepath:
  print(filepath)
  srag = SRAG(filepath,old_filter=True)
  X_aux, y_aux = srag.generate_training_data(None, 'multiclass', cols_X, col_y, demais_virus)
  list_X.append(X_aux)
  list_y.append(y_aux)
  list_training_weeks.append(srag.generate_training_weeks())

  weeks = np.unique(X_aux['ANO_SEM_SIN_PRI'])
  train = srag.get_start_day_of_week(0)
  list_train.append(train)
  list_weeks.append(weeks)

X = pd.concat(list_X).reset_index(drop=True)
y = pd.concat(list_y).reset_index(drop=True)
df_training_weeks = pd.concat(list_training_weeks).reset_index(drop=True)

# Treinar o modelo
trainer = GBMTrainer(objective='multiclass', eval_metric='multi_logloss')
trainer.fit(X, y)

# Salvar o modelo
model = {'filename': list_filepath,
         'weeks': list_weeks,
         'train': list_train,
          'virus': trainer.model.classes_,
          'df_training_weeks': df_training_weeks,
          'model': trainer,
          'train_size': len(y),
          'best_boost_iteration': trainer.model.best_iteration_}

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
