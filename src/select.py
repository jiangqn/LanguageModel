import os
import pandas as pd
import numpy as np

def select(config: dict) -> None:

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu'])

    base_path = config['base_path']
    annotated_save_path = os.path.join(base_path, 'annotated_train.tsv')
    select_save_path = os.path.join(base_path, 'select.tsv')

    df = pd.read_csv(annotated_save_path, delimiter='\t')
    index = np.where(np.asarray(df['ppl']) <= config['select_max_ppl'])[0]
    df = df.iloc[index]
    df.to_csv(select_save_path, sep='\t')