# Batches preparation
import artm
import glob
import os
import pandas as pd
from pyarrow import parquet

def get_words_dict(text, stop_list):
    all_words = text
    words = sorted(set(all_words) - stop_list)

    return {w: all_words.count(w) for w in words}


def return_string_part(name_type, text):
    tokens = text.split()
    tokens = [item for item in tokens if item != '']
    tokens_dict = get_words_dict(tokens, set())

    return " |" + name_type + ' ' + ' '.join(['{}:{}'.format(k, v) for k, v in tokens_dict.items()])


def prepare_batches(batches_dir, vw_path, data_path, column_name='processed_text'):
    print('Starting...')
    with open(vw_path, 'w', encoding='utf8') as ofile:
        num_parts = 0
        for file in os.listdir(data_path):
            if file.startswith('part'):
                print('part_{}'.format(num_parts), end='\r')
                if file.split('.')[-1] == 'csv':
                    part = pd.read_csv(os.path.join(data_path, file))
                else:
                    part = pd.read_parquet(os.path.join(data_path, file))
                part_processed = part[column_name].tolist()
                for text in part_processed:
                    result = return_string_part('text', text)
                    ofile.write(result + '\n')
                num_parts += 1
    print(' batches {} \n vocabulary {} \n are ready'.format(batches_dir, vw_path))


def prepare_batch_vectorizer(batches_dir, vw_path, data_path, column_name):
    if not glob.glob(os.path.join(batches_dir, "*")):
        prepare_batches(batches_dir, vw_path, data_path, column_name=column_name)
        batch_vectorizer = artm.BatchVectorizer(data_path=vw_path,
                                                data_format="vowpal_wabbit",
                                                target_folder=batches_dir,
                                                batch_size=100)
    else:
        batch_vectorizer = artm.BatchVectorizer(data_path=batches_dir, data_format='batches')

    return batch_vectorizer
