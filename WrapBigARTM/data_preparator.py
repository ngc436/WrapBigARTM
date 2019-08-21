# Batches preparation
import artm
import glob
import os
import pandas as pd
from pyarrow import parquet


def get_words_dict(text, stop_list):
    all_words = text
    words = sorted(set(all_words) - set(stop_list))

    return {w: all_words.count(w) for w in words}


def return_string_part(name_type, text):
    tokens = text.split()
    tokens = [item for item in tokens if item != '']
    tokens_dict = get_words_dict(tokens, set())

    tokens_freq = ' '.join(['{}:{}'.format(k, v) for k, v in tokens_dict.items()])

    return " |{} {}".format(name_type, tokens_freq)


def file_reader(full_path):
    if full_path.endswith('.csv'):
        part = pd.read_csv(full_path)

    elif full_path.endswith('.parquet'):
        part = pd.read_parquet(full_path)
    
    else:
        print('Unsupported file\'s type encountered')
        part = None

    return part


def prepare_batches(batches_dir, vw_path, data_path, column_name='processed_text'):
    print('Starting...')
    with open(vw_path, 'w', encoding='utf8') as ofile:

        for num_parts, file_name in enumerate(os.listdir(data_path)):

            if file_name.startswith('part'):
                full_path = os.path.join(data_path, file_name)

                print('part_{}'.format(num_parts), end='\r')

                part = file_reader(full_path)
                if part is None:
                    continue

                part_processed = part[column_name].tolist()

                for text in part_processed:
                    result = return_string_part('text', text)
                    ofile.write('{}\n'.format(result))

    print(' batches {} \n vocabulary {} \n are ready'.format(batches_dir, vw_path))


def prepare_batch_vectorizer(batches_dir, vw_path, data_path, column_name):
    if not glob.glob(os.path.join(batches_dir, '*')):
        prepare_batches(batches_dir, vw_path, data_path, column_name=column_name)

        batch_vectorizer = artm.BatchVectorizer(
            data_path=vw_path,
            data_format='vowpal_wabbit',
            target_folder=batches_dir,
            batch_size=100)
    else:
        batch_vectorizer = artm.BatchVectorizer(data_path=batches_dir, data_format='batches')

    return batch_vectorizer
