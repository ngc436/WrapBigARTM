import re
import html

import re
from nltk.corpus import stopwords
from pymystem3 import Mystem

r_vk_ids = re.compile(r'(id{1}[0-9]*)')
r_num = re.compile(r'([0-9]+)')
r_punct = re.compile(r'[."\[\]/,()!?;:*#|\\%^$&{}~_`=-@]')
r_white_space = re.compile(r'\s{2,}')
r_words = re.compile(r'\W+')

stop = stopwords.words("russian")


def process_punkt(text):
    text = r_punct.sub(" ", text)
    text = r_vk_ids.sub(" ", text)
    text = r_num.sub(" ", text)
    text = r_white_space.sub(" ", text)
    return text.strip()


def lemmatize_text(text):
    m = Mystem()
    text = text.lower()
    text = process_punkt(text)
    try:
        tokens = r_words.split(text)
    except:
        return ''
    tokens = (x for x in tokens if len(x) >= 2 and not x.isdigit())
    tokens = (m.lemmatize(x)[0] for x in tokens)
    tokens = (x for x in tokens if x not in stop)
    tokens = (x for x in tokens if x.isalpha())
    text = ' '.join(tokens)
    return text


def text_to_tokens(text):
    return text.split()


# clean texs from html
re1 = re.compile(r'  +')

url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')


def remove_more_html(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('<', ' ').replace('>', ' ').replace('#36;', '$').replace(
        '\\n', "\n").replace('quot;', "'").replace('<br />', "\n").replace(
        '\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
        ' @-@ ', '-').replace('\\', ' \\ ').replace('img', ' ').replace('class', ' ').replace(
        'src', ' ').replace('alt', ' ').replace('email', ' ').replace('icq', ' ').replace(
        'href', ' ').replace('mem', ' ').replace('link', ' ').replace('mention', ' ').replace(
        'onclick', ' ').replace('icq', ' ').replace('onmouseover', ' ').replace('post', ' ').replace(
        'local', ' ').replace('key', ' ').replace('target', ' ').replace('amp', ' ').replace(
        'section', ' ').replace('search', ' ').replace('css', ' ').replace('style', ' ').replace(
        'cc', ' ').replace('text', ' ').replace("img", ' ').replace("expand", ' ').replace(
        "text", ' ').replace('\n', ' ').replace('dnum', ' ')
    return re1.sub(' ', html.unescape(x))


def clear_url(text):
    return re.sub(url_pattern, ' ', text)


def clean_html(text):
    new_string = []
    try:
        for chunk in text.split('<'):
            splitted_chunk = chunk.split('>')
            if len(splitted_chunk) > 1:
                new_string.append(splitted_chunk[1])
            else:
                new_string.append(chunk)
        result = remove_more_html(' '.join(new_string))
        return clear_url(result).strip()
    except:
        return ''