import html
from pymystem3 import Mystem
import re
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

r_vk_ids = re.compile(r'(id{1}[0-9]*)')
r_num = re.compile(r'([0-9]+)')
r_punct = re.compile(r'[."\[\]/,()!?;:*#|\\%^$&{}~_`=-@]')
r_white_space = re.compile(r'\s{2,}')
r_words = re.compile(r'\W+')
r_rus = re.compile(r'[а-яА-Я]\w+')
r_html = re.compile(r'(\<[^>]*\>)')
# clean texs from html
re1 = re.compile(r'  +')
r_eng = re.compile(r"[^A-Za-z\s]")
url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

m = Mystem()
lmtzr = WordNetLemmatizer()


def process_punkt(text):
    text = r_punct.sub(" ", text)
    text = r_vk_ids.sub(" ", text)
    text = r_num.sub(" ", text)
    text = r_white_space.sub(" ", text)
    return text.strip()


def lemmatize_text(text, language='en'):
    try:
        text = new_html(text)
    except:
        return ''
    text = text.lower()
    text = process_punkt(text)
    try:
        tokens = r_words.split(text)
    except:
        return ''
    tokens = (x for x in tokens if len(x) >= 2 and not x.isdigit())
    text = ' '.join(tokens)
    if language == 'en':
        stop = stopwords.words("english")
        text = r_eng.sub("", text.strip())
        tokens = lmtzr.lemmatize(text).split()
    elif language == 'ru':
        stop = stopwords.words("russian")
        text = re.findall(r_rus, text)
        tokens = m.lemmatize(text)
    tokens = (x for x in tokens if x not in stop)
    tokens = (x for x in tokens if x.isalpha())
    text = ' '.join(tokens)
    return text


def tokens_bigrams_to_text(tokens):
    return ' '.join(['_'.join(tok.split()) for tok in tokens])


def text_to_tokens(text):
    return text.split()


def get_tokens_count(text):
    return len(text.split())


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
        "text", ' ').replace('\n', ' ').replace('dnum', ' ').replace('xnum', ' ').replace('nnum', ' ')
    return re1.sub(' ', html.unescape(x))


def clear_url(text):
    return re.sub(url_pattern, ' ', text)


def new_html(text):
    text = r_html.sub("", text)
    return text
