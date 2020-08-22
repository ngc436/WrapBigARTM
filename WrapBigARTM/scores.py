from tqdm import tqdm


def calculate_topic_coherence(tokens, mutual_info_dict, top=50):
    tokens = tokens[:top]
    total_sum = 0
    for ix, token_1 in enumerate(tokens[:-1]):
        for ij, token_2 in enumerate(tokens[(ix + 1):]):
            try:
                total_sum += mutual_info_dict['{}_{}'.format(token_1, token_2)]
            except KeyError:
                total_sum += 0

    coherence = 2 / (top * (top - 1)) * total_sum
    return coherence

def return_all_tokens_coherence(model, S, B, mutual_info_dict, top=50, return_backs=True):
    topics = list(model.score_tracker['TopTokensScore'].last_tokens.keys())

    res = model.score_tracker['TopTokensScore'].last_tokens

    topics_main = [topic for topic in topics if topic.startswith('main')]
    topics_back = [topic for topic in topics if topic.startswith('back')]

    all_topics_main = [i for i in range(S)]
    existing_topics_main = [int(i[4:]) for i in topics_main]
    inexisting_topics_main = [i for i in all_topics_main if i not in existing_topics_main]

    all_topics_back = [i for i in range(B)]
    existing_topics_back = [int(i[4:]) for i in topics_back]
    inexisting_topics_back = [i for i in all_topics_back if i not in existing_topics_back]

    coh_vals_main = {}
    # coherence for main topics
    for i, topic in tqdm(enumerate(topics_main)):
        coh_vals_main[topic] = calculate_topic_coherence(res[topic][:50], mutual_info_dict, top=top)
    for i, topic in tqdm(enumerate(inexisting_topics_main)):
        coh_vals_main['main{}'.format(i)] = 0

    coh_vals_back = {}
    # coherence for back topics
    for i, topic in tqdm(enumerate(topics_back)):
        coh_vals_back[topic] = calculate_topic_coherence(res[topic][:50], mutual_info_dict, top=top)
    for i, topic in tqdm(enumerate(inexisting_topics_back)):
        coh_vals_back['back{}'.format(i)] = 10  # penalty for not creating backs

    if return_backs:
        return coh_vals_main, coh_vals_back
    else:
        return coh_vals_main


def return_all_coherence_types(model, S, mutual_info_dict, top=[10, 25, 50]):
    topics = list(model.score_tracker['TopTokensScore'].last_tokens.keys())
    coh_vals = {}

    res = model.score_tracker['TopTokensScore'].last_tokens

    topics = [topic for topic in topics if topic.startswith('main')]

    all_topics = [i for i in range(S)]
    existing_topics = [int(i[4:]) for i in topics]
    inexisting_topics = [i for i in all_topics if i not in existing_topics]

    for num_tokens in top:
        coh_vals['coherence_{}'.format(num_tokens)] = {}
        for i, topic in tqdm(enumerate(topics)):
            coh_vals['coherence_{}'.format(num_tokens)][topic] = calculate_topic_coherence(res[topic][:50],
                                                                                           mutual_info_dict,
                                                                                           top=num_tokens)
        for i, topic in tqdm(enumerate(inexisting_topics)):
            coh_vals['coherence_{}'.format(num_tokens)][topic] = 0
    return coh_vals