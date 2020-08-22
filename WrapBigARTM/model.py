# Currently only main and back topics and text modality are supported

import os
import numpy as np
import artm
from WrapBigARTM.scores import return_all_tokens_coherence


class Topic_model:

    def __init__(self, experiments_path, S=100, decor_test=False):

        # experiments_path:

        self.model = None
        self.S = S
        self.specific = ['main{}'.format(i) for i in range(S)]

        self.save_path = os.path.join(experiments_path, 'best_model')
        self.decor_test = decor_test

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def set_params(self, params_string):
        self.decor = params_string[0]
        self.n1 = params_string[1]

        if self.decor_test:
            return

        self.spb = params_string[2]
        self.stb = params_string[3]
        self.n2 = params_string[4]
        self.sp1 = params_string[5]
        self.st1 = params_string[6]
        self.n3 = params_string[7]
        #         self.sp2 = params_string[8]
        #         self.st2 = params_string[9]
        #         self.n4 = params_string[10]
        #         self.B = params_string[11]
        #         self.decor_2 = params_string[14]

        self.B = params_string[8]

    #         self.decor_2 = params_string[8]

    def init_model(self, params_string, dict_path):

        self.set_params(params_string)
        self.back = ['back{}'.format(i) for i in range(self.B)]
        self.dictionary = artm.Dictionary()
        self.dictionary.load_text(dictionary_path=dict_path)

        self.model = artm.ARTM(num_topics=self.S + self.B,
                               class_ids=['@default_class'],
                               dictionary=self.dictionary,
                               show_progress_bars=False,
                               #                   cache_theta=True,
                               topic_names=self.specific + self.back,
                               num_processors=32)

        self.set_scores()

    def save_model(self, ii):
        self.model.dump_artm_model(os.path.join(self.save_path, 'model_{}'.format(ii)))

    def set_scores(self):

        self.model.scores.add(artm.PerplexityScore(name='PerplexityScore', dictionary=self.dictionary))

        self.model.scores.add(
            artm.SparsityPhiScore(name='SparsityPhiScore', class_id='@default_class', topic_names=self.specific))
        self.model.scores.add(artm.SparsityThetaScore(name='SparsityThetaScore', topic_names=self.specific))

        # Fraction of background words in the whole collection
        self.model.scores.add(
            artm.BackgroundTokensRatioScore(name='BackgroundTokensRatioScore', class_id='@default_class'))

        # Kernel characteristics
        self.model.scores.add(
            artm.TopicKernelScore(name='TopicKernelScore', class_id='@default_class', topic_names=self.specific,
                                  probability_mass_threshold=0.5, dictionary=self.dictionary))

        # Looking at top tokens
        self.model.scores.add(artm.TopTokensScore(name='TopTokensScore', class_id='@default_class', num_tokens=100))

    def train(self, batch_vectorizer):
        if self.model is None:
            print('Initialise the model first!')
            return

        self.model.regularizers.add(artm.DecorrelatorPhiRegularizer(name='decorr',
                                                                    topic_names=self.specific, tau=self.decor))
        #         self.model.regularizers.add(artm.DecorrelatorPhiRegularizer(name='decorr_2',
        #                                                               topic_names=self.back, tau=self.decor_2))
        self.model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=self.n1)

        #         if ((self.n2 != 0) and (self.B != 0)):
        if (self.B != 0):
            self.model.regularizers.add(artm.SmoothSparseThetaRegularizer(name='SmoothPhi',
                                                                          topic_names=self.back, tau=self.spb))
            self.model.regularizers.add(artm.SmoothSparseThetaRegularizer(name='SmoothTheta',
                                                                          topic_names=self.back, tau=self.stb))
            self.model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=self.n2)

        self.model.regularizers.add(artm.SmoothSparseThetaRegularizer(name='SparsePhi',
                                                                      topic_names=self.specific, tau=self.sp1))
        self.model.regularizers.add(artm.SmoothSparseThetaRegularizer(name='SparseTheta',
                                                                      topic_names=self.specific, tau=self.st1))
        self.model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=self.n3)

        #         if (self.n4 != 0):
        #             self.model.regularizers['SparsePhi'].tau = self.sp2
        #             self.model.regularizers['SparseTheta'].tau = self.st2
        #             self.model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=self.n4)

        print('Training is complete')

    def decor_train(self):
        if self.model is None:
            print('Initialise the model first')
            return

        self.model.regularizers.add(artm.DecorrelatorPhiRegularizer(name='decorr',
                                                                    topic_names=self.specific, tau=self.decor))

    def get_avg_coherence_score(self, mutual_info_dict, only_specific=True, for_individ_fitness=False):
        coherences_main, coherences_back = return_all_tokens_coherence(self.model, S=self.S, B=self.B,
                                                                       mutual_info_dict=mutual_info_dict)
        if for_individ_fitness:
            print('COMPONENTS: ', np.mean(list(coherences_main.values())), np.min(list(coherences_main.values())))
            return np.mean(list(coherences_main.values())) + np.min(list(coherences_main.values()))
        return np.mean(list(coherences_main.values()))

    def print_topics(self):
        res = self.model.score_tracker['TopTokensScore'].last_tokens
        for i, topic in enumerate(self.model.topic_names):
            print(topic)
            print(" ".join(res[topic][:50]))
            print()

    def get_prob_mixture(self, batches_path):
        theta_test = self.model.transform(batch_vectorizer=batches_path)
        theta_test_trans = theta_test.T
        theta_test_trans = theta_test_trans.sort_index()
        return theta_test_trans
