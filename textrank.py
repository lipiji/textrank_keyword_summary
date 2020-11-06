import jieba.posseg as pseg
import networkx as nx
import re
import numpy as np

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class TextProcessor():
    def __init__(self, stopfile="./stopwords.txt"):
        def load_stopwords(fpath):
            sws = []
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        sws.append(line)
            return set(sws)
        self.stopwords = load_stopwords(stopfile)
    
    def get_seged_sentences(self, text):
        def cut_sents(para):
            para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)
            para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)
            para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)
            para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
            para = para.rstrip()
            return para.split("\n")
        
        def segment(text):
            word_tag_list = pseg.cut(text)
            word_tag_list = list(map(lambda x: [x.word, x.flag], word_tag_list))
            word_list, word_list_4_keyword, word_list_4_summary = [], [], []
            
            for wt in word_tag_list:
                word, tag = wt[0], wt[1]
                word_list.append(word)
                if word not in self.stopwords:
                    word_list_4_summary.append(word)
                    if "n" in tag:
                        word_list_4_keyword.append(word)
            return (word_list, word_list_4_keyword, word_list_4_summary)

        sentences = cut_sents(text.replace('\n', '').replace(' ', '').lower())
        sentences = list(filter(lambda x: len(x)>1, sentences))
        word_list, word_list_4_keyword, word_list_4_summary = [], [], []
        for ws, ws_ke, ws_sum in list(map(lambda x: segment(x), sentences)):
            word_list.append(ws)
            word_list_4_keyword.append(ws_ke)
            word_list_4_summary.append(ws_sum)
        assert len(sentences) == len(word_list)
        sentences_, word_list_, word_list_4_keyword_, word_list_4_summary_ = [], [], [], []
        for sent, words, words_kw, words_sum in \
                zip(sentences, word_list, word_list_4_keyword, word_list_4_summary):
            if len(words_kw) < 2:
                continue
            if float(len(words_sum)) / len(words) < 0.1:
                continue
            sentences_.append(sent)
            word_list_.append(words)
            word_list_4_keyword_.append(words_kw)
            word_list_4_summary_.append(words_sum)

        sentences = sentences_
        word_list = word_list_
        word_list_4_keyword = word_list_4_keyword_
        word_list_4_summary = word_list_4_summary_
        return sentences, word_list, word_list_4_keyword, word_list_4_summary


class TextRankKeyword():
    def combine(self, word_list, window = 2):
        if window < 2: window = 2
        for x in range(1, window):
            if x >= len(word_list):
                break
            word_list2 = word_list[x:]
            res = zip(word_list, word_list2)
            for r in res:
                yield r
    def get_keyword_with_textrank(self, word_list, word_list_4_keyword,\
            top_n=10, min_len_word=2, window=5, pagerank_config = {'alpha': 0.85}):
        sorted_words   = []
        word_index     = {}
        index_word     = {}
        words_number = 0
        for sent in word_list_4_keyword:
            for word in sent:
                if not word in word_index and len(word) >= min_len_word:
                    word_index[word] = words_number
                    index_word[words_number] = word
                    words_number += 1

        graph = np.zeros((words_number, words_number))
        for sent in word_list:
            for w1, w2 in self.combine(sent, window):
                if w1 in word_index and w2 in word_index and (w1 != w2):
                    index1 = word_index[w1]
                    index2 = word_index[w2]
                    graph[index1][index2] = 1.0
                    graph[index2][index1] = 1.0

        nx_graph = nx.from_numpy_matrix(graph)
        scores = nx.pagerank(nx_graph, **pagerank_config)
        sorted_scores = sorted(scores.items(), key = lambda item: item[1], reverse=True)
        for index, score in sorted_scores:
            item = AttrDict(word=index_word[index], weight=score)
            sorted_words.append(item)

        return sorted_words[:top_n]

class TextRankSummary():
    def sentence_simlarity(self, words1, words2):
        word_set1, word_set2 = set(words1), set(words2)
        simlarity = len(word_set1 & word_set2)/len(word_set2 | word_set2)
        return simlarity
    
    def get_summary_with_textrank(self, sentences, words, top_n=10, pagerank_config = {'alpha': 0.85}):
        sorted_sentences = []
        _source = words
        sentences_num = len(_source)        
        graph = np.zeros((sentences_num, sentences_num))
    
        for x in range(sentences_num):
            for y in range(x, sentences_num):
                similarity = self.sentence_simlarity( _source[x], _source[y])
                if similarity > 0.:
                    graph[x, y] = similarity
                    graph[y, x] = similarity
            
        nx_graph = nx.from_numpy_matrix(graph)
        scores = nx.pagerank(nx_graph, **pagerank_config)              # this is a dict
        sorted_scores = sorted(scores.items(), key = lambda item: item[1], reverse=True)

        for index, score in sorted_scores:
            item = AttrDict(index=index, sentence=sentences[index], weight=score)
            sorted_sentences.append(item)

        return sorted_sentences[:top_n]

    
if __name__ == '__main__':
    text = ''.join(list(open('./data/06.txt', 'r', encoding='utf8').readlines()))
    textprocessor = TextProcessor()

    sentences, word_list, word_list_4_keyword, word_list_4_summary = textprocessor.get_seged_sentences(text)
    keyword_extractor = TextRankKeyword()
    keyword_weight_list = keyword_extractor.get_keyword_with_textrank(word_list, word_list_4_keyword, top_n=20, min_len_word=2)
    print("Keywords:")
    print(keyword_weight_list)


    summary = TextRankSummary()
    summary_sentences = summary.get_summary_with_textrank(sentences, word_list_4_summary, top_n=5)
    print("Summary:")
    print(summary_sentences)
