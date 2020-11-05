# code from:
#   https://github.com/letiantian/TextRank4ZH
#   https://zhuanlan.zhihu.com/p/84754176
import jieba.posseg as pseg
import networkx as nx
import re
import numpy as np

def cut_sents(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    para = para.rstrip()
    return para.split("\n")

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def load_stopwords(fpath):
    sws = []
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if line:
                sws.append(line)
    return set(sws)
stop_words = load_stopwords("./stopwords.txt")

class TextRankKeyword():
    def segment(self, text):
        word_tag_list = pseg.cut(text)
        word_tag_list = list(map(lambda x: [x.word, x.flag], word_tag_list))
        return word_tag_list
    
    def get_word_links(self, text):
        word_tag_list = self.segment(text)
        word_keyword_1 = list(map(lambda x: [x[0], True] if 'n' in x[1] and x[0] not in stop_words else [x[0], False], word_tag_list))
        word_link_list = []
        for i in range(len(word_keyword_1)):
            current_word, current_flag = word_keyword_1[i][0], word_keyword_1[i][1]
            if current_flag:
                word_link_list.append(current_word)
        return word_link_list
    
    def combine(self, word_list, window = 2):
        if window < 2: window = 2
        for x in range(1, window):
            if x >= len(word_list):
                break
            word_list2 = word_list[x:]
            res = zip(word_list, word_list2)
            for r in res:
                yield r

    def get_keyword_with_textrank(self, text, top_n=10, window=5, pagerank_config = {'alpha': 0.85}):
        sorted_words   = []
        word_index     = {}
        index_word     = {}
        words_number = 0
        word_list = self.get_word_links(text)
        for word in word_list:
            if not word in word_index:
                word_index[word] = words_number
                index_word[words_number] = word
                words_number += 1

        graph = np.zeros((words_number, words_number))
    
        for w1, w2 in self.combine(word_list, window):
            if w1 in word_index and w2 in word_index and w1 != w2:
                index1 = word_index[w1]
                index2 = word_index[w2]
                graph[index1][index2] = 1.0
                graph[index2][index1] = 1.0

        nx_graph = nx.from_numpy_matrix(graph)
        scores = nx.pagerank(nx_graph, **pagerank_config)          # this is a dict
        sorted_scores = sorted(scores.items(), key = lambda item: item[1], reverse=True)
        for index, score in sorted_scores:
            item = AttrDict(word=index_word[index], weight=score)
            sorted_words.append(item)

        return sorted_words[:top_n]

class TextRankSummary():
    def text2words(self, text):
        sentences = cut_sents(text.replace('\n', '').replace(' ', ''))
        sentences = list(filter(lambda x: len(x)>1, sentences))
        words_list = list(map(lambda x: self.word_segment(x), sentences))
        new_sentences, new_words_list = [], []
        for i in range(len(sentences)):
            if len(words_list[i])>0:
                new_sentences.append(sentences[i])
                new_words_list.append(words_list[i])
        return new_sentences, new_words_list
    
    def word_segment(self, sentence):
        word_tag_list = pseg.cut(sentence)
        words = []
        for word_tag in word_tag_list:
            word_tag = [word_tag.word, word_tag.flag]
            if len(word_tag)==2:
                word, tag = word_tag
                if 'n' in tag and word not in stop_words:
                    words.append(word)
        return set(words)
            
    def sentence_simlarity(self, words1, words2):
        word_set1, word_set2 = set(words1), set(words2)
        simlarity = len(word_set1 & word_set2)/len(word_set2 | word_set2)
        return simlarity
    
    def get_summary_with_textrank(self, text, top_n=10, pagerank_config = {'alpha': 0.85}):
        sorted_sentences = []
        sentences, words = self.text2words(text) 
        _source = words
        sentences_num = len(_source)        
        graph = np.zeros((sentences_num, sentences_num))
    
        for x in range(sentences_num):
            for y in range(x, sentences_num):
                similarity = self.sentence_simlarity( _source[x], _source[y])
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
    text = ''.join(list(open('./data/07.txt', 'r', encoding='utf8').readlines()))
    keyword_extractor = TextRankKeyword()
    keyword_weight_list = keyword_extractor.get_keyword_with_textrank(text)
    print("关键词是:")
    print(keyword_weight_list)


    summary = TextRankSummary()
    summary_sentences = summary.get_summary_with_textrank(text, 5)
    print("摘要是：")
    print(summary_sentences)
