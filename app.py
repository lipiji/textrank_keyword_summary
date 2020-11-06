import sys
import random
import numpy as np
import logging
import time
import json
import operator
import traceback

import tornado.ioloop
import tornado.web
from textrank import *

mstime = lambda: int(round(time.time() * 1000))
RES_MSG = {0:"successful", 1:"empty input", 2:"error"}

logging.basicConfig(filename='./logs/query.log.' + str(sys.argv[1]),
                    filemode='a',
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S %p',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)

def do_res(code, rs, time):
    res = {}
    res["code"] = code
    res["msg"] = RES_MSG[code]
    res["result"] = rs
    res["time_ms"] = time
    r = json.dumps(res, ensure_ascii=False)
    logger.info(r)
    return r

class MainHandler(tornado.web.RequestHandler):
    
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "*")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.set_header("Content-Type", 'application/json')
    
    def options(self):
        self.set_status(204)
        self.finish()
    
    def post(self):
        start = mstime()
        '''
		input: '{"text":"艾灵", "num_kw":5, "num_sum":5}'
		'''
        try:
            q = self.request.body.decode()
            json_q = json.loads(q)
            if not json_q or "text" not in json_q or not json_q["text"]:
                self.write(do_res(1, {}, mstime() - start))
                return
            text = json_q["text"]
            if "num_kw" not in json_q or json_q["num_kw"] < 0:
                num_kw = 10
            else:
                num_kw = json_q["num_kw"]
            
            if "num_sum" not in json_q or json_q["num_sum"] < 0:
                num_sum = 10
            else:
                num_sum = json_q["num_sum"]

            res_kws = []
            res_sum = []
            textprocessor = TextProcessor()
            sentences, word_list, word_list_4_keyword, word_list_4_summary = textprocessor.get_seged_sentences(text)
            if num_kw > 0:
                keyword_extractor = TextRankKeyword()
                keyword_weight_list = keyword_extractor.get_keyword_with_textrank(word_list, word_list_4_keyword, top_n=num_kw, min_len_word=2)
                for kw in keyword_weight_list:
                    res_kws.append(kw.word)
            if num_sum > 0:
                summary = TextRankSummary()
                summary_sentences = summary.get_summary_with_textrank(sentences, word_list_4_summary, top_n=num_sum)
                for sums in summary_sentences:
                    res_sum.append(sums.sentence)
            self.write(do_res(0, {"keywords":res_kws, "sentences":res_sum}, mstime() - start))
            return
        except Exception as e:
            print(e)
            traceback.print_exc()
            self.write(do_res(2, {}, start - mstime()))
            return

def make_app():
    return tornado.web.Application([
               (r"/nlu/kwsum", MainHandler),
           ])
 
if __name__ == "__main__":
    app = make_app()
    app.listen(sys.argv[1])
    tornado.ioloop.IOLoop.current().start()

