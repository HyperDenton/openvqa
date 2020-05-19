# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# Modified by Tian Deng https://github.com/HyperDenton
# --------------------------------------------------------

import en_vectors_web_lg
import json
import re
import numpy as np
import os
import glob
from openvqa.core.base_dataset import BaseDataSet
import time

FRCN_PATH = '/demo/input'
QUES_PATH = '/demo/input'

class DataSet():
    def __init__(self, __C):
        super(DataSet, self).__init__()
        self.__C = __C
        
        # Loading question word list
        print("Loading question word list...")
        stat_ques_list = \
            json.load(open(__C.RAW_PATH[__C.DATASET]['train'], 'r'))['questions'] + \
            json.load(open(__C.RAW_PATH[__C.DATASET]['val'], 'r'))['questions'] + \
            json.load(open(__C.RAW_PATH[__C.DATASET]['test'], 'r'))['questions'] + \
            json.load(open(__C.RAW_PATH[__C.DATASET]['vg'], 'r'))['questions']
        print("Done!")

        print("Getting frcnn-feature and question...")
        # Get frcnn-feature and question 
        self.frcn_feat = self.get_frcn()
        self.ques = self.get_ques()
        self.data_size = len(self.ques)
        self.jsonfile = {}
        print("Done!")

        print("Tokenizing...")
        # Tokenize
        self.token_to_ix, self.pretrained_emb = self.tokenize(stat_ques_list, __C.USE_GLOVE)
        self.token_size = self.token_to_ix.__len__()
        print("Done!")
        # print(' ========== Question token vocab size:', self.token_size)

        print("Loading answer dictionary...")
        # Answers statistic
        self.ans_to_ix, self.ix_to_ans = \
            self.ans_stat('openvqa/datasets/vqa/answer_dict.json')
        self.ans_size = self.ans_to_ix.__len__()
        print("Done!")

        print("Loading question and image feature...") 
        self.ques_ix_iter = self.load_ques()
        self.frcn_feat_iter, self.bbox_feat_iter = self.load_img_feats()
        self.grid_feat_iter = None
        self.ans_iter = None
        print("Done!")


        # print(' ========== Answer token vocab size (occur more than {} times):'.format(8), self.ans_size)
        # print('Finished!')
        # print('')

    def get_frcn(self):
        frcn_path = glob.glob("demo/input/*.npz")
        if len(frcn_path) > 1:
            raise FileExistsError("Can only process exact one file at a time")
        elif len(frcn_path) == 0:
            raise FileNotFoundError("Frcn file not found")
        else:
            frcn_feat = np.load(frcn_path[0])
        return frcn_feat

    def get_ques(self):
        json_path = glob.glob("demo/input/*.json")
        if len(json_path) > 1:
            raise FileExistsError("Can only process exact one file at a time")
        elif len(json_path) == 0:
            raise FileNotFoundError("File not found")
        else:
             self.jsonfile = json.load(open(json_path[0]))
             ques = self.jsonfile['ques']
        return ques

    def tokenize(self, stat_ques_list, use_glove):
        t1 = time.time()
        token_to_ix = {
            'PAD': 0,
            'UNK': 1,
            'CLS': 2,
        }
        

        spacy_tool = None
        pretrained_emb = []
        if use_glove:
            spacy_tool = en_vectors_web_lg.load()
            pretrained_emb.append(spacy_tool('PAD').vector)
            pretrained_emb.append(spacy_tool('UNK').vector)
            pretrained_emb.append(spacy_tool('CLS').vector)

        t2 = time.time()
        print("first part: %f"%(t2-t1))

        for ques in stat_ques_list:
            words = re.sub(
                r"([.,'!?\"()*#:;])",
                '',
                ques['question'].lower()
            ).replace('-', ' ').replace('/', ' ').split()

            for word in words:
                if word not in token_to_ix:
                    token_to_ix[word] = len(token_to_ix)
                    if use_glove:
                        pretrained_emb.append(spacy_tool(word).vector)

        pretrained_emb = np.array(pretrained_emb)

        t3 = time.time()
        print("second part: %f"%(t3-t2))

        return token_to_ix, pretrained_emb

    def ans_stat(self, json_file):
        ans_to_ix, ix_to_ans = json.load(open(json_file, 'r'))
        return ans_to_ix, ix_to_ans

    def get_ans(self, ix):
        return self.ix_to_ans[str(ix)]

    # ----------------------------------------------
    # ---- Real-Time Processing Implementations ----
    # ----------------------------------------------

    def load_ques(self):
        ques = self.get_ques()
        ques_ix_iter = self.proc_ques(ques, self.token_to_ix, max_token = 14)
        return [ques_ix_iter]

    def load_img_feats(self):
        frcn_feat = self.get_frcn()
        frcn_feat_x = frcn_feat['x'].transpose((1, 0))
        frcn_feat_iter = self.proc_img_feat(
            frcn_feat_x, img_feat_pad_size = 
                self.__C.FEAT_SIZE['vqa']['FRCN_FEAT_SIZE'][0])

        bbox_feat_iter = self.proc_img_feat(
            self.proc_bbox_feat(
                frcn_feat['bbox'],
                (frcn_feat['image_h'], frcn_feat['image_w'])
            ),
            img_feat_pad_size=self.__C.FEAT_SIZE['vqa']['BBOX_FEAT_SIZE'][0]
        )

        return [frcn_feat_iter], [bbox_feat_iter]



    # ------------------------------------
    # ---- Real-Time Processing Utils ----
    # ------------------------------------

    def proc_img_feat(self, img_feat, img_feat_pad_size):
        if img_feat.shape[0] > img_feat_pad_size:
            img_feat = img_feat[:img_feat_pad_size]

        img_feat = np.pad(
            img_feat,
            ((0, img_feat_pad_size - img_feat.shape[0]), (0, 0)),
            mode='constant',
            constant_values=0
        )

        return img_feat


    def proc_bbox_feat(self, bbox, img_shape):
        bbox_feat = np.zeros((bbox.shape[0], 5), dtype=np.float32)

        bbox_feat[:, 0] = bbox[:, 0] / float(img_shape[1])
        bbox_feat[:, 1] = bbox[:, 1] / float(img_shape[0])
        bbox_feat[:, 2] = bbox[:, 2] / float(img_shape[1])
        bbox_feat[:, 3] = bbox[:, 3] / float(img_shape[0])
        bbox_feat[:, 4] = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1]) / float(img_shape[0] * img_shape[1])

        return bbox_feat


    def proc_ques(self, question, token_to_ix, max_token):
        ques_ix = np.zeros(max_token, np.int64)
        print(question)
        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            question.lower()
        ).replace('-', ' ').replace('/', ' ').split()

        for ix, word in enumerate(words):
            if word in token_to_ix:
                ques_ix[ix] = token_to_ix[word]
            else:
                ques_ix[ix] = token_to_ix['UNK']

            if ix + 1 == max_token:
                break
        return ques_ix


    def get_score(self, occur):
        if occur == 0:
            return .0
        elif occur == 1:
            return .3
        elif occur == 2:
            return .6
        elif occur == 3:
            return .9
        else:
            return 1.