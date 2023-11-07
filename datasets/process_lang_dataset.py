# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from sys import exit
import pkg_resources
import os
import errno
from pathlib import Path
import pickle
import numpy as np
import string
from re import sub
from collections import defaultdict
from gensim.models import Word2Vec, fasttext
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sentence_transformers import SentenceTransformer
import torch

translator = str.maketrans('','', sub('\-', '', string.punctuation))
exclude = set(string.punctuation)
stop_words=['the', 'a', 'an', 'and', 'is', 'be', 'will', '-', 'or', 'to']

DATA_PATH = "../data/"

def prepare_dataset(path, dataset_name):
    """
    Given a path to a folder containing tab separated files :
     train, test, valid
    In the format :
    (lhs)\t(rel)\t(rhs)\n
    Maps each entity and relation to a unique id, create corresponding folder
    name in pkg/data, with mapped train/test/valid files.
    Also create to_skip_lhs / to_skip_rhs for filtered metrics and
    rel_id / ent_id for analysis.
    """
    files = ['train', 'valid', 'test']
    entities, relations, description, words = set(), set(), set(), set()
    for f in files:
        file_path = os.path.join(path, f)
        to_read = open(file_path+'.txt', 'r')
        for line in to_read.readlines():
            lhs, rel, rhs = line.strip().split('\t')
            entities.add(lhs)
            entities.add(rhs)
            relations.add(rel)
        to_read.close()
    entities_to_id = {x: i for (i, x) in enumerate(sorted(entities))}
    relations_to_id = {x: i for (i, x) in enumerate(sorted(relations))}
    #####
    n_relations = len(relations)
    n_entities = len(entities)
    
    id_to_desc = [0] * n_entities
    all_sentences = [0] * n_entities
    
    to_read_text = open(os.path.join(path, 'description.txt'), 'r')
    for line in to_read_text.readlines():
        ent, txt = line.strip().split('\t')
        if ent not in entities_to_id.keys():
            continue
        description.add(txt)
        all_sentences[entities_to_id[ent]]=txt

        txt = txt.translate(translator).lower()
        tmp = txt.split()
        resultwords  = [word for word in tmp if word not in stop_words]
        for item in resultwords:
            words.add(item)
        id_to_desc[entities_to_id[ent]] =  resultwords

    to_read_text.close()

    #give id to all description
    desc_to_id = {x: i for (i, x) in enumerate(sorted(description))}

    n_texts = len(description)
    n_words = len(words)

    print("{} entities , {} relations, {} description, and {} words".format(n_entities, n_relations, n_texts, n_words))

    # write ent to id / rel to id
    for (dic, f) in zip([entities_to_id, relations_to_id, desc_to_id], ['ent2id', 'rel2id', 'desc2id']):
        ff = open(os.path.join(DATA_PATH, dataset_name, f), 'w+')
        for (x, i) in dic.items():
            ff.write("{}\t{}\n".format(x, i))
        ff.close()

    entity_to_id = {}
    with open(os.path.join(DATA_PATH, dataset_name, 'ent2id'), "r") as lines:
        for line in lines:
            name, id = line.strip().split("\t")
            ent = name.replace('_', ' ')
            entity_to_id[name] = ent

    ff = open(os.path.join(DATA_PATH, dataset_name, 'entities.dict'), 'w+')
    for (x, i) in entity_to_id.items():
        ff.write("{}\t{}\n".format(x, i))
    ff.close()

    ########
    all_ent_names = [0] * n_entities
    to_read_text = open(os.path.join(DATA_PATH, dataset_name, 'entities.dict'), 'r')
    for line in to_read_text.readlines():
        ent, txt = line.strip().split('\t')
        if ent not in entities_to_id.keys():
            continue
        txt = txt.replace('/','')
        txt = txt.replace(',','')
        txt = "\"" + txt + "\""
        txt = txt.translate(translator).lower()
        tmp = txt.split()
        resultwords  = [word for word in tmp]
        all_ent_names[entities_to_id[ent]] = resultwords
    to_read_text.close()

    #######

    word2vec_model = Word2Vec(sentences=id_to_desc, vector_size=128, min_count=1, workers=4, epochs=10)
    word_vectors = word2vec_model.wv

    sentenceTrans = SentenceTransformer('distilbert-base-nli-mean-tokens')
    fasttext_vec = fasttext.load_facebook_vectors("wiki.simple.bin")

    data = np.squeeze(all_sentences)
    tagged_data = [TaggedDocument(doc, [i]) for i, doc in enumerate(data)]
    doc = Doc2Vec(documents = tagged_data, vector_size=128, window=2, min_count=1, workers=4)
    doc_vec = doc.dv

    entity_model = Word2Vec(sentences=all_ent_names, vector_size=128, min_count=1, workers=4, epochs=10)
    entity_vectors = entity_model.wv

    #store feature vector of fasttext, SentenceTransformer and Doc2Vec
    word_feature_sentenceTrans = [0] * len(all_sentences)
    word_feature_fasttext = [0] * len(all_sentences)
    word_feature_vec = [0] * len(id_to_desc)
    doc_feature_vec = [0] * len(all_sentences)
    for i in range(len(all_sentences)):
        word_feature_fasttext[i] = [0] * 300
        doc_feature_vec[i] = [0] * 128
        word_feature_sentenceTrans[i] = [0] * 768
        if len(all_sentences[i]) > 0:
            word_feature_sentenceTrans[i] = np.array(sentenceTrans.encode(all_sentences[i]))
            word_feature_sentenceTrans[i] = word_feature_sentenceTrans[i].tolist()
            word_feature_fasttext[i] = np.array(fasttext_vec.get_vector(all_sentences[i]))
            word_feature_fasttext[i] = word_feature_fasttext[i].tolist()
            doc_feature_vec[i] = np.array(doc_vec[i])
            doc_feature_vec[i] = doc_feature_vec[i].tolist()
    #store feature vector of Word2Vec for entity description
    for i in range(len(id_to_desc)):
        word_feature_vec[i] = [0] * 128
        if len(id_to_desc[i]) > 0:
            temp_arr = []
            for ii in range(len(id_to_desc[i])):
                if id_to_desc[i][ii] in word_vectors.key_to_index:
                    temp_arr.append(np.array(word_vectors[id_to_desc[i][ii]]))
            word_feature_vec[i] = np.array(temp_arr).sum(axis=0)
            word_feature_vec[i] = word_feature_vec[i].tolist()

    #store feature vector of Word2Vec for entity name
    entity_feature_vec = [0] * len(all_ent_names)
    for i in range(len(all_ent_names)):
        entity_feature_vec[i] = [0] * 128
        if len(all_ent_names[i]) > 0:
            temp_arr = []
            for ii in range(len(all_ent_names[i])):
                if all_ent_names[i][ii] in entity_vectors.key_to_index:
                    temp_arr.append(np.array(entity_vectors[all_ent_names[i][ii]]))
            entity_feature_vec[i] = np.array(temp_arr).sum(axis=0)
            entity_feature_vec[i] = entity_feature_vec[i].tolist()

    #store word embedding data with id
    with open(Path(DATA_PATH) / dataset_name / ('word_embedd_data.pickle'), 'wb') as handle:
        pickle.dump(np.array(word_feature_vec).astype('float32'), handle)
        handle.close()
    with open(Path(DATA_PATH) / dataset_name / ('sentenseTransformer_embedd_data.pickle'), 'wb') as handle:
        pickle.dump(np.array(word_feature_sentenceTrans).astype('float32'), handle)
        handle.close()
    with open(Path(DATA_PATH) / dataset_name / ('fasttext_embedd_data.pickle'), 'wb') as handle:
        pickle.dump(np.array(word_feature_fasttext).astype('float32'), handle)
        handle.close()
    with open(Path(DATA_PATH) / dataset_name / ('doc2vec_embedd_data.pickle'), 'wb') as handle:
        pickle.dump(np.array(doc_feature_vec).astype('float32'), handle)
        handle.close()
    with open(Path(DATA_PATH) / dataset_name / ('entity_embedd_data.pickle'), 'wb') as handle:
        pickle.dump(np.array(entity_feature_vec).astype('float32'), handle)
        handle.close()

    # map train/test/valid with the ids
    for f in files:
        file_path = os.path.join(path, f)
        to_read = open(file_path+'.txt', 'r')
        examples = []
        for line in to_read.readlines():
            lhs, rel, rhs = line.strip().split('\t')
            try:
                examples.append([entities_to_id[lhs], relations_to_id[rel], entities_to_id[rhs]])
            except ValueError:
                continue
        out = open(os.path.join(DATA_PATH, dataset_name, (f + '.pickle')), 'wb')
        pickle.dump(np.array(examples).astype('uint64'), out)
        out.close()
        
        ff = open(os.path.join(DATA_PATH, dataset_name, f), 'w+')
        for i in range(len(examples)):
            ff.write("{}\t{}\t{}\n".format(examples[i][0], examples[i][1], examples[i][2]))
        ff.close()
    
    print("creating filtering lists")

    # create filtering files
    to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}
    for f in files:
        examples = pickle.load(open(Path(DATA_PATH) / dataset_name / (f + '.pickle'), 'rb'))
        for lhs, rel, rhs in examples:
            to_skip['lhs'][(rhs, rel + n_relations)].add(lhs)  # reciprocals
            to_skip['rhs'][(lhs, rel)].add(rhs)

    to_skip_final = {'lhs': {}, 'rhs': {}}

    for kk, skip in to_skip.items():
        for k, v in skip.items():
            to_skip_final[kk][k] = sorted(list(v))

    out = open(Path(DATA_PATH) / dataset_name / 'to_skip.pickle', 'wb')
    pickle.dump(to_skip_final, out)
    out.close()

    examples = pickle.load(open(Path(DATA_PATH) / dataset_name / 'train.pickle', 'rb'))
    counters = {
        'lhs': np.zeros(n_entities),
        'rhs': np.zeros(n_entities),
        'both': np.zeros(n_entities)
    }

    for lhs, rel, rhs in examples:
        counters['lhs'][lhs] += 1
        counters['rhs'][rhs] += 1
        counters['both'][lhs] += 1
        counters['both'][rhs] += 1
    for k, v in counters.items():
        counters[k] = v / np.sum(v)
    out = open(Path(DATA_PATH) / dataset_name / 'probas.pickle', 'wb')
    pickle.dump(counters, out)
    out.close()

if __name__ == "__main__":
    datasets = [
            "yago_10",
            #"fb_ilt"

            ]
    for d in datasets:
        print("Preparing dataset {}".format(d))
        try:
            prepare_dataset(
                os.path.join(DATA_PATH, d ),
                d
            )
        except OSError as e:
            if e.errno == errno.EEXIST:
                print(e)
                print("File exists. skipping...")
            else:
                raise
