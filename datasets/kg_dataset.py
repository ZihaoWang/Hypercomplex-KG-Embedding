"""Dataset class for loading and processing KG datasets."""

import os
import pickle as pkl
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

class KGDataset(object):
    """Knowledge Graph dataset class."""

    def __init__(self, data_path, debug):
        """Creates KG dataset object for data loading.

        Args:
             data_path: Path to directory containing train/valid/test pickle files produced by process.py
             debug: boolean indicating whether to use debug mode or not
             if true, the dataset will only contain 1000 examples for debugging.
        """
        self.data_path = data_path
        self.debug = debug
        self.data = {}
        for split in ["train", "test", "valid"]:
            file_path = os.path.join(self.data_path, split + ".pickle")
            with open(file_path, "rb") as in_file:
                self.data[split] = pkl.load(in_file)
        filters_file = open(os.path.join(self.data_path, "to_skip.pickle"), "rb")
        self.to_skip = pkl.load(filters_file)
        filters_file.close()
        max_axis = np.max(self.data["train"], axis=0)
        
        self.n_entities = int(max(max_axis[0], max_axis[2]) + 1)
        self.n_predicates = int(max_axis[1] + 1) * 2
        
        self.dataset_dim = len(max_axis)
        if self.dataset_dim == 4:
            self.n_time = int(max_axis[3] + 1)
        if self.dataset_dim == 5:
            self.n_location = int(max_axis[3] + 1)
            self.n_time = int(max_axis[4] + 1)
        
        file_name= os.path.join(self.data_path , ('entity_embedd_data.pickle'))
        if os.path.exists(file_name): 
            with open(file_name, 'rb') as ent_data:
                self.entity_embedd_data = pkl.load(ent_data)
        
        file_name= os.path.join(self.data_path , ('rel_embedd_data.pickle'))
        if os.path.exists(file_name):
            with open(file_name, 'rb') as rel_data:
                self.rel_embedd_data = pkl.load(rel_data)
        
        file_name= os.path.join(self.data_path , ('image_embedd_data.pickle'))
        if os.path.exists(file_name):
            with open(file_name, 'rb') as img_data:
                self.image_embedd_data = pkl.load(img_data)
        
        file_name= os.path.join(self.data_path , ('word_embedd_data.pickle'))
        if os.path.exists(file_name):
            with open(file_name, 'rb') as text_data:
                self.word_embedd_data = pkl.load(text_data)
        
        file_name= os.path.join(self.data_path , ('fasttext_embedd_data.pickle'))
        if os.path.exists(file_name):
            with open(file_name, 'rb') as fasttext_data:
                self.fasttext_embedd_data = pkl.load(fasttext_data)
        
        file_name= os.path.join(self.data_path , ('sentenseTransformer_embedd_data.pickle'))
        if os.path.exists(file_name):
            with open(file_name, 'rb') as sentenseTransformer_data:
                self.sentenseTransformer_embedd_data = pkl.load(sentenseTransformer_data)
        
        file_name= os.path.join(self.data_path , ('doc2vec_embedd_data.pickle'))
        if os.path.exists(file_name):
            with open(file_name, 'rb') as doc2vec_data:
                self.doc2vec_embedd_data = pkl.load(doc2vec_data)
        
    def get_examples(self, split, rel_idx=-1):
        """Get examples in a split.

        Args:
            split: String indicating the split to use (train/valid/test)
            rel_idx: integer for relation index to keep (-1 to keep all relation)

        Returns:
            examples: torch.LongTensor containing KG triples in a split
        """
        examples = self.data[split]
        if split == "train":
            copy = np.copy(examples)
            tmp = np.copy(copy[:, 0])
            copy[:, 0] = copy[:, 2]
            copy[:, 2] = tmp
            copy[:, 1] += self.n_predicates // 2
            examples = np.vstack((examples, copy))
        if rel_idx >= 0:
            examples = examples[examples[:, 1] == rel_idx]
        if self.debug:
            examples = examples[:1000]
        return torch.from_numpy(examples.astype("int64"))
        
    def get_entity_embedding_lm_data(self):
        txt = torch.from_numpy(self.entity_embedd_data.astype('float32'))
        txt = F.normalize(txt, p = 2, dim = 1)
        return txt
    def get_relation_embedding_lm_data(self, dr = False):
        txt = torch.from_numpy(self.rel_embedd_data.astype('float32'))
        txt = F.normalize(txt, p = 2, dim = 1)
        if (dr == True):
            return torch.cat((txt, txt), 0)
        return txt
        
    def get_image_embedd_data(self):
        img = torch.from_numpy(self.image_embedd_data.astype('float32'))
        return img
    
    def get_text_embedd_data(self):
        txt = torch.from_numpy(self.word_embedd_data.astype('float32'))
        txt = F.normalize(txt, p = 2, dim = 1)
        return txt
    def get_fasttext_embedd_data(self):
        txt = torch.from_numpy(self.fasttext_embedd_data.astype('float32'))
        return txt
    def get_sentenseTransformer_embedd_data(self):
        txt = torch.from_numpy(self.sentenseTransformer_embedd_data.astype('float32'))
        return txt
    def get_doc2vec_embedd_data(self):
        txt = torch.from_numpy(self.doc2vec_embedd_data.astype('float32'))
        return txt
    
    def get_filters(self, ):
        """Return filter dict to compute ranking metrics in the filtered setting."""
        return self.to_skip

    def get_shape(self):
        """Returns KG dataset shape."""
        if self.dataset_dim == 3:
            return self.n_entities, self.n_predicates, self.n_entities
        elif self.dataset_dim == 4:
            return self.n_entities, self.n_predicates, self.n_entities, self.n_time
        else:
            return self.n_entities, self.n_predicates, self.n_entities, self.n_location, self.n_time

