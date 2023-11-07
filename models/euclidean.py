import numpy as np
import torch
from torch import nn
import os
from models.base import KGModel
from datasets.kg_dataset import KGDataset
from utils.euclidean import euc_sqdistance, givens_rotations, givens_reflection, givens_DE_rotations, givens_DE_rotations_m, givens_QuatE_rotations, givens_DE_product

EUC_MODELS = ["TransE", "AttE", "Tetra_zero", "Lion_FS", "Lion_SD", "Robin_S", "Robin_W", "Robin_D", "Robin_F", "Tetra_WSF", "Robin_S_Quat", "Lion_FS_Quat", "Tetra_WSF_Quat", "Tetra_SF"]

DATA_PATH = "./data/"

class AdjustDim(nn.Module):
    def __init__(self, in_dim:int, out_dim:int):
        super(AdjustDim, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)  # input shape
        self.fc2 = nn.Linear(out_dim,out_dim) # output shape
        self.fc1_relu = nn.Tanh()
        self.fc2_relu = nn.Tanh()
        
    def forward(self, x):
        x = self.fc1_relu(self.fc1(x))
        x = self.fc2_relu(self.fc2(x))
        return x

class BaseE(KGModel):
    """Euclidean Knowledge Graph Embedding models.

    Attributes:
        sim: similarity metric to use (dist for distance and dot for dot product)
    """

    def __init__(self, args):
        super(BaseE, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size, args)
        self.indices = []#torch.FloatTensor(400).uniform_(-100, 100)
        self.args = args
        
    def get_rhs(self, queries, eval_mode):        
        """Get embeddings and biases of target entities."""                        
        if eval_mode:
            if self.lm == True:
                return self.entity_lm.weight, self.bt.weight 
            else:
                return self.entity.weight, self.bt.weight 
        else:
            if self.lm == True:
                return self.entity_lm(queries[:, 2]), self.bt(queries[:, 2])
            else:
                return self.entity(queries[:, 2]), self.bt(queries[:, 2])

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
        if self.sim == "dot":
            if eval_mode:
                score = lhs_e @ rhs_e.transpose(0, 1)
            else:
                score = torch.sum(lhs_e * rhs_e, dim=-1, keepdim=True)
        else:
            score = - euc_sqdistance(lhs_e, rhs_e, eval_mode)
        return score

class TransE(BaseE):
    """Euclidean translations https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf"""

    def __init__(self, args):
        super(TransE, self).__init__(args)
        self.entity.weight.data = 0.1 * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)
        self.rel.weight.data = 0.1 * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.sim = "dist"

    def get_queries(self, queries):        
        head_e = self.entity(queries[:, 0])
        rel_e = self.rel(queries[:, 1])
        lhs_e = head_e + rel_e
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases

class Tetra_zero(BaseE):
    """Euclidean translations https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf"""

    def __init__(self, args):
        super(Tetra_zero, self).__init__(args)
        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.rel_rot = nn.Embedding(self.sizes[1], self.rank)
        self.rel_rot.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.sim = "dist"

    def get_queries(self, queries):
        head_e = self.entity(queries[:, 0])
        rel_e = self.rel(queries[:, 1])
        rel_rot_e = self.rel_rot(queries[:, 1])
        rank = self.rank//2

        head_e = head_e[:, :rank//2], head_e[:, rank//2:rank], head_e[:, rank:3*rank//2], head_e[:, 3*rank//2:]
        #rel_e = rel_e[:, :rank//2], rel_e[:, rank//2:rank], rel_e[:, rank:3*rank//2], rel_e[:, 3*rank//2:]
        rel_rot_e = rel_rot_e[:, :rank//2], rel_rot_e[:, rank//2:rank], rel_rot_e[:, rank:3*rank//2], rel_rot_e[:, 3*rank//2:]

        A, B, C, D = givens_DE_rotations(rel_rot_e, head_e)

        E = torch.cat((A, B), 1)
        F = torch.cat((E, C), 1)
        h_e = torch.cat((F, D), 1)
        
        lhs_e = h_e + rel_e
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases

class Robin_S(BaseE):
    """Euclidean translations https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf"""
    def __init__(self, args):
        super(Robin_S, self).__init__(args)
        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        dataset_path = os.path.join(DATA_PATH, args.dataset)
        dataset = KGDataset(dataset_path, args.debug)

        image_embedd_data = dataset.get_entity_embedding_lm_data()
        text_embedd_data = dataset.get_sentenseTransformer_embedd_data()
        #self.img_ln = nn.Parameter(image_embedd_data).requires_grad_(False)

        self.dim_red_model_img = AdjustDim(in_dim = len(image_embedd_data[1]),out_dim=self.rank//2)
        self.dim_red_model_txt = AdjustDim(in_dim = len(text_embedd_data[1]),out_dim=self.rank//2)

        self.dim_red_model_img_tr = AdjustDim(in_dim = len(image_embedd_data[1]),out_dim=self.rank)
        self.dim_red_model_txt_tr = AdjustDim(in_dim = len(text_embedd_data[1]),out_dim=self.rank)

        self.orig_image_embeddings= image_embedd_data.to(self.cuda1)
        self.orig_text_embeddings = text_embedd_data.to(self.cuda1)
        
        self.sim = "dist"

    def get_queries(self, queries):
        img_data = self.dim_red_model_img(self.orig_image_embeddings)
        self.img_rot = img_data
        txt_data = self.dim_red_model_txt(self.orig_text_embeddings)
        self.text_rot = txt_data

        img_data = self.dim_red_model_img_tr(self.orig_image_embeddings)
        self.img_trans = img_data
        txt_data = self.dim_red_model_txt_tr(self.orig_text_embeddings)
        self.text_trans = txt_data

        head_e = self.entity(queries[:, 0])
        rel_e = self.rel(queries[:, 1])
        img_rot_e = self.img_rot[queries[:, 0]]
        text_rot_e = self.text_rot[queries[:, 0]]

        img_e = self.img_trans[queries[:, 0]]
        text_e = self.text_trans[queries[:, 0]]
        #Ended        

        rank = self.rank//2
        head_e = head_e[:, :rank//2], head_e[:, rank//2:rank], head_e[:, rank:3*rank//2], head_e[:, 3*rank//2:]

        
        rel_rot_e = img_rot_e[:, :rank//2], img_rot_e[:, rank//2:rank], text_rot_e[:, :rank//2], text_rot_e[:, rank//2:rank]
        
        A, B, C, D = givens_DE_rotations_m(rel_rot_e, head_e)
        h_e = torch.cat((A, B, C, D), 1)

        lhs_e = h_e + rel_e + img_e + text_e

        lhs_biases = self.bh(queries[:, 0])

        return lhs_e, lhs_biases

class Robin_W(BaseE):
    """Euclidean translations https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf"""
    def __init__(self, args):
        super(Robin_W, self).__init__(args)
        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        dataset_path = os.path.join(DATA_PATH, args.dataset)
        dataset = KGDataset(dataset_path, args.debug)

        image_embedd_data = dataset.get_entity_embedding_lm_data()
        text_embedd_data = dataset.get_text_embedd_data()
        
        self.dim_red_model_img = AdjustDim(in_dim = len(image_embedd_data[1]),out_dim=self.rank//2)
        self.dim_red_model_txt = AdjustDim(in_dim = len(text_embedd_data[1]),out_dim=self.rank//2)

        self.dim_red_model_img_tr = AdjustDim(in_dim = len(image_embedd_data[1]),out_dim=self.rank)
        self.dim_red_model_txt_tr = AdjustDim(in_dim = len(text_embedd_data[1]),out_dim=self.rank)

        self.orig_image_embeddings= image_embedd_data.to(self.cuda1)
        self.orig_text_embeddings = text_embedd_data.to(self.cuda1)

        self.sim = "dist"

    def get_queries(self, queries):
        img_data = self.dim_red_model_img(self.orig_image_embeddings)
        self.img_rot = img_data
        txt_data = self.dim_red_model_txt(self.orig_text_embeddings)
        self.text_rot = txt_data

        img_data = self.dim_red_model_img_tr(self.orig_image_embeddings)
        self.img_trans = img_data
        txt_data = self.dim_red_model_txt_tr(self.orig_text_embeddings)
        self.text_trans = txt_data

        head_e = self.entity(queries[:, 0])
        rel_e = self.rel(queries[:, 1])
        img_rot_e = self.img_rot[queries[:, 0]]
        text_rot_e = self.text_rot[queries[:, 0]]

        img_e = self.img_trans[queries[:, 0]]
        text_e = self.text_trans[queries[:, 0]]
        #Ended        

        rank = self.rank//2
        head_e = head_e[:, :rank//2], head_e[:, rank//2:rank], head_e[:, rank:3*rank//2], head_e[:, 3*rank//2:]

        
        rel_rot_e = img_rot_e[:, :rank//2], img_rot_e[:, rank//2:rank], text_rot_e[:, :rank//2], text_rot_e[:, rank//2:rank]
        
        A, B, C, D = givens_DE_rotations_m(rel_rot_e, head_e)
        h_e = torch.cat((A, B, C, D), 1)

        lhs_e = h_e + rel_e + img_e + text_e

        lhs_biases = self.bh(queries[:, 0])

        return lhs_e, lhs_biases

class Robin_D(BaseE):
    """Euclidean translations https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf"""
    def __init__(self, args):
        super(Robin_D, self).__init__(args)
        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        dataset_path = os.path.join(DATA_PATH, args.dataset)
        dataset = KGDataset(dataset_path, args.debug)

        image_embedd_data = dataset.get_entity_embedding_lm_data()
        text_embedd_data = dataset.get_doc2vec_embedd_data()

        self.dim_red_model_img = AdjustDim(in_dim = len(image_embedd_data[1]),out_dim=self.rank//2)
        self.dim_red_model_txt = AdjustDim(in_dim = len(text_embedd_data[1]),out_dim=self.rank//2)

        self.dim_red_model_img_tr = AdjustDim(in_dim = len(image_embedd_data[1]),out_dim=self.rank)
        self.dim_red_model_txt_tr = AdjustDim(in_dim = len(text_embedd_data[1]),out_dim=self.rank)

        self.orig_image_embeddings= image_embedd_data.cuda(self.cuda1)
        self.orig_text_embeddings = text_embedd_data.cuda(self.cuda1)

        self.sim = "dist"

    def get_queries(self, queries):
        img_data = self.dim_red_model_img(self.orig_image_embeddings)
        self.img_rot = img_data
        txt_data = self.dim_red_model_txt(self.orig_text_embeddings)
        self.text_rot = txt_data

        img_data = self.dim_red_model_img_tr(self.orig_image_embeddings)
        self.img_trans = img_data
        txt_data = self.dim_red_model_txt_tr(self.orig_text_embeddings)
        self.text_trans = txt_data

        head_e = self.entity(queries[:, 0])
        rel_e = self.rel(queries[:, 1])
        img_rot_e = self.img_rot[queries[:, 0]]
        text_rot_e = self.text_rot[queries[:, 0]]

        img_e = self.img_trans[queries[:, 0]]
        text_e = self.text_trans[queries[:, 0]]
        #Ended        

        rank = self.rank//2
        head_e = head_e[:, :rank//2], head_e[:, rank//2:rank], head_e[:, rank:3*rank//2], head_e[:, 3*rank//2:]

        
        rel_rot_e = img_rot_e[:, :rank//2], img_rot_e[:, rank//2:rank], text_rot_e[:, :rank//2], text_rot_e[:, rank//2:rank]
        
        A, B, C, D = givens_DE_rotations_m(rel_rot_e, head_e)
        h_e = torch.cat((A, B, C, D), 1)

        lhs_e = h_e + rel_e + img_e + text_e

        lhs_biases = self.bh(queries[:, 0])

        return lhs_e, lhs_biases

class Robin_F(BaseE):
    """Euclidean translations https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf"""
    def __init__(self, args):
        super(Robin_F, self).__init__(args)
        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        dataset_path = os.path.join(DATA_PATH, args.dataset)
        dataset = KGDataset(dataset_path, args.debug)

        image_embedd_data = dataset.get_entity_embedding_lm_data()
        text_embedd_data = dataset.get_fasttext_embedd_data()

        self.dim_red_model_img = AdjustDim(in_dim = len(image_embedd_data[1]),out_dim=self.rank//2)
        self.dim_red_model_txt = AdjustDim(in_dim = len(text_embedd_data[1]),out_dim=self.rank//2)

        self.dim_red_model_img_tr = AdjustDim(in_dim = len(image_embedd_data[1]),out_dim=self.rank)
        self.dim_red_model_txt_tr = AdjustDim(in_dim = len(text_embedd_data[1]),out_dim=self.rank)

        self.orig_image_embeddings= image_embedd_data.cuda(self.cuda1)
        self.orig_text_embeddings = text_embedd_data.cuda(self.cuda1)

        self.sim = "dist"

    def get_queries(self, queries):
        img_data = self.dim_red_model_img(self.orig_image_embeddings)
        self.img_rot = img_data
        txt_data = self.dim_red_model_txt(self.orig_text_embeddings)
        self.text_rot = txt_data

        img_data = self.dim_red_model_img_tr(self.orig_image_embeddings)
        self.img_trans = img_data
        txt_data = self.dim_red_model_txt_tr(self.orig_text_embeddings)
        self.text_trans = txt_data

        head_e = self.entity(queries[:, 0])
        rel_e = self.rel(queries[:, 1])
        img_rot_e = self.img_rot[queries[:, 0]]
        text_rot_e = self.text_rot[queries[:, 0]]

        img_e = self.img_trans[queries[:, 0]]
        text_e = self.text_trans[queries[:, 0]]
        #Ended        

        rank = self.rank//2
        head_e = head_e[:, :rank//2], head_e[:, rank//2:rank], head_e[:, rank:3*rank//2], head_e[:, 3*rank//2:]

        
        rel_rot_e = img_rot_e[:, :rank//2], img_rot_e[:, rank//2:rank], text_rot_e[:, :rank//2], text_rot_e[:, rank//2:rank]
        
        A, B, C, D = givens_DE_rotations_m(rel_rot_e, head_e)
        h_e = torch.cat((A, B, C, D), 1)

        lhs_e = h_e + rel_e + img_e + text_e

        lhs_biases = self.bh(queries[:, 0])

        return lhs_e, lhs_biases

class Lion_SD(BaseE):
    """Euclidean translations https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf"""
    def __init__(self, args):
        super(Lion_SD, self).__init__(args)
        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        dataset_path = os.path.join(DATA_PATH, args.dataset)
        dataset = KGDataset(dataset_path, args.debug)

        image_embedd_data = dataset.get_sentenseTransformer_embedd_data()
        text_embedd_data = dataset.get_doc2vec_embedd_data()
        
        self.dim_red_model_img = AdjustDim(in_dim = len(image_embedd_data[1]),out_dim=self.rank//2)
        self.dim_red_model_txt = AdjustDim(in_dim = len(text_embedd_data[1]),out_dim=self.rank//2)

        self.dim_red_model_img_tr = AdjustDim(in_dim = len(image_embedd_data[1]),out_dim=self.rank)
        self.dim_red_model_txt_tr = AdjustDim(in_dim = len(text_embedd_data[1]),out_dim=self.rank)

        self.orig_image_embeddings= image_embedd_data.to(self.cuda1)
        self.orig_text_embeddings = text_embedd_data.to(self.cuda1)

        self.sim = "dist"

    def get_queries(self, queries):
        img_data = self.dim_red_model_img(self.orig_image_embeddings)
        self.img_rot = img_data
        txt_data = self.dim_red_model_txt(self.orig_text_embeddings)
        self.text_rot = txt_data

        img_data = self.dim_red_model_img_tr(self.orig_image_embeddings)
        self.img_trans = img_data
        txt_data = self.dim_red_model_txt_tr(self.orig_text_embeddings)
        self.text_trans = txt_data

        head_e = self.entity(queries[:, 0])
        rel_e = self.rel(queries[:, 1])
        img_rot_e = self.img_rot[queries[:, 0]]
        text_rot_e = self.text_rot[queries[:, 0]]

        img_e = self.img_trans[queries[:, 0]]
        text_e = self.text_trans[queries[:, 0]]
        #Ended        

        rank = self.rank//2
        head_e = head_e[:, :rank//2], head_e[:, rank//2:rank], head_e[:, rank:3*rank//2], head_e[:, 3*rank//2:]

        
        rel_rot_e = img_rot_e[:, :rank//2], img_rot_e[:, rank//2:rank], text_rot_e[:, :rank//2], text_rot_e[:, rank//2:rank]
        
        A, B, C, D = givens_DE_rotations_m(rel_rot_e, head_e)
        h_e = torch.cat((A, B, C, D), 1)

        lhs_e = h_e + rel_e + img_e + text_e

        lhs_biases = self.bh(queries[:, 0])

        return lhs_e, lhs_biases

class Lion_FS(BaseE):
    """Euclidean translations https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf"""
    def __init__(self, args):
        super(Lion_FS, self).__init__(args)
        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        dataset_path = os.path.join(DATA_PATH, args.dataset)
        dataset = KGDataset(dataset_path, args.debug)

        fasttext_embedd_data = dataset.get_fasttext_embedd_data()
        sentenseTransformer_embedd_data = dataset.get_sentenseTransformer_embedd_data()
        
        self.dim_red_model_fasttext = AdjustDim(in_dim = len(fasttext_embedd_data[1]),out_dim=self.rank//2)
        self.dim_red_model_sentensetrans = AdjustDim(in_dim = len(sentenseTransformer_embedd_data[1]),out_dim=self.rank//2)
        
        
        self.dim_red_model_fasttext_tr = AdjustDim(in_dim = len(fasttext_embedd_data[1]),out_dim=self.rank)
        self.dim_red_model_sentensetrans_tr = AdjustDim(in_dim = len(sentenseTransformer_embedd_data[1]),out_dim=self.rank)

        self.orig_fasttext_embedd_data = fasttext_embedd_data.cuda(self.cuda1)
        self.orig_sentenseTransformer_embedd_data = sentenseTransformer_embedd_data.cuda(self.cuda1)

        self.sim = "dist"

    def get_queries(self, queries):

        fasttext_data = self.dim_red_model_fasttext(self.orig_fasttext_embedd_data)
        self.fasttext_rot = fasttext_data
        sentence_data = self.dim_red_model_sentensetrans(self.orig_sentenseTransformer_embedd_data)
        self.sentence_data_rot = sentence_data
        
        fasttext_data = self.dim_red_model_fasttext_tr(self.orig_fasttext_embedd_data)
        self.fasttext_trans = fasttext_data
        sentence_data = self.dim_red_model_sentensetrans_tr(self.orig_sentenseTransformer_embedd_data)
        self.sentence_data_trans = sentence_data
        

        head_e = self.entity(queries[:, 0])
        rel_e = self.rel(queries[:, 1])
        fasttext_rot_e = self.fasttext_rot[queries[:, 0]]
        sentence_rot_e = self.sentence_data_rot[queries[:, 0]]

        fasttext_e = self.fasttext_trans[queries[:, 0]]
        sentence_e = self.sentence_data_trans[queries[:, 0]]

        rank = self.rank//2
        head_e = head_e[:, :rank//2], head_e[:, rank//2:rank], head_e[:, rank:3*rank//2], head_e[:, 3*rank//2:]

        
        rel_rot_e = fasttext_rot_e[:, :rank//2], fasttext_rot_e[:, rank//2:rank], sentence_rot_e[:, :rank//2], sentence_rot_e[:, rank//2:rank]
        A, B, C, D = givens_DE_rotations_m(rel_rot_e, head_e)

        h_e = torch.cat((A, B, C, D), 1)
        lhs_e = h_e + rel_e + fasttext_e + sentence_e

        lhs_biases = self.bh(queries[:, 0])

        return lhs_e, lhs_biases

class AttE(BaseE):
    """Euclidean attention model combining translations, reflections and rotations"""

    def __init__(self, args):
        super(AttE, self).__init__(args)
        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.sim = "dist"

        # reflection
        self.ref = nn.Embedding(self.sizes[1], self.rank)
        self.ref.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0

        # rotation
        self.rot = nn.Embedding(self.sizes[1], self.rank)
        self.rot.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0

        # attention
        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.act = nn.Softmax(dim=1)
        self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).cuda(self.cuda1)

    def get_reflection_queries(self, queries):
        lhs_ref_e = givens_reflection(
            self.ref(queries[:, 1]), self.entity(queries[:, 0])
        )
        return lhs_ref_e

    def get_rotation_queries(self, queries):
        lhs_rot_e = givens_rotations(
            self.rot(queries[:, 1]), self.entity(queries[:, 0])
        )
        return lhs_rot_e

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        lhs_ref_e = self.get_reflection_queries(queries).view((-1, 1, self.rank))
        lhs_rot_e = self.get_rotation_queries(queries).view((-1, 1, self.rank))

        # self-attention mechanism
        cands = torch.cat([lhs_ref_e, lhs_rot_e], dim=1)
        context_vec = self.context_vec(queries[:, 1]).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        lhs_e = torch.sum(att_weights * cands, dim=1) + self.rel(queries[:, 1])
        return lhs_e, self.bh(queries[:, 0])

class Robin_S_Quat(BaseE):
    def __init__(self, args):
        super(Robin_S_Quat, self).__init__(args)
        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        dataset_path = os.path.join(DATA_PATH, args.dataset)
        dataset = KGDataset(dataset_path, args.debug)

        image_embedd_data = dataset.get_entity_embedding_lm_data()
        text_embedd_data = dataset.get_sentenseTransformer_embedd_data()
        #self.img_ln = nn.Parameter(image_embedd_data).requires_grad_(False)

        self.dim_red_model_img = AdjustDim(in_dim = len(image_embedd_data[1]),out_dim=self.rank//2)
        self.dim_red_model_txt = AdjustDim(in_dim = len(text_embedd_data[1]),out_dim=self.rank//2)

        self.dim_red_model_img_tr = AdjustDim(in_dim = len(image_embedd_data[1]),out_dim=self.rank)
        self.dim_red_model_txt_tr = AdjustDim(in_dim = len(text_embedd_data[1]),out_dim=self.rank)

        self.orig_image_embeddings= image_embedd_data.to(self.cuda1)
        self.orig_text_embeddings = text_embedd_data.to(self.cuda1)
        
        self.sim = "dist"

    def get_queries(self, queries):
        img_data = self.dim_red_model_img(self.orig_image_embeddings)
        self.img_rot = img_data
        txt_data = self.dim_red_model_txt(self.orig_text_embeddings)
        self.text_rot = txt_data

        img_data = self.dim_red_model_img_tr(self.orig_image_embeddings)
        self.img_trans = img_data
        txt_data = self.dim_red_model_txt_tr(self.orig_text_embeddings)
        self.text_trans = txt_data

        head_e = self.entity(queries[:, 0])
        rel_e = self.rel(queries[:, 1])
        img_rot_e = self.img_rot[queries[:, 0]]
        text_rot_e = self.text_rot[queries[:, 0]]

        img_e = self.img_trans[queries[:, 0]]
        text_e = self.text_trans[queries[:, 0]]
        #Ended        

        rank = self.rank//2
        head_e = head_e[:, :rank//2], head_e[:, rank//2:rank], head_e[:, rank:3*rank//2], head_e[:, 3*rank//2:]

        
        rel_rot_e = img_rot_e[:, :rank//2], img_rot_e[:, rank//2:rank], text_rot_e[:, :rank//2], text_rot_e[:, rank//2:rank]
        
        A, B, C, D = givens_QuatE_rotations(rel_rot_e, head_e)
        h_e = torch.cat((A, B, C, D), 1)

        lhs_e = h_e + rel_e + img_e + text_e

        lhs_biases = self.bh(queries[:, 0])

        return lhs_e, lhs_biases

class Lion_FS_Quat(BaseE):
    """Euclidean translations https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf"""
    def __init__(self, args):
        super(Lion_FS_Quat, self).__init__(args)
        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        dataset_path = os.path.join(DATA_PATH, args.dataset)
        dataset = KGDataset(dataset_path, args.debug)

        fasttext_embedd_data = dataset.get_fasttext_embedd_data()
        sentenseTransformer_embedd_data = dataset.get_sentenseTransformer_embedd_data()
        
        self.dim_red_model_fasttext = AdjustDim(in_dim = len(fasttext_embedd_data[1]),out_dim=self.rank//2)
        self.dim_red_model_sentensetrans = AdjustDim(in_dim = len(sentenseTransformer_embedd_data[1]),out_dim=self.rank//2)
        
        
        self.dim_red_model_fasttext_tr = AdjustDim(in_dim = len(fasttext_embedd_data[1]),out_dim=self.rank)
        self.dim_red_model_sentensetrans_tr = AdjustDim(in_dim = len(sentenseTransformer_embedd_data[1]),out_dim=self.rank)

        self.orig_fasttext_embedd_data = fasttext_embedd_data.cuda(self.cuda1)
        self.orig_sentenseTransformer_embedd_data = sentenseTransformer_embedd_data.cuda(self.cuda1)

        self.sim = "dist"

    def get_queries(self, queries):

        fasttext_data = self.dim_red_model_fasttext(self.orig_fasttext_embedd_data)
        self.fasttext_rot = fasttext_data
        sentence_data = self.dim_red_model_sentensetrans(self.orig_sentenseTransformer_embedd_data)
        self.sentence_data_rot = sentence_data
        
        fasttext_data = self.dim_red_model_fasttext_tr(self.orig_fasttext_embedd_data)
        self.fasttext_trans = fasttext_data
        sentence_data = self.dim_red_model_sentensetrans_tr(self.orig_sentenseTransformer_embedd_data)
        self.sentence_data_trans = sentence_data
        

        head_e = self.entity(queries[:, 0])
        rel_e = self.rel(queries[:, 1])
        fasttext_rot_e = self.fasttext_rot[queries[:, 0]]
        sentence_rot_e = self.sentence_data_rot[queries[:, 0]]

        fasttext_e = self.fasttext_trans[queries[:, 0]]
        sentence_e = self.sentence_data_trans[queries[:, 0]]

        rank = self.rank//2
        head_e = head_e[:, :rank//2], head_e[:, rank//2:rank], head_e[:, rank:3*rank//2], head_e[:, 3*rank//2:]

        
        rel_rot_e = fasttext_rot_e[:, :rank//2], fasttext_rot_e[:, rank//2:rank], sentence_rot_e[:, :rank//2], sentence_rot_e[:, rank//2:rank]
        A, B, C, D = givens_QuatE_rotations(rel_rot_e, head_e)

        h_e = torch.cat((A, B, C, D), 1)
        lhs_e = h_e + rel_e + fasttext_e + sentence_e

        lhs_biases = self.bh(queries[:, 0])

        return lhs_e, lhs_biases

class Tetra_WSF_Quat(BaseE):
    def __init__(self, args):
        super(Tetra_WSF_Quat, self).__init__(args)
        
        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        dataset_path = os.path.join(DATA_PATH, args.dataset)
        dataset = KGDataset(dataset_path, args.debug)

        text_embedd_data = dataset.get_text_embedd_data()
        fasttext_embedd_data = dataset.get_fasttext_embedd_data()
        sentenseTransformer_embedd_data = dataset.get_sentenseTransformer_embedd_data()
        
        self.dim_red_model_txt = AdjustDim(in_dim = len(text_embedd_data[1]),out_dim=self.rank//4)
        self.dim_red_model_fstxt = AdjustDim(in_dim = len(fasttext_embedd_data[1]),out_dim=self.rank//4)
        self.dim_red_model_st = AdjustDim(in_dim = len(sentenseTransformer_embedd_data[1]),out_dim=self.rank//4)
        
        self.orig_text_embeddings = text_embedd_data.cuda(self.cuda1)
        self.orig_fstxt_emb_data = fasttext_embedd_data.cuda(self.cuda1)
        self.orig_st_emb_data = sentenseTransformer_embedd_data.cuda(self.cuda1)

        self.sim = "dist"

    def get_queries(self, queries):
        word2vec_data = self.dim_red_model_txt(self.orig_text_embeddings)
        fasttext_data = self.dim_red_model_fstxt(self.orig_fstxt_emb_data)
        sentence_data = self.dim_red_model_st(self.orig_st_emb_data)
        
        h_s = self.entity(queries[:, 0])
        h_x = word2vec_data[queries[:, 0]]
        h_y = fasttext_data[queries[:, 0]]
        h_z = sentence_data[queries[:, 0]]
        rel = self.rel(queries[:, 1])
        
        rank = self.rank//4
        head_e = h_s[:,:rank], h_x[:,:rank], h_y[:,:rank], h_z[:,:rank]
        rel_e = rel[:,:rank],rel[:,rank:2*rank],rel[:,2*rank:3*rank],rel[:,3*rank:]
        
        A, B, C, D = givens_QuatE_rotations(rel_e, head_e)
        lhs_e = torch.cat((A, B, C, D), 1)
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases

    def get_rhs(self, queries, eval_mode):        
        word2vec_data = self.dim_red_model_txt(self.orig_text_embeddings)
        fasttext_data = self.dim_red_model_fstxt(self.orig_fstxt_emb_data)
        sentence_data = self.dim_red_model_st(self.orig_st_emb_data)
        rank = self.rank//4
        
        if eval_mode:
            rhs_e = torch.cat((self.entity.weight[:, :rank], word2vec_data, fasttext_data, sentence_data), 1)
            return rhs_e, self.bt.weight 
        else:
            t_s = self.entity(queries[:, 2])[:, :rank]
            t_x = word2vec_data[queries[:, 2]]
            t_y = fasttext_data[queries[:, 2]]
            t_z = sentence_data[queries[:, 2]]
            rhs_e = torch.cat((t_s, t_x, t_y, t_z), 1)
            rhs_biases = self.bt(queries[:, 2])
            return rhs_e, rhs_biases

class Tetra_WSF(BaseE):
    def __init__(self, args):
        super(Tetra_WSF, self).__init__(args)
        
        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        dataset_path = os.path.join(DATA_PATH, args.dataset)
        dataset = KGDataset(dataset_path, args.debug)

        text_embedd_data = dataset.get_text_embedd_data()
        fasttext_embedd_data = dataset.get_fasttext_embedd_data()
        sentenseTransformer_embedd_data = dataset.get_sentenseTransformer_embedd_data()
        
        self.dim_red_model_txt = AdjustDim(in_dim = len(text_embedd_data[1]),out_dim=self.rank//4)
        self.dim_red_model_fstxt = AdjustDim(in_dim = len(fasttext_embedd_data[1]),out_dim=self.rank//4)
        self.dim_red_model_st = AdjustDim(in_dim = len(sentenseTransformer_embedd_data[1]),out_dim=self.rank//4)
        
        self.orig_text_embeddings = text_embedd_data.cuda(self.cuda1)
        self.orig_fstxt_emb_data = fasttext_embedd_data.cuda(self.cuda1)
        self.orig_st_emb_data = sentenseTransformer_embedd_data.cuda(self.cuda1)

        self.sim = "dist"

    def get_queries(self, queries):
        word2vec_data = self.dim_red_model_txt(self.orig_text_embeddings)
        fasttext_data = self.dim_red_model_fstxt(self.orig_fstxt_emb_data)
        sentence_data = self.dim_red_model_st(self.orig_st_emb_data)
        
        rank = self.rank//4
        rel = self.rel(queries[:, 1])

        h_s = self.entity(queries[:, 0])
        h_x = self.init_size * word2vec_data[queries[:, 0]]
        # h_x = self.entity(queries[:, 0])[:, rank:2*rank]
        h_y = self.init_size *fasttext_data[queries[:, 0]]
        h_z = self.init_size *sentence_data[queries[:, 0]]

        head_e = h_s[:,:rank], h_x, h_y, h_z
        rel_e = rel[:,:rank],rel[:,rank:2*rank],rel[:,2*rank:3*rank],rel[:,3*rank:]
        
        A, B, C, D = givens_DE_rotations_m(rel_e, head_e)
        lhs_e = torch.cat((A, B, C, D), 1)
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases

    def get_rhs(self, queries, eval_mode):        
        word2vec_data = self.dim_red_model_txt(self.orig_text_embeddings)
        fasttext_data = self.dim_red_model_fstxt(self.orig_fstxt_emb_data)
        sentence_data = self.dim_red_model_st(self.orig_st_emb_data)
        rank = self.rank//4
        
        if eval_mode:
            rhs_e = torch.cat((self.entity.weight[:, :rank], self.init_size * word2vec_data, self.init_size *fasttext_data, self.init_size *sentence_data), 1)
            #rhs_e = torch.cat((self.entity.weight[:, :rank], self.entity.weight[:, rank:2*rank], fasttext_data, sentence_data), 1)
            return rhs_e, self.bt.weight 
        else:
            t_s = self.entity(queries[:, 2])[:, :rank]
            t_x = self.init_size * word2vec_data[queries[:, 2]]
            #t_x = self.entity(queries[:, 2])[:, rank:2*rank]
            t_y = fasttext_data[queries[:, 2]]
            t_z = sentence_data[queries[:, 2]]
            rhs_e = torch.cat((t_s, t_x, t_y, t_z), 1)
            rhs_biases = self.bt(queries[:, 2])
            return rhs_e, rhs_biases



class Tetra_SF(BaseE):
    def __init__(self, args):
        super(Tetra_SF, self).__init__(args)
        
        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        dataset_path = os.path.join(DATA_PATH, args.dataset)
        dataset = KGDataset(dataset_path, args.debug)

        fasttext_embedd_data = dataset.get_fasttext_embedd_data()
        sentenseTransformer_embedd_data = dataset.get_sentenseTransformer_embedd_data()
        
        self.dim_red_model_fstxt = AdjustDim(in_dim = len(fasttext_embedd_data[1]),out_dim=self.rank//4)
        self.dim_red_model_st = AdjustDim(in_dim = len(sentenseTransformer_embedd_data[1]),out_dim=self.rank//4)
        
        self.orig_fstxt_emb_data = fasttext_embedd_data.cuda(self.cuda1)
        self.orig_st_emb_data = sentenseTransformer_embedd_data.cuda(self.cuda1)

        self.sim = "dist"

    def get_queries(self, queries):
        fasttext_data = self.dim_red_model_fstxt(self.orig_fstxt_emb_data)
        sentence_data = self.dim_red_model_st(self.orig_st_emb_data)
        
        h_s = self.entity(queries[:, 0])
        h_x = self.entity(queries[:, 0])
        h_y = fasttext_data[queries[:, 0]]
        h_z = sentence_data[queries[:, 0]]
        rel = self.rel(queries[:, 1])
        
        rank = self.rank//4
        head_e = h_s[:,:rank], h_x[:, rank:2 * rank], h_y[:,:rank], h_z[:,:rank]
        rel_e = rel[:,:rank],rel[:,rank:2*rank],rel[:,2*rank:3*rank],rel[:,3*rank:]
        
        A, B, C, D = givens_DE_rotations_m(rel_e, head_e)
        lhs_e = torch.cat((A, B, C, D), 1)
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases

    def get_rhs(self, queries, eval_mode):        
        fasttext_data = self.dim_red_model_fstxt(self.orig_fstxt_emb_data)
        sentence_data = self.dim_red_model_st(self.orig_st_emb_data)
        rank = self.rank//4
        
        if eval_mode:
            rhs_e = torch.cat((self.entity.weight[:, :rank], self.entity.weight[:, rank:2*rank], fasttext_data, sentence_data), 1)
            return rhs_e, self.bt.weight 
        else:
            t_s = self.entity(queries[:, 2])[:, :rank]
            t_x = self.entity(queries[:, 2])[:, rank:2*rank]
            t_y = fasttext_data[queries[:, 2]]
            t_z = sentence_data[queries[:, 2]]
            rhs_e = torch.cat((t_s, t_x, t_y, t_z), 1)
            rhs_biases = self.bt(queries[:, 2])
            return rhs_e, rhs_biases



