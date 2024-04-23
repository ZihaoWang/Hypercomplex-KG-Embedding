# Hypercomplex Knowledge Graph Embeddings with Multiple Language Models

Here is the code of our paper [Integrating Knowledge Graph embedding and pretrained Language Models in Hypercomplex Spaces](https://arxiv.org/abs/2208.02743).

1. Please download the dataset at https://drive.google.com/file/d/1Za6KCacWVJe3aTTCD-5zTVbNKUHkoGBw/view?usp=drive_link

2. Unzip the dataset, put the data/ folder under here.

3. Run set_env.sh

3. Run our model and baselines like:
--datasets: can be [nations, diabetes, fb15k237, yago10]
--models: can be [TransE, ComplEx, AttE, AttH, Lion_FS, Lion_SD, Robin_S, Robin_W, Robin_D, Robin_F, Tetra_zero, Tetra_SF, Tetra_WSF, Robin_S_Quat, Lion_FS_Quat, Tetra_WSF_Quat]
--cuda_n: -1 means CPU; 0, 1, 2 ... means GPU.
--rank: dimension of embeddings
For hyper-parameters to reproduce our results, please see the Appendix.

python run.py --model Tetra --rank 32 --neg_sample_size 100 --batch_size 400 --learning_rate 0.01 --cuda_n 3 --dataset nations
