# diatom-recogintion
数据集: https://www.kaggle.com/datasets/siyuepu/diatom-datasets.

预训练模型: https://huggingface.co/google/vit-base-patch16-224-in21k

    # Make sure you have git-lfs installed (https://git-lfs.com)
    git lfs install
    git clone https://huggingface.co/google/vit-base-patch16-224-in21k

    # if you want to clone without large files – just their pointers
    # prepend your git clone with the following env var:
    GIT_LFS_SKIP_SMUDGE=1

pip install -r requirements 

python ./train.py 

python ./finetune_vit.py 

python ./txt_annotation.py 

python ./eval_top1.py 

python ./eval_finetune_vit.py 

