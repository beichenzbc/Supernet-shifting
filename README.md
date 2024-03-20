# Supernet-shifting

## Introduction
The official code for [**Boosting Order-Preserving and Transferability for Neural Architecture Search: a Joint Architecture Refined Search and Fine-tuning Approach**](https://export.arxiv.org/abs/2403.11380) (accepted by **CVPR2024**)


## Train From Scratch
### 1.Set Up Dataset and Prepare Flops Table
Download the ImageNet Dataset and move the images to labeled folders.
Download the Flops table used in Flops calculation. It is proposed by SPOS and can be founded in [Link](https://1drv.ms/f/s!AtjF1mI6H6IrbY2ODayNOdsc8SQ?e=Cx955o)
The structure of dataset should be
```
data
|--- train                 ImageNet Training Dataset
|--- val                   ImageNet Validation Dataset
|--- op_flops_dict.pkl     Flops Table
```
### 2.Train Supernet
Train the supernet with the following command:

```bash
cd supernet
python3 train.py --train-dir $YOUR_TRAINDATASET_PATH --val-dir $YOUR_VALDATASET_PATH
```

### 3.Supernet Shifting and Architecture Seaching
First, change the data root in `imagenet_dataset.py`
Apply supernet shifting and architecture searching in the following command
```bash
cd search
python3 search.py
```
If you want to transfer the supernet weight to a new dataset, first, change the data root and the dataloader in `imagenet_dataset.py`, then run the following comand
```bash
cd search
python3 search.py --new_dataset True --n_class $new_dataset_classes 
```
### 4.Get Searched Architecture
Get searched architecture with the following command:
```bash
cd evaluation
python3 eval.py
```
### 5. Train from Scratch

Finally, train and evaluate the searched architecture with the following command.
```bash
cd evaluation/data/$YOUR_ARCHITECTURE
python3 train.py --train-dir $YOUR_TRAINDATASET_PATH --val-dir $YOUR_VALDATASET_PATH
```

## Citation

If you use these models in your research, please cite:

```
@article{zhang2024boosting,
        title={Boosting Order-Preserving and Transferability for Neural Architecture Search: a Joint Architecture Refined Search and Fine-tuning Approach},
        author={Beichen Zhang and Xiaoxing Wang and Xiaohan Qin and Junchi Yan},
        journal={arXiv preprint arXiv:2403.11380},
        year={2024}
}
```
