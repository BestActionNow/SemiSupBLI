# Semi-Supervised Bilingual Lexicon Induction with Two-Way Message Passing Mechanisms

In this repository, We present the implementation of our two poposed semi-supervised approches **CSS** and **PSS** for BLI. 

## Dependencies
* python 3
* Pytorch
* Numpy 
* Faiss

## How to get the datasets
You need to download the **MUSE** dataset from [here](https://github.com/facebookresearch/MUSE) to the **./muse_data** directory. 

You need to  download the **VecMap** dataset from [here](https://github.com/artetxem/vecmap) to the **./vecmap_data** directory.

## How to run
You can run the following command to evaluate **CSS** on the MUSE dataset with "5k all" annotated lexicon:

```
python main.py --config_file ./configs/config-CSS-muse-en-es-5kall.yaml
```

You can run the following command to evaluate **PSS** on the VecMap dataset with "5k all" annotated lexicon:

```
python main.py --config_file ./configs/config-PSS-vecmap-en-es-5kall.yaml
```

## Configuration
Then we briefly discribe some important fields in the configuration file:<br>
* **"method"**" specifies the model to evaludate. "CSSBli" for **CSS** or "PSSBli" for **PSS**.<br>
* **"src"** and **"tgt"** indicate the source and target languages of BLI task.<br>
* **"data_params/data_dir"** specifies which dataset to use where "./muse_data/" for MUSE or "./vecmap_data/" for VevMap.<br>
* **"supervised/max_count"** indicates the size of annotated lexicon where "-1" for "5k all", "100" for "100 unique" and "5000" for "5000 unique".<br>

Other fields specify the hyperparameters for **CSS** and **PSS**.



