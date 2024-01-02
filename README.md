# Caseformer
**Source code of our long paper:** 

**Caseformer: Pre-training for Legal Case Retrieval**

[GitHub Link](https://github.com/oneal2000/Caseformer)

<p align="center">
  <img src="https://github.com/oneal2000/Caseformer/blob/main/pics/logo2.png" width="50%" height="auto" />
</p>

```
@article{su2023caseformer,
  title={Caseformer: Pre-training for Legal Case Retrieval},
  author={Su, Weihang and Ai, Qingyao and Wu, Yueyue and Ma, Yixiao and Li, Haitao and Liu, Yiqun},
  journal={arXiv preprint arXiv:2311.00333},
  year={2023}
}
```

### The file structure of this repository:

```
.
└── caseformer
    ├── data_preprocess
    │   ├── crime_extraction.py
    │   └── law_article_extration.py
    ├── demo_data
    │   ├── legal_documents
    │   │   ├── file_format.txt
    │   │   └── legal_documents.jsonl
    │   └── preprocessed_training_data
    │       ├── FDM_task.jsonl
    │       ├── file_format.txt
    │       └── LJP_task.jsonl
    ├── pre-training
    │   ├── pre-train_reranker.sh
    │   └── pre-train_retriever.sh
    ├── pre-training_data_generation
    │   ├── calc_LP-ICF_score.py
    │   ├── demo_data
    │   │   ├── bm25_top100.jsonl
    │   │   ├── extracted_crimes.jsonl
    │   │   ├── extracted_law_articles.jsonl
    │   │   └── LP-ICF_top100.jsonl
    │   ├── generate_FDM_task_data.py
    │   └── generate_LJP_task_data.py
    ├── README.md
    └── requirements.txt

```


### Pre-installation

```
git clone git@github.com:caseformer/caseformer.git
cd caseformer
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```



### Extract structured information from legal documents

#### Extract law articles

```
cd caseformer
python ./data_preprocess/law_article_extraction.py \
--path_to_documents your_path \
--output_path your_path
```

Format of the input documents:

```
{"docID":string,"content":string}
{"docID":string,"content":string}
{"docID":string,"content":string}
......
{"docID":string,"content":string}
```



#### Extract Crimes

```
cd caseformer
python ./data_preprocess/crime_extraction.py \
--path_to_documents your_path \
--output_path your_path
```

Format of the input documents:

```
{"docID":string,"content":string}
{"docID":string,"content":string}
{"docID":string,"content":string}
......
{"docID":string,"content":string}
```



### Prepare the Training Data

#### LJP Task

```
cd caseformer
python ./pre-training_data_generation/generate_LJP_task_data.py \
--BM25_top_100  path \
--law_articles path \
--crimes path \
--output_path your_path
```


#### FDM Task

```
cd caseformer
python ./pre-training_data_generation/generate_FDM_task_data.py \
--LP-ICF_top_100  path \
--law_articles path \
--crimes path \
--output_path your_path
```


### Running Pre-training

We will disclose the complete code and data in this repository.

