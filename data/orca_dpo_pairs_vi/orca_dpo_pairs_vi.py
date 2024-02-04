import json
import datasets
import pandas as pd
from typing import List

_DESCRIPTION = "ORCA DPO Pairs in Vietnamese"
_CITATION = ""
_HOMEPAGE = "https://huggingface.co/datasets/ura-hcmut/orca_dpo_pairs"
_LICENSE = "mit"
_URL = "https://huggingface.co/datasets/ura-hcmut/orca_dpo_pairs/resolve/main/"
_URLS = {
    "train": [
        _URL + "orca_dpo_pairs_train_filtered.csv",
    ],
}


class OrcaDPOPairsVi(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("0.0.0")

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features({
            "system": datasets.Value("string"),
            "question": datasets.Value("string"),
            "response": datasets.Sequence(datasets.Value("string")),
            "chosen": datasets.Value("string"),
            "rejected": datasets.Value("string"),
        })
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        file_path = dl_manager.download_and_extract(_URLS)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepaths": file_path["train"]
                }
            ),
        ]

    def _generate_examples(self, filepaths: List[str]):
        for filepath in filepaths:
            df = pd.read_csv(filepath)
            for key, row in df.iterrows():
                chosen = str(row["chosen"])
                rejected = str(row["rejected"])
                
                yield key, {
                    "system": row["system"],
                    "question": row["question"],
                    "response": [chosen, rejected],
                    "chosen": chosen,
                    "rejected": rejected
                    
                }
