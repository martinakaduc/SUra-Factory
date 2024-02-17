import json
import datasets
import pandas as pd
from typing import List
import random

_DESCRIPTION = "OPUS100 Dataset for DPO"
_CITATION = ""
_HOMEPAGE = "https://huggingface.co/datasets/ura-hcmut/OPUS100-dpo"
_LICENSE = "mit"
_URL = "https://huggingface.co/datasets/ura-hcmut/OPUS100-dpo/resolve/main/"
_URLS = {
    "train": [
        _URL + "opus100_vien-dpo.json",
    ],
}


class OPUS100ViEnDPO(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("0.0.0")

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features({
            "system": datasets.Value("string"),
            "instruction": datasets.Value("string"),
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
            df = pd.read_json(filepath)
            for key, row in df.iterrows():
                chosen = random.choice(row["chosen"])
                rejected = random.choice(row["rejected"]) if len(row["rejected"]) > 0 else ""

                yield key, {
                    "system": row["system"],
                    "instruction": row["instruction"],
                    "response": [chosen, rejected],
                    "chosen": chosen,
                    "rejected": rejected
                }
