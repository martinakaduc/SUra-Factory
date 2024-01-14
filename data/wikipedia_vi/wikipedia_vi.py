import json
import datasets
import pandas as pd
from typing import List

_DESCRIPTION = "Vietnamese Wikipedia"
_CITATION = ""
_HOMEPAGE = "https://huggingface.co/datasets/wikimedia/wikipedia"
_LICENSE = "mit"
_URL = "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.vi/"
_URLS = {
    "train": [
        _URL + "train-00000-of-00004.parquet",
        _URL + "train-00001-of-00004.parquet",
        _URL + "train-00002-of-00004.parquet",
        _URL + "train-00003-of-00004.parquet",
    ],
}


class WikipediaVi(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("0.0.0")

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features({
            "text": datasets.Value("string")
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
        for file in filepaths:
            df = pd.read_parquet(file)
            for key, row in df.iterrows():
                yield key, {
                    "text": row["text"].split("Liên kết ngoài")[0].split("Xem thêm")[0].split("Tham khảo")[0].split("Chú thích")[0].strip()
                }
