# coding=utf-8
# Lint as: python3
"""The Caption Contest benchmark."""


import json
import os
import datasets
import base64
import pprint


_CAPTION_CONTEST_TASKS_CITATION = """\
@article{hessel2022androids,
  title={Do Androids Laugh at Electric Sheep? Humor" Understanding" Benchmarks from The New Yorker Caption Contest},
  author={Hessel, Jack and Marasovi{\'c}, Ana and Hwang, Jena D and Lee, Lillian and Da, Jeff and Zellers, Rowan and Mankoff, Robert and Choi, Yejin},
  journal={arXiv preprint arXiv:2209.06293},
  year={2022}
}

www.capcon.dev

Our data contributions are:

- The cartoon-level annotations;
- The joke explanations;
- and the framing of the tasks
We release these data we contribute under CC-BY (see DATASET_LICENSE).

If you find this data useful in your work, in addition to citing our contributions, please also cite the following, from which the cartoons/captions in our corpus are derived:

@misc{newyorkernextmldataset,
  author={Jain, Lalit  and Jamieson, Kevin and Mankoff, Robert and Nowak, Robert and Sievert, Scott},
  title={The {N}ew {Y}orker Cartoon Caption Contest Dataset},
  year={2020},
  url={https://nextml.github.io/caption-contest-data/}
}

@inproceedings{radev-etal-2016-humor,
  title = "Humor in Collective Discourse: Unsupervised Funniness Detection in The {New Yorker} Cartoon Caption Contest",
  author = "Radev, Dragomir  and
      Stent, Amanda  and
      Tetreault, Joel  and
      Pappu, Aasish  and
      Iliakopoulou, Aikaterini  and
      Chanfreau, Agustin  and
      de Juan, Paloma  and
      Vallmitjana, Jordi  and
      Jaimes, Alejandro  and
      Jha, Rahul  and
      Mankoff, Robert",
  booktitle = "LREC",
  year = "2016",
}

@inproceedings{shahaf2015inside,
  title={Inside jokes: Identifying humorous cartoon captions},
  author={Shahaf, Dafna and Horvitz, Eric and Mankoff, Robert},
  booktitle={KDD},
  year={2015},
}
"""


_CAPTION_CONTEST_DESCRIPTION = """\
There are 3 caption contest tasks, described in the paper. In the Matching multiple choice task, models must recognize a caption written about a cartoon (vs. options that were not). In the Quality Ranking task, models must evaluate the quality
of that caption by scoring it more highly than a lower quality option from the same contest. In the Explanation Generation task, models must explain why the joke is funny.
"""

_MATCHING_DESCRIPTION = """\
You are given a cartoon and 5 captions. Only one of the captions was truly written about the cartoon. You must select it.
"""

_RANKING_DESCRIPTION = """\
You are given a cartoon and 2 captions. One of the captions was selected by crowd voting or New Yorker editors as high quality. You must select it.
"""

_EXPLANATION_DESCRIPTION = """\
You are given a cartoon and a caption that was written about it. You must autoregressively generate a joke explanation.
"""


_IMAGES_URL = "https://storage.googleapis.com/ai2-jack-public/caption_contest_data_public/all_contest_images.zip"


def _get_configs_crossvals():
    cross_val_configs = []
    for split_idx in [1,2,3,4]:
        cur_split_configs = [
            CaptionContestConfig(
            name='matching_{}'.format(split_idx),
                description=_MATCHING_DESCRIPTION,
                features=[
                    'image',
                    'contest_number',
                    'image_location',
                    'image_description',
                    'image_uncanny_description',
                    'entities',
                    'questions',
                    'caption_choices',
                    'from_description',
                ],
                label_classes=["A", "B", "C", "D", "E"],
                data_url='https://storage.googleapis.com/ai2-jack-public/caption_contest_data_public/huggingface_hub/v1.0/matching_{}.zip'.format(split_idx),
                url='www.capcon.dev',
                citation=_CAPTION_CONTEST_TASKS_CITATION,
            ),

            CaptionContestConfig(
                name='matching_from_pixels_{}'.format(split_idx),
                description=_MATCHING_DESCRIPTION,
                features=[
                    'image',
                    'contest_number',
                    'caption_choices',
                ],
                label_classes=["A", "B", "C", "D", "E"],
                data_url='https://storage.googleapis.com/ai2-jack-public/caption_contest_data_public/huggingface_hub/v1.0/matching_from_pixels_{}.zip'.format(split_idx),
                url='www.capcon.dev',
                citation=_CAPTION_CONTEST_TASKS_CITATION,
            ),

            CaptionContestConfig(
                name='ranking_{}'.format(split_idx),
                description=_RANKING_DESCRIPTION,
                features=[
                    'image',
                    'contest_number',
                    'image_location',
                    'image_description',
                    'image_uncanny_description',
                    'entities',
                    'questions',
                    'caption_choices',
                    'from_description',
                    'winner_source',
                ],
                
                label_classes=["A", "B"],
                data_url='https://storage.googleapis.com/ai2-jack-public/caption_contest_data_public/huggingface_hub/v1.0/ranking_{}.zip'.format(split_idx),
                url='www.capcon.dev',
                citation=_CAPTION_CONTEST_TASKS_CITATION,
            ),

            CaptionContestConfig(
                name='ranking_from_pixels_{}'.format(split_idx),
                description=_RANKING_DESCRIPTION,
                features=[
                    'image',
                    'contest_number',
                    'caption_choices',
                    'winner_source',
                ],
                label_classes=["A", "B"],
                data_url='https://storage.googleapis.com/ai2-jack-public/caption_contest_data_public/huggingface_hub/v1.0/ranking_from_pixels_{}.zip'.format(split_idx),
                url='www.capcon.dev',
                citation=_CAPTION_CONTEST_TASKS_CITATION,
            ),


            CaptionContestConfig(
                name='explanation_{}'.format(split_idx),
                description=_EXPLANATION_DESCRIPTION,
                features=[
                    'image',
                    'contest_number',
                    'image_location',
                    'image_description',
                    'image_uncanny_description',
                    'entities',
                    'questions',
                    'caption_choices',
                    'from_description',
                ],
                label_classes=None,
                data_url='https://storage.googleapis.com/ai2-jack-public/caption_contest_data_public/huggingface_hub/v1.0/explanation_{}.zip'.format(split_idx),
                url='www.capcon.dev',
                citation=_CAPTION_CONTEST_TASKS_CITATION,
            ),

            CaptionContestConfig(
                name='explanation_from_pixels_{}'.format(split_idx),
                description=_EXPLANATION_DESCRIPTION,
                features=[
                    'image',
                    'contest_number',
                    'caption_choices',
                ],
                label_classes=None,
                data_url='https://storage.googleapis.com/ai2-jack-public/caption_contest_data_public/huggingface_hub/v1.0/explanation_from_pixels_{}.zip'.format(split_idx),
                url='www.capcon.dev',
                citation=_CAPTION_CONTEST_TASKS_CITATION,
            ),
        ]
        cross_val_configs.extend(cur_split_configs)
    return cross_val_configs


class CaptionContestConfig(datasets.BuilderConfig):
    """BuilderConfig for Caption Contest."""

    def __init__(self, features, data_url, citation, url, label_classes=None, **kwargs):
        """BuilderConfig for Caption Contest.
        Args:
          features: `list[string]`, list of the features that will appear in the
            feature dict. Should not include "label".
          data_url: `string`, url to download the zip file from.
          citation: `string`, citation for the data set.
          url: `string`, url for information about the data set.
          label_classes: `list[string]`, the list of classes for the label if the
            label is present as a string. If not provided, there is no fixed label set.
          **kwargs: keyword arguments forwarded to super.
        """

        super(CaptionContestConfig, self).__init__(version=datasets.Version("1.0.0"), **kwargs)
        self.features = features
        self.data_url = data_url
        self.citation = citation
        self.url = url
        self.label_classes = label_classes


class CaptionContest(datasets.GeneratorBasedBuilder):
    """The CaptionContest benchmark."""

    BUILDER_CONFIGS = [
        CaptionContestConfig(
            name='matching',
            description=_MATCHING_DESCRIPTION,
            features=[
                'image',
                'contest_number',
                'image_location',
                'image_description',
                'image_uncanny_description',
                'entities',
                'questions',
                'caption_choices',
                'from_description',
            ],
            label_classes=["A", "B", "C", "D", "E"],
            data_url='https://storage.googleapis.com/ai2-jack-public/caption_contest_data_public/huggingface_hub/v1.0/matching.zip',
            url='www.capcon.dev',
            citation=_CAPTION_CONTEST_TASKS_CITATION,
        ),

        CaptionContestConfig(
            name='matching_from_pixels',
            description=_MATCHING_DESCRIPTION,
            features=[
                'image',
                'contest_number',
                'caption_choices',
            ],
            label_classes=["A", "B", "C", "D", "E"],
            data_url='https://storage.googleapis.com/ai2-jack-public/caption_contest_data_public/huggingface_hub/v1.0/matching_from_pixels.zip',
            url='www.capcon.dev',
            citation=_CAPTION_CONTEST_TASKS_CITATION,
        ),

        CaptionContestConfig(
            name='ranking',
            description=_RANKING_DESCRIPTION,
            features=[
                'image',
                'contest_number',
                'image_location',
                'image_description',
                'image_uncanny_description',
                'entities',
                'questions',
                'caption_choices',
                'from_description',
                'winner_source',
            ],
            label_classes=["A", "B"],
            data_url='https://storage.googleapis.com/ai2-jack-public/caption_contest_data_public/huggingface_hub/v1.0/ranking.zip',
            url='www.capcon.dev',
            citation=_CAPTION_CONTEST_TASKS_CITATION,
        ),

        CaptionContestConfig(
            name='ranking_from_pixels',
            description=_RANKING_DESCRIPTION,
            features=[
                'image',
                'contest_number',
                'caption_choices',
                'winner_source',
            ],
            label_classes=["A", "B"],
            data_url='https://storage.googleapis.com/ai2-jack-public/caption_contest_data_public/huggingface_hub/v1.0/ranking_from_pixels.zip',
            url='www.capcon.dev',
            citation=_CAPTION_CONTEST_TASKS_CITATION,
        ),


        CaptionContestConfig(
            name='explanation',
            description=_EXPLANATION_DESCRIPTION,
            features=[
                'image',
                'contest_number',
                'image_location',
                'image_description',
                'image_uncanny_description',
                'entities',
                'questions',
                'caption_choices',
                'from_description',
            ],
            label_classes=None,
            data_url='https://storage.googleapis.com/ai2-jack-public/caption_contest_data_public/huggingface_hub/v1.0/explanation.zip',
            url='www.capcon.dev',
            citation=_CAPTION_CONTEST_TASKS_CITATION,
        ),

        CaptionContestConfig(
            name='explanation_from_pixels',
            description=_EXPLANATION_DESCRIPTION,
            features=[
                'image',
                'contest_number',
                'caption_choices',
            ],
            label_classes=None,
            data_url='https://storage.googleapis.com/ai2-jack-public/caption_contest_data_public/huggingface_hub/v1.0/explanation_from_pixels.zip',
            url='www.capcon.dev',
            citation=_CAPTION_CONTEST_TASKS_CITATION,
        ),
    ] + _get_configs_crossvals()
    

    def _info(self):
        features = {feature: datasets.Value("string") for feature in self.config.features}
        # things are strings except for contest_number, entities, questions, and caption choices (if not explanation)
        features['contest_number'] = datasets.Value("int32")
        if 'explanation' not in self.config.name:
            features['caption_choices'] = datasets.features.Sequence(datasets.Value("string"))

        if 'entities' in features:
            features['entities'] = datasets.features.Sequence(datasets.Value("string"))

        if 'questions' in features:
            features['questions'] = datasets.features.Sequence(datasets.Value("string"))

        if 'image' in features:
            features['image'] = datasets.Image()
            
        features['label'] = datasets.Value("string")
        features['n_tokens_label'] = datasets.Value("int32")
        features['instance_id'] = datasets.Value("string")
        
        return datasets.DatasetInfo(
            description=_CAPTION_CONTEST_DESCRIPTION + self.config.description,
            features=datasets.Features(features),
            homepage=self.config.url,
            citation=self.config.citation
        )

    def _split_generators(self, dl_manager):
        dl_dir = dl_manager.download_and_extract(self.config.data_url) or ""
        self.images_dir = dl_manager.download_and_extract(_IMAGES_URL)
        task_name = _get_task_name_from_data_url(self.config.data_url)
        dl_dir = os.path.join(dl_dir, task_name)
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_file": os.path.join(dl_dir, "train.jsonl"),
                    "split": datasets.Split.TRAIN,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_file": os.path.join(dl_dir, "val.jsonl"),
                    "split": datasets.Split.VALIDATION,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_file": os.path.join(dl_dir, "test.jsonl"),
                    "split": datasets.Split.TEST,
                },
            ),
        ]

    def _generate_examples(self, data_file, split):
        with open(data_file, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                with open(self.images_dir + "/all_contest_images/{}.jpeg".format(row['contest_number']), "rb") as image:
                    row['image'] = {"path": self.images_dir + "/all_contest_images/{}.jpeg".format(row['contest_number']),
                                    "bytes": image.read()}
                yield row['instance_id'], row

def _get_task_name_from_data_url(data_url):
    return data_url.split("/")[-1].split(".")[0]
