from urllib.parse import ParseResultBytes
import numpy as np
import torch
from args import SNNArgs
import model
import random
import torch.nn as nn
from snntorch import utils
from snntorch import spikegen
from textattack.attacker import Attacker
from textattack.datasets import HuggingFaceDataset
from textattack.constraints.pre_transformation import InputColumnModification
from textattack.transformations import WordSwapEmbedding
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.goal_functions import UntargetedClassification
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.attack_recipes import (PWWSRen2019,
                                       DeepWordBugGao2018,
                                       PSOZang2020,
                                       TextBuggerLi2018,
                                       BERTAttackLi2020,
                                       TextFoolerJin2019,
                                       HotFlipEbrahimi2017)

def build_attacker(args:SNNArgs, model:nn.Module):
    if args.attack_method == 'textfooler':
        attack = TextFoolerJin2019.build(model)
    elif args.attack_method == 'bae':
        attack = BERTAttackLi2020.build(model)
    elif args.attack_method == 'textbugger':
        attack = TextBuggerLi2018.build(model)
    elif args.attack_method == 'pso':
        attack = PSOZang2020.build(model)
    elif args.attack_method == 'pwws':
        attack = PWWSRen2019.build(model)
    elif args.attack_method == 'deepwordbug':
        attack = DeepWordBugGao2018.build(model)
    elif args.attack_method == 'hotflip':
        attack = HotFlipEbrahimi2017.build(model)
    # for pre_constraints in attacker.pre_transformation_constraints:
    #     if isinstance(pre_constraints, InputColumnModification):
    #         attacker.pre_transformation_constraints.remove(pre_constraints)
    # attacker.pre_transformation_constraints.append(
    #     InputColumnModification(["premise", "hypothesis"], {args.nli_not_modify})
    # )
    if args.attack_method in ['textfooler', 'pwws', 'textbugger', 'pso']:
        attack.transformation = WordSwapEmbedding(max_candidates=args.neighbour_vocab_size)
        for constraint in attack.constraints:
            if isinstance(constraint, WordEmbeddingDistance):
                attack.constraints.remove(constraint)
    
    attack.constraints.append(MaxWordsPerturbed(max_percent=args.modify_ratio))
    use_constraint = UniversalSentenceEncoder(
        threshold=args.sentence_similarity,
        metric="angular",
        compare_against_original=False,
        window_size=15,
        skip_text_shorter_than_window=True,
    )
    attack.constraints.append(use_constraint)

    input_column_modification = InputColumnModification(
        ["premise", "hypothesis"], {"premise"}
    )
    attack.pre_transformation_constraints.append(input_column_modification)

    # attack.goal_function = UntargetedClassification(model, query_budget=args.query_budget_size)
    return attack


"""
Some Attack Functions that SNN differ from pytorch base model
"""

def batch_model_predict(model_predict, inputs):
    """Runs prediction on iterable ``inputs`` using batch size ``batch_size``.

    Aggregates all predictions into an ``np.ndarray``.
    """
    outputs = []
    batch = inputs
    batch_preds = model_predict(batch)

    # Some seq-to-seq models will return a single string as a prediction
    # for a single-string list. Wrap these in a list.
    if isinstance(batch_preds, str):
        batch_preds = [batch_preds]

    # Get PyTorch tensors off of other devices.
    if isinstance(batch_preds, torch.Tensor):
        batch_preds = batch_preds.cpu()

    # Cast all predictions iterables to ``np.ndarray`` types.
    if not isinstance(batch_preds, np.ndarray):
        batch_preds = np.array(batch_preds)
    outputs.append(batch_preds)

    return np.concatenate(outputs, axis=0)


def batch_snn_model_predict(model_predict, inputs, num_step=32, spike_gen=spikegen.rate):
    utils.reset(net=model_predict)
    num_return = utils._final_layer_check(model_predict)
    outputs = []
    
    mem_rec_trunc = []
    spk_rec_trunc = []
    batch = spike_gen(inputs, num_steps=num_step)

    ### need to debug the num_step
    ### batch = batch.transpose(0, 1)
    num_step = batch.shape[0]
    for step in range(num_step):
        if num_return == 2:
            _, spk, mem = model_predict(batch[step])
        elif num_return == 3:
            spk, _, mem = model_predict(batch[step])
        elif num_return == 4:
            spk, _, _, mem = model_predict(batch[step])
        spk_rec_trunc.append(spk)
        mem_rec_trunc.append(mem)
    spk_rec_trunc = torch.stack(spk_rec_trunc, dim=0)
    logits = torch.mean(spk_rec_trunc, dim=0)
    preds = logits.cpu()
    if not isinstance(preds, np.ndarray):
        preds = np.array(preds)
    outputs.append(preds)
    
    return np.concatenate(outputs, axis=0)

def batch_snn_ensemble_model_predict(model_predict, inputs, label_num, num_step=32, spike_gen=spikegen.rate):
    utils.reset(net=model_predict)
    num_return = utils._final_layer_check(model_predict)
    outputs = []
    
    spk_rec_trunc = []
    batch = spike_gen(inputs, num_steps=num_step)

    ### need to debug the num_step
    ### batch = batch.transpose(0, 1)
    num_step = batch.shape[0]
    for step in range(num_step):
        if num_return == 2:
            _, spk, mem = model_predict(batch[step])
        elif num_return == 3:
            spk, _, mem = model_predict(batch[step])
        elif num_return == 4:
            spk, _, _, mem = model_predict(batch[step])
        temp = []
        for i in range(spk.shape[-1] // label_num * 2):
            temp.append(torch.mean(spk[:, i*label_num//2:(i+1)*label_num//2], dim=-1))
        spk = torch.stack(temp).transpose(1,0)
        spk_rec_trunc.append(spk)


    spk_rec_trunc = torch.stack(spk_rec_trunc, dim=0)
    logits = torch.mean(spk_rec_trunc, dim=0)
    preds = logits.cpu()
    if not isinstance(preds, np.ndarray):
        preds = np.array(preds)
    outputs.append(preds)
    
    return np.concatenate(outputs, axis=0)

class CustomTextAttackDataset(HuggingFaceDataset):
    """Loads a dataset from HuggingFace ``datasets`` and prepares it as a
    TextAttack dataset.

    - name: the dataset name
    - subset: the subset of the main dataset. Dataset will be loaded as ``datasets.load_dataset(name, subset)``.
    - label_map: Mapping if output labels should be re-mapped. Useful
      if model was trained with a different label arrangement than
      provided in the ``datasets`` version of the dataset.
    - output_scale_factor (float): Factor to divide ground-truth outputs by.
        Generally, TextAttack goal functions require model outputs
        between 0 and 1. Some datasets test the model's correlation
        with ground-truth output, instead of its accuracy, so these
        outputs may be scaled arbitrarily.
    - shuffle (bool): Whether to shuffle the dataset on load.
    """
    def __init__(
            self,
            name,
            instances,
            label_map=None,
            output_scale_factor=None,
            dataset_columns=None,
            shuffle=False,
            ):
        assert instances is not None and len(instances) != 0
        self._name = name
        self._i = 0
        self.label_map = label_map
        self.output_scale_factor = output_scale_factor
        if len(instances[0]) == 2:
            self.input_columns, self.output_column = ("text",), "label"
            self.examples = [{"text": instance[0], "label": instances[1]} for instance in instances]
            self.label_map = {"negetive": 0, "positive":1}
        elif len(instances[0]) == 3:
            self.input_columns, self.output_column = ("premise", "hypothesis"), "label"
            self.examples = [{"premise": instance[0], "hypothesis": instance[1], "label": int(instance[2])}
                            for
                            instance in instances]
            self.label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
        
        self.shuffled = shuffle
        if shuffle:
            random.shuffle()
        self._dataset = self.examples
            
        
    @classmethod
    def from_instances(cls, name: str, instances,
                       labels=None) -> "CustomTextAttackDataset":
        return cls(name, instances, labels)

class Instance:
    def __init__(self, test_a, label, text_b) -> None:
        self.text_a
        pass