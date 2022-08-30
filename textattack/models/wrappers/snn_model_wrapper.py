"""
PyTorch SNN Model Wrapper
--------------------------
"""


import torch
from torch.nn import CrossEntropyLoss
from args import SNNArgs
from data_preprocess.green_encoder import GreenEncoder

import textattack
import utils.attackutils
from snntorch import spikegen
from .model_wrapper import ModelWrapper

torch.cuda.empty_cache()


class SNNModelWrapper(ModelWrapper):
    """Loads a PyTorch model (`nn.Module`) and tokenizer.

    Args:
        model (torch.nn.Module): PyTorch model
        tokenizer: tokenizer whose output can be packed as a tensor and passed to the model.
            No type requirement, but most have `tokenizer` method that accepts list of strings.
    """

    def __init__(self, args:SNNArgs, model, tokenizer):
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                f"PyTorch model must be torch.nn.Module, got type {type(model)}"
            )

        self.model = model
        self.tokenizer = tokenizer
        self.num_steps = args.num_steps
        self.use_codebook = args.use_codebook
        if self.use_codebook == 'True':
            self.encoder = GreenEncoder(self.args)
            self.encoder.write_codebook()
            self.spike_gen = self.encoder.spike_gen
        else:
            self.spike_gen = spikegen.rate


    def to(self, device):
        self.model.to(device)

    def __call__(self, text_input_list):
        model_device = next(self.model.parameters()).device
        ids = self.tokenizer(text_input_list)
        ids = ids.clone().detach().to(model_device)

        with torch.no_grad():
            outputs = utils.attackutils.batch_snn_model_predict(
                self.model, ids, num_step=self.num_steps, spike_gen=self.spike_gen
            )

        return outputs

    def get_grad(self, text_input, loss_fn=CrossEntropyLoss()):
        """Get gradient of loss with respect to input tokens.

        Args:
            text_input (str): input string
            loss_fn (torch.nn.Module): loss function. Default is `torch.nn.CrossEntropyLoss`
        Returns:
            Dict of ids, tokens, and gradient as numpy array.
        """

        if not hasattr(self.model, "get_input_embeddings"):
            raise AttributeError(
                f"{type(self.model)} must have method `get_input_embeddings` that returns `torch.nn.Embedding` object that represents input embedding layer"
            )
        if not isinstance(loss_fn, torch.nn.Module):
            raise ValueError("Loss function must be of type `torch.nn.Module`.")

        raise NotImplementedError
        # self.model.train()

        # embedding_layer = self.model.get_input_embeddings()
        # original_state = embedding_layer.weight.requires_grad
        # embedding_layer.weight.requires_grad = True

        # emb_grads = []

        # def grad_hook(module, grad_in, grad_out):
        #     emb_grads.append(grad_out[0])

        # emb_hook = embedding_layer.register_backward_hook(grad_hook)

        # self.model.zero_grad()
        # model_device = next(self.model.parameters()).device
        # ids = self.tokenizer([text_input])
        # ids = torch.tensor(ids).to(model_device)

        # predictions = self.model(ids)

        # output = predictions.argmax(dim=1)
        # loss = loss_fn(predictions, output)
        # loss.backward()

        # # grad w.r.t to word embeddings

        # # Fix for Issue #601

        # # Check if gradient has shape [max_sequence,1,_] ( when model input in transpose of input sequence)

        # if emb_grads[0].shape[1] == 1:
        #     grad = torch.transpose(emb_grads[0], 0, 1)[0].cpu().numpy()
        # else:
        #     # gradient has shape [1,max_sequence,_]
        #     grad = emb_grads[0][0].cpu().numpy()

        # embedding_layer.weight.requires_grad = original_state
        # emb_hook.remove()
        # self.model.eval()

        # output = {"ids": ids[0].tolist(), "gradient": grad}

        # return output

    def _tokenize(self, inputs):
        """Helper method that for `tokenize`
        Args:
            inputs (list[str]): list of input strings
        Returns:
            tokens (list[list[str]]): List of list of tokens as strings
        """
        return [self.tokenizer.convert_ids_to_tokens(self.tokenizer.tokenize_sentence(x)) for x in inputs]
