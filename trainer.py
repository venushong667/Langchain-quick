
import os
from typing import Dict, List, Optional, Union
from torch.utils.data import DataLoader
from trl import SFTTrainer
from peft import PeftModel
from transformers import PreTrainedModel, AutoTokenizer
import torch
import torch.nn as nn


class RAGTrainer(SFTTrainer):

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module, str] = None, resume_from_checkpoint: Optional[str] = None,
        **kwargs
    ):
        self.resume_from_checkpoint = resume_from_checkpoint
        if resume_from_checkpoint:
            self.resume_from_checkpoint, model = self._validate_peft_checkpoint(model, resume_from_checkpoint)
        
        tokenizer = AutoTokenizer.from_pretrained(
            model.config._name_or_path,
            add_eos_token=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'

        super().__init__(
            model,
            preprocess_logits_for_metrics=self.preprocess_logits_for_metrics,
            tokenizer=tokenizer,
            **kwargs
        )

    def _validate_peft_checkpoint(self, model, resume_from_checkpoint):
        peft_checkpoint_name = os.path.join(
            resume_from_checkpoint, "adapter_model.bin"
        )
        if os.path.exists(peft_checkpoint_name):
            print(f"Restarting from LoRA Adapter {peft_checkpoint_name}")
            model = PeftModel.from_pretrained(model, resume_from_checkpoint)
            checkpoint_name = None
        else:
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "pytorch_model.bin"
            )  # Full checkpoint

        return checkpoint_name, model
    
    def train(self):
        super().train(resume_from_checkpoint=self.resume_from_checkpoint)
    
    def preprocess_logits_for_metrics(self, logits, labels):
        if type(logits) is tuple:
            logits = logits[0]
        pred_ids = torch.argmax(logits, dim=-1)

        return pred_ids, labels
    
    # def compute_metrics(self, pred_output: EvalPrediction):
    #     metrics, eval_gt_list, eval_pred_list = evaluate_output(
    #         pred_output.label_ids, pred_output.predictions[0]
    #     )

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Union[bool, None] = None,
        ignore_keys: Union[List[str], None] = None,
        metric_key_prefix: str = "eval"
    ):
        prediction_loss_only = False
        output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix
        )
        
        self._write_result(
            output.metrics,
            dataloader,
            output.label_ids,
            output.predictions[0],
            os.path.join(self.args.output_dir, f"{metric_key_prefix}_result.txt"),
        )

        return output
    
    def _write_result(
        self,
        metrics: Dict[str, float],
        dataloader: DataLoader,
        label_ids: List[float],
        predictions: List[float],
        output_path: str,
    ):
        with open(output_path, "w") as f:
            label_ids[label_ids == -100] = self.tokenizer.pad_token_id
            predictions[predictions == -100] = self.tokenizer.pad_token_id
            try:
                for feature, label, pred in zip(dataloader.dataset, label_ids, predictions):
                    # inputs = self.tokenizer.decode(feature["input_ids"], skip_special_tokens=True)
                    output = self.tokenizer.decode(label, skip_special_tokens=True)
                    pred_output = self.tokenizer.decode(pred, skip_special_tokens=True)
                    # f.write("Input: {}\n".format(inputs))
                    f.write("Gt: \n{}\n".format(output))
                    f.write("*********\n")
                    f.write("Pred: \n{}\n".format(pred_output))
                    f.write("--------------------------------------------------------------------------------\n\n")
            except:
                import ipdb; ipdb.set_trace()
