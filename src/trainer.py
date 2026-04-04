import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from .optimizer import SGLD

class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, temperature=2.0, alpha=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs_student = model(**inputs)
        student_logits = outputs_student.logits

        with torch.no_grad():
            outputs_teacher = self.teacher_model(**inputs)
            teacher_logits = outputs_teacher.logits

        loss_hard = outputs_student.loss
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_prob = F.log_softmax(student_logits / self.temperature, dim=-1)
        loss_soft = self.kl_loss(soft_prob, soft_targets) * (self.temperature ** 2)

        loss = (self.alpha * loss_hard) + ((1 - self.alpha) * loss_soft)
        return (loss, outputs_student) if return_outputs else loss

class SGLDDistillationTrainer(DistillationTrainer):
    def create_optimizer(self):
        total_train_examples = len(self.train_dataset)
        batch_size = self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps
        num_pseudo_batches = total_train_examples // batch_size

        self.optimizer = SGLD(
            self.model.parameters(),
            lr=self.args.learning_rate,
            num_pseudo_batches=num_pseudo_batches,
            num_burn_in_steps=2600
        )
        return self.optimizer