from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import math
import numpy as np
from translate import utils
from translate.translation_model import TranslationModel, BaseTranslationModel


class MultiTaskModel(BaseTranslationModel):
  def __init__(self, name, tasks, checkpoint_dir, keep_best=1, main_task=None, **kwargs):
    """
    Proxy for several translation models that are trained jointly
    This class presents the same interface as TranslationModel
    """
    self.models = []
    self.ratios = []

    for task in tasks:
      self.checkpoint_dir = checkpoint_dir
      # merging both dictionaries (task parameters have a higher precedence)
      kwargs_ = dict(task)
      for k, v in kwargs.iteritems():
        kwargs_.setdefault(k, v)

      model = TranslationModel(checkpoint_dir=None, keep_best=keep_best, **kwargs_)

      self.models.append(model)
      self.ratios.append(task.ratio if task.ratio is not None else 1)

    self.ratios = [ratio / sum(self.ratios) for ratio in self.ratios]   # unit normalization

    self.main_task = main_task
    self.global_step = 0  # steps of all tasks combined
    super(MultiTaskModel, self).__init__(name, checkpoint_dir, keep_best)

  def train(self, sess, beam_size, steps_per_checkpoint, steps_per_eval=None, scoring_script=None,
            max_train_size=None, max_dev_size=None, eval_output=None, remove_unk=False, max_steps=0, **kwargs):
    utils.log('reading training and development data')

    self.global_step = 0
    for model in self.models:
      model.read_data(max_train_size, max_dev_size)
      # those parameters are used to track the progress of each task
      model.loss, model.time, model.steps = 0, 0, 0
      model.previous_losses = []
      self.global_step += model.global_step.eval(sess)

    utils.log('starting training')
    while True:
      i = np.random.choice(len(self.models), 1, p=self.ratios)[0]
      model = self.models[i]

      start_time = time.time()
      model.loss += model.train_step(sess)
      model.time += (time.time() - start_time)
      model.steps += 1
      self.global_step += 1

      if steps_per_checkpoint and self.global_step % steps_per_checkpoint == 0:
        for model_ in self.models:
          if model_.steps == 0:
            continue

          loss_ = model_.loss / model_.steps
          step_time_ = model_.time / model_.steps
          perplexity = math.exp(loss_) if loss_ < 300 else float('inf')

          utils.log('{} step {} learning rate {:.4f} step-time {:.2f} perplexity {:.2f}'.format(
                    model_.name, model_.global_step.eval(sess), model_.learning_rate.eval(),
                    step_time_, perplexity))

          if len(model_.previous_losses) > 2 and loss_ > max(model_.previous_losses[-3:]):
            sess.run(model_.learning_rate_decay_op)

          model_.previous_losses.append(loss_)
          model_.loss, model_.time, model_.steps = 0, 0, 0
          model_.eval_step(sess)

        self.save(sess)

      if steps_per_eval and scoring_script and self.global_step % steps_per_eval == 0:
        score = 0

        for ratio, model_ in zip(self.ratios, self.models):
          output = None if eval_output is None else '{}.{}.{}'.format(eval_output, model_.name,
                                                                      model_.global_step.eval(sess))
          scores_ = model_.evaluate(sess, beam_size, scoring_script, on_dev=True, output=output,
                                    remove_unk=remove_unk)
          score_ = scores_[0]

          # if there is a main task, pick best checkpoint according to its score
          # otherwise use the average score across tasks
          if self.main_task is None:
            score += ratio * score_
          elif model_.name == self.main_task:
            score = score_

        self.manage_best_checkpoints(self.global_step, score)

      if 0 < max_steps < self.global_step:
        utils.log('finished training')
        return

  def decode(self, *args, **kwargs):
    if self.main_task is not None:
      model = next(model for model in self.models if model.name == self.main_task)
    else:
      model = self.models[0]
    return model.decode(*args, **kwargs)

  def evaluate(self, *args, **kwargs):
    if self.main_task is not None:
      model = next(model for model in self.models if model.name == self.main_task)
    else:
      model = self.models[0]
    return model.evaluate(*args, **kwargs)

  def align(self, *args, **kwargs):
    if self.main_task is not None:
      model = next(model for model in self.models if model.name == self.main_task)
    else:
      model = self.models[0]
    return model.align(*args, **kwargs)
