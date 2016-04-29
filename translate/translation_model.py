import tensorflow as tf
import random


class TranslationModel(object):
  def __init__(self):
    self.model = self.create_model()
    self.partial_models = []
    self.models = [self.model]
    pass

  def create_model(self):
    return None

  def initialize(self, session=None):
    session.run(tf.initialize_all_variables())
    # load and/or initialize models
    pass

  def train(self, session=None):
    while True:
      model = random.choice(self.models)
      self.train_step(model)

  def train_step(self, model):
    pass

  def decode(self, sentence, session=None):
    pass

  def evaluate(self, session=None):
    pass
