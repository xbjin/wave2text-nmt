from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from contextlib import contextmanager


@contextmanager
def open_files(names, mode='r'):
  """ Safely open a list of files in a context manager.
  Example:
  >>> with open_files(['foo.txt', 'bar.csv']) as f:
  ...   pass
  """

  files = []
  try:
    for name_ in names:
      files.append(open(name_, mode=mode))
    yield files
  finally:
    for file_ in files:
      file_.close()
