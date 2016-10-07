#!/usr/bin/python2

"""
Pre-processing file for BTEC spoken data (by Laurent & Margaux)
"""

import wave
import os
import shutil

txt_filename = 'data/raw/btec.fr-en/btec-Margaux.en'
audio_dir = 'data/raw/btec.fr-en/speech_fr/btec-Margaux'

output_filename = 'data/raw/btec.fr-en/btec-Margaux-fixed.en'
output_audio_dir = 'data/raw/btec.fr-en/speech_fr/btec-Margaux-fixed'

try:
  os.makedirs(output_audio_dir)
except:
  pass

with open(txt_filename) as txt_file, open(output_filename, 'w') as output_file:
  i = 1
  for line, audio_filename in zip(txt_file, os.listdir(audio_dir)):
    audio_filename = os.path.join(audio_dir, audio_filename)
    try:
      wave.open(audio_filename)
      shutil.copy(audio_filename, os.path.join(output_audio_dir, '{:03d}.wav'.format(i)))
      output_file.write(line)
      i += 1
    except:
      continue
