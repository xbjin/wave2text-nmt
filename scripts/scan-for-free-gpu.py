#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
import itertools
import subprocess
from collections import Counter
import time

servers = [('dvorak0', 2), ('ceos', 1), ('hyperion', 2)]

all_devices = [(server_name, gpu_id) for server_name, gpu_count in servers
               for gpu_id in range(gpu_count)]

free_devices = set()

for i in itertools.count():
    new_free_devices = set()

    for server_name, gpu_count in servers:
        output = subprocess.check_output(['ssh', server_name, 'nvidia-smi'])
        lines = output.split('\n')

        jobs = list(itertools.dropwhile(
          (lambda fields: fields != ['GPU', 'PID', 'Type', 'Process', 'name', 'Usage']),
          (line.strip('|').split() for line in lines)))[2:-2]

        try:
            job_counts = Counter(int(fields[0]) for fields in jobs)
        except:
            job_counts = Counter()

        for gpu_id in range(gpu_count):
            if job_counts[gpu_id] == 0:
                new_free_devices.add((server_name, gpu_id))

    if i == 0:
        changed_state = set(all_devices)
    else:
        changed_state = (free_devices | new_free_devices) - (free_devices &
                                                             new_free_devices)

    free_devices = new_free_devices

    messages = []
    for server_name, gpu_id in changed_state:
        messages.append('GPU {} on {} is now {}'.format(gpu_id, server_name,
            'free' if (server_name, gpu_id) in free_devices else 'used'))

    if messages:
        summary = ', '.join('{}/{}'.format(server_name, gpu_id)
            for server_name, gpu_id in free_devices)
        messages.append('Free devices: {}'.format(summary or 'none'))

    message = '\n'.join(messages)
    if message:
        print '[{}]:\n{}'.format(time.strftime("%H:%M:%S"), message)
        subprocess.call(['notify-send', 'Free GPUs', message])

    time.sleep(60)
