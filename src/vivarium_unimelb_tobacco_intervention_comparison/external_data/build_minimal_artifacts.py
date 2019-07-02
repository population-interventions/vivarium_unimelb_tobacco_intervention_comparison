#!/usr/bin/env python

import logging

from .artifact import assemble_tobacco_artifacts


logging.basicConfig(level=logging.INFO)
num_draws = 0
assemble_tobacco_artifacts(num_draws)
