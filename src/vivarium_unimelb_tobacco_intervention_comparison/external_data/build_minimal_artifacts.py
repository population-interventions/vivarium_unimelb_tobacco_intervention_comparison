#!/usr/bin/env python

import logging

import mslt_port.artifact


logging.basicConfig(level=logging.INFO)
num_draws = 0
mslt_port.artifact.assemble_tobacco_artifacts(num_draws)
