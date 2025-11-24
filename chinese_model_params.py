import numpy as np
import squigglepy as sq

from datetime import datetime

from model_params import inference_compute_adj, reliability_needed, reliability_count_to_penalty, task_type_penalty, messy_tasks_penalty, experience_penalty

# -----------
# GET INITIAL
# -----------

# START TASK LENGTH: How many max minutes of all AGI-relevant tasks can AI reliably do to a sufficient degree of reliability?
# define current best
current_best = 54 / 60
current_best_date = datetime(2025, 11, 4, 0, 0)


# ------------------
# DEFINE ADJUSTMENTS
# ------------------

# 1. Elicitiation boost - Can you get a boost to scores by iterating on scaffolding and other elicitation techniques? How much should we multiply up to adjust for this?
elicitation_boost = sq.mixture([
        [0.1, 1.0],  # 10% chance no, 50% chance you can get a 1.2x speed up, 40% chance of 1.5x.
        [0.5, 1.2],
        [0.4, 1.5],
    ])


# ---------------------
# CREATE DISTRIBUTION #
# ---------------------

# Start with current best, add elicitation boost
start_task_length = current_best * elicitation_boost

# add inference compute adjustment
start_task_length = start_task_length * inference_compute_adj

# add reliability penalty
start_task_length = start_task_length * sq.dist_fn(
    reliability_needed, reliability_count_to_penalty
)

# Add task type penalty
start_task_length *= task_type_penalty

# Add messy tasks penalty
start_task_length *= messy_tasks_penalty

# Add experience penalty
start_task_length *= experience_penalty

# Add a minimum value of 1sec
start_task_length = sq.dist_max(1 / 60 / 60, start_task_length)

# -----------
# SHIFT PARAMETER: How much earlier (in days) are capabilities developed internally versus made available to the public?
shift = sq.norm(0, 30 * 4, credibility=90, lclip=0)
