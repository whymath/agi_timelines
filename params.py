import numpy as np
import squigglepy as sq

from pprint import pprint


print('## START task length (displayed in min) ##')

# -- DEFINE CURRENT BEST
current_best = 1.75 # o3 task length at 50% reliability?

# -- DEFINE ADJUSTMENTS
elicitation_boost = sq.mixture([[0.3, 1], # Can you get a boost to scores by iterating on scaffolding and other elicitation techniques? 30% chance no, 40% chance you can get a 1.2x speed up, 30% chance of 1.5x.
                                [0.4, 1.2],
                                [0.3, 1.5]])

inference_compute_adj = sq.lognorm(lognorm_mean=2, lognorm_sd=1, lclip=1) # Can you get a boost to scores by increasing inference compute to human level? Approx doubling
                               
reliability_needed = sq.mixture([[0.2, 0.5], # What amount of reliability will we need? Probability distribution over hypotheses
                                 [0.4, 0.8],
                                 [0.2, 0.9],
                                 [0.1, 0.95],
                                 [0.1, 0.99]])

def reliability_count_to_penalty(reliability):
    r = np.asarray(reliability, dtype=float)
    reliability = np.array([0.50, 0.80, 0.90, 0.95, 0.99])
    penalty = np.array([1.0, 0.25, 0.25**2, 0.25**3, 0.25**4])
    matches = r[..., None] == reliability
    hit_any = matches.any(axis=-1)
    idx = matches.argmax(axis=-1)
    out = np.full_like(r, np.nan, dtype=float)
    out[hit_any] = penalty[idx[hit_any]]
    return out

task_type_penalty = sq.mixture([[0.1, 1],                         # 10% chance that METR's software tasks are sufficient for AGI
                                [0.3, 1 / sq.lognorm(5, 20)],     # 30% chance that true AGI tasks are 5-20x harder than METR's software tasks
                                [0.6, 1 / sq.lognorm(10, 1000)]]) # 60% chance that true AGI tasks are 10-1000x harder than METR's software tasks

# -- CREATE DISTRIBUTION
# Start with current best, add elicitation boost
start_task_length = current_best * elicitation_boost

# add inference compute adjustment
start_task_length = start_task_length * inference_compute_adj

# add reliability penalty
start_task_length = start_task_length * sq.dist_fn(reliability_needed, reliability_count_to_penalty)

# Add task type penalty
start_task_length *= task_type_penalty

# Add a minimum value of 1sec
start_task_length = sq.dist_max(1/60/60, start_task_length)

# Show samples in minutes (naturally in hours)
pprint(sq.get_percentiles((start_task_length * 60) @ 100_000, digits=2))

print('\n\n')
print('## AGI task length (displayed in hrs) ##')
agi_task_length = sq.lognorm(80, 2000, credibility=80, lclip=40)
pprint(sq.get_percentiles(agi_task_length @ 100_000, digits=0))

print('\n\n')
print('## DOUBLING TIME (displayed in days) ##')
doubling_time = sq.mixture([[0.3, 212],
                            [0.1, 118],
                            [0.6, sq.lognorm(lognorm_mean=185.25, lognorm_sd=40)]])
pprint(sq.get_percentiles(doubling_time @ 100_000, digits=0))

print('\n\n')
print('## ACCELERATION (displayed in days)')
acceleration = sq.mixture([[0.1, 1 + sq.lognorm(0.005, 0.1, credibility=80)],
                           [0.8, 1],
                           [0.1, 1 - sq.lognorm(0.005, 0.1, credibility=80)]])
pprint(sq.get_percentiles(acceleration @ 100_000, digits=3))

print('\n\n')
print('## SHIFT (displayed in days) ##')
shift = sq.norm(30, 30*5, credibility=80, lclip=0)
pprint(sq.get_percentiles(shift @ 100_000, digits=0))
