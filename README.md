# METR Timelines Model

A probabilistic modeling tool for forecasting AGI timeline trajectories based on the paper by METR entitled ["Measuring AI Ability to Complete Long Tasks"](https://arxiv.org/abs/2503.14499).

This probabilistic approach accounts for uncertainties in starting capabilities, required end capabilities, and the rate of progress, providing a distribution of possible timelines rather than a single point estimate.


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/peterhurford/agi_timelines.git
   cd agi_timelines
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run Juptyer notebook


## Notes

* The model is based significantly on [squigglepy](http://github.com/rethinkpriorities/squigglepy), a Python library for Monte Carlo simulation.
