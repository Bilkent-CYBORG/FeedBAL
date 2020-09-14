# FeedBAL
*Updated  (major revision 30/08/20)* implementation of the FeedBack Adaptive Learning (FeedBAL) algorithm for the episodic multi-armed bandit (eMAB) setting.

# How to run

 1. Install requirements in [requirements.txt](requirements.txt).
 2. To run simulation I (no dropouts) and simulation II (dropouts), set DROPOUT_PROB to 0 and 0.1, respectively, in the [user_arrival_simulator.py](user_arrival_simulator.py) file.
 3. Run the [main.py](main.py) script.

# Specs of the PC used for the paper
A PC with the following specs was used for the simulations whose results/figures are presented in the paper:

CPU: Intel Core i7-7700 @ 3.60 GHz |
RAM: 32 GB |
OS: Ubuntu 18.10
