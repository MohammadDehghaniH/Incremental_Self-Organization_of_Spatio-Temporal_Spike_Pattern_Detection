# Incremental_Self-Organization_of_Spatio-Temporal_Spike_Pattern_Detection

This repository contains the codes to implement the model described in the manuscript "Incremental Self-Organization of Spatio-Temporal Spike Pattern Detection".

**Single post-synaptic neuron**

This repository contains the MATLAB implementation for the **single postsynaptic neuron** simulations described in the paper.

Core files:
Model_One_With_PreHSP.m: This script serves as the main driver for the simulation. It contains the core logic needed to run the model.

Subfunctions:
Model_Input.m and Model_Input_test.m: These subfunctions are responsible for calculating the primary input to the model for learning and testing.

**More than one post-synaptic neuron (Incremental Learning)**

This repository contains the MATLAB implementation for the **more then one postsynaptic neuron** simulations described in the paper.

Core files: Model_NET_Incremental.m: This script serves as the main driver for the simulation. It contains the core logic needed to run the model.

Subfunctions: Model_Input_NET_WITHOUT_EM.m, Model_Input_NET_WITH_EM_ORIGINAL.m, and Model_Input_NET_WITH_EM: These subfunctions are responsible for calculating the primary input to the model for learning and testing.


