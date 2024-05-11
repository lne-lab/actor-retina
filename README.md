# Actor-retina

Code used for data analysis in the article "An actor-model framework for visual sensory encoding"

# Description

The source code for the article "An actor-model framework for visual sensory encoding" by Leong et. al. is provided. To train the forward model, run `train_fwd.py`. To train the actor network, run `train_actor.py`. The optimized model is provided to generate the figures found in the article. To understand the data analysis pipeline and generate the figures presented in the article, look at `gen_fig_X.py`.

# Software-requirements
The pipeline is written fully in python. The requirements file for the libraries used is provided. To install the necessary libraries, run `pip install -r requirements.txt`

# Hardware-requirements
Our customed-scripts have been executed on an NVIDIA RTX 3090 GPU. Be advised that execution of the code without GPU-backed machines (CPU only) can considerably increases the training time.

