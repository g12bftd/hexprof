"""Shrunk ("dummy") Gemma 4 / SigLIP 2 variants with random weights, deployed and profiled on AI Hub.

The models are built by Qualcomm's own NPU-adapted code (``qai_hub_models``) from a Hugging Face
config with the requested dimensions, so the graphs have the same operator structure as the
published deployments; only the sizes change. See ``hexprof dummy --help`` and README.
"""
