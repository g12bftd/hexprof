"""Systematic deployment studies on AI Hub's hosted Galaxy S25.

A *study* sweeps one deployment setting (an *axis*) over a set of values given on the
command line; every value becomes a *variant* that is compiled and profiled on the
real device, then verified: hexprof reads back from AI Hub what actually ran (graph
names, input shapes, quantization dtypes, QAIRT version, device) and checks it against
the requested setting. Unverified variants are flagged and kept out of the comparison.

Targets and axes
    gemma4-text    context-length | prompt-length | decode-tokens   (Qualcomm's export)
    gemma4-vision  soft-tokens | precision | qairt | mixed-precision (Qualcomm's VEG class)
    siglip2-image  batch | precision | qairt | mixed-precision      (Qualcomm's SigLIP 2)
    siglip2-text   batch | precision | qairt | mixed-precision
    dummy-*        any knob of a shrunk random-weight model (layers, hidden, heads, ...; see
                   ``hexprof dummy knobs``), built by Qualcomm's own code (hexprof.dummy)
"""
