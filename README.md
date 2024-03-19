Multiscale Spectral Similarity Index
======================

An MS-SSIM inspired metric for assessing imputation accuracy in spatial transcriptomics data. MSSI works by iterativly pooling the graph representing the spatial data and relies on the graph-coarsening package (https://github.com/loukasa/graph-coarsening) which can be installed through the commands:

    git clone git@github.com:loukasa/graph-coarsening.git
    cd graph-coarsening
    pip install .
    
And then mssi can be installed via pip:

    pip install mssi