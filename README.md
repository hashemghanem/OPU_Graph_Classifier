# OPU_Graph_Classifier
## Fast graph classifier with optical random features

The package we built in this work is available in the directory: <./code/RF_modules>


To run one of the example codes in <./code/examples> directory (let's say example_name.py), place yourself in <./code> directory and run the example from there, i.e. run the following command from the <./code> directory:

python -m examples.example_name

In all examples, we use one of the techniques (OPU RFs, Gs RFs, GS+EIG RFs) to represent graphs, then a linear SVM model is used to learn how to classify them.

**Note:** In order to execute the codes, you must have access to [LightOn](https://docs.lighton.ai/) servers. Pleaser refer to the this guiding [tutorial](https://community.lighton.ai/t/how-to-use-lighton-cloud-general-guide/20) to help you do that correctly.


## Available examples:
### 1. OPU_SBM_varying_r.py 
In this example, we classify graphs in the SBM_based dataset. We vary the value of the Inter-class similarity parameter r, generate the corresponding graph datset, and learn how to classify graphs. We choose the follwing:
* S_k : uniform sampling
*  s = 2000 , m = 5000 , k = 6
*  OPU RFs
### 2. Gs_DD_varying_m.py
In this example, we classify graphs in the D&D dataset. We vary the RFs number m and for each value we learn how to classify the D&D graphs. It is expected to see that when m grows, the test accuracy improves too.  We choose the follwing:
* S_k : Induced random walk
*  s = 4000 , m = 5000 , k = 7
*  Gassian RFs
