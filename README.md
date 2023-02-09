# BYTEEYE
BYTEEYE is a bytecode-level smart contract vulnerability detection framework.
# Environmental Dependency
To successfully set up BYTEEYE, we need to install the necessary environmental dependencies.  

  1 support by GPU  
    A GPU, the memory of which is over 21G. We use RTX 3090 in experiments.   

  2 pytorch  
    A stable version that is suitable to the local environment. Users can use pip or anaconda instruction to install pytorch. Our install version is 1.12.0+cu113.  
# Tool Execution
Running the command belows can execute BYTEEYE, including cfg construction, feature extraction, and vulnerability detection.  
  ```python main.py```  
Running the following command can only build cfg.  
  ```python.main.py --cfg 1```  
Running the following command can generate features, train GNN models and detect vulnerabilities.  
  ```python main.py --training 1```  
  
# Illustraction
We make our datasets and source code available private upon submission and will be made public upon acceptance.

Now, we release the core code of BYTEEYE and datasets to help understand the paper. And we will further release a more completely version upon acceptance.
