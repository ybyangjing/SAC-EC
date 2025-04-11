# SAC-DC

This is the official PyTorch code for the paper:

**Green Optimization for Micro Data Centers: Task Scheduling for a Combined Energy Consump tion Strategy.**

Yuanyuan Hu,Jing Yang, Xiaoli Ruan, Yulin Chen,Chengjiang Li, Zhaohu Zhang, Wei Zhang

**This is the overall methodology diagram of the paper:**
![系统图](https://github.com/user-attachments/assets/878d6064-2fd2-4b08-86d8-2aa9167742de)

Fig. 1. Overall framework diagram of SAC-EC.In this section, the system models are presented, in cluding the task load model, the MDC environment model, the combined energy consumption model, and the dynamic pricing strategy. In Figure 1, the orange block is the task load model, which is used for task allocation. The green block is the MDC environment model, which describes the environmental resources. The blue block is the joint energy consumption model, which represents the overall energy consumption of the MDCs.




## Prerequisites

The following packages are required to run the scripts:

- [torch](https://github.com/pytorch/pytorch)
- [torchvision](https://github.com/pytorch/vision)
- [numpy](https://github.com/numpy/numpy)

## Training scripts

run: ``` PPO_discrete_main_2.py ```



