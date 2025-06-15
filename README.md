# ISSM_GNN
Deep learning emulators for ISSM (Ice-sheet and Sea-level System Model) ice sheet model using graph neural networks (GNNs)

## Experiments for Pine Island Glacier (PIG), Antarctica
(1) Data generation
- Read_PIG.ipynb: Prepare regular-grid data from ISSM meshes for training CNN (convolutional neural network) benchmark models

(2) Model training
- ISSM_DGL_PIG2.py: Train GNN emulators on multi-GPU environments
- ISSM_DGL_PIG2_single.py: Train GNN emulators on single-GPU environments
- ISSM_FCN_PIG2.py: Train CNN emulators on multi-GPU environments

(3) Result analysis and visualization
- PIG2_results_cleanup.ipynb: Analyze emulator results and visualize maps

## Experiments for Helheim Glacier, Greenland
(1) Data generation
- Read_Helheim.ipynb: Prepare regular-grid data from ISSM meshes for training CNN benchmark models

(2) Model training
- ISSM_DGL_Helheim_flow.py: Train GNN emulators on multi-GPU environments (ISSM_DGL_Helheim.py: previous version only without numerical modeling of VM calving law)
- ISSM_DGL_Helheim_single.py: Train GNN emulators on single-GPU environments
- ISSM_FCN_Helheim.py: Train CNN emulators on multi-GPU environments

(3) Result analysis and visualization
- Helhiem_results_cleanup.ipynb: Analyze emulator results and visualize maps

### References
- Koo, Y., & Rahnemoonfar, M. (2025). Graph convolutional network as a fast statistical emulator for numerical ice sheet modeling. Journal of Glaciology, 71, e15. doi:10.1017/jog.2024.93
- Koo, Y., Cheng, G., Morlighem, M., and Rahnemoonfar, M.: Calibrating calving parameterizations using graph neural network emulators: Application to Helheim Glacier, East Greenland, EGUsphere [preprint], https://doi.org/10.5194/egusphere-2024-1620, 2024.
