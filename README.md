# MoveR_AT_GREAT
## It’s GREAT: Gesture REcognition for Arm Translation.

Myoelectric control has emerged as a promising approach for a wide range of applications, including controlling limb prosthetics, teleoperating robots and enabling immersive interactions in the metaverse. However, the accuracy and robustness of MEC systems are often affected by various factors, including muscle fatigue, perspiration, drifts in electrode positions and changes in arm position. The latter has received less attention despite its significant impact on signal quality and decoding accuracy. To address this gap, we present GREAT, a novel dataset of surface electromyographic (EMG) signals captured from multiple arm positions. This dataset, comprising EMG and hand kinematics data from 8 participants performing 6 different hand gestures, provides a comprehensive resource for investigating position-invariant Myoelectric control decoding algorithms. We envision this dataset to serve as a valuable resource for both training and benchmarking arm position-invariant Myoelectric control algorithms. Additionally, to further expand the publicly available data capturing the variability of EMG signals across diverse arm positions, we propose a novel data acquisition protocol that can be utilized for future data collection.

![_protocol_figure](https://github.com/KatarzynaSzymaniak/arm_dataset_translation/assets/66835164/623a1e17-f2f7-4120-b2c7-6efb262e0174)

Data collection protocol: https://github.com/MoveR-Digital-Health-and-Care-Hub/posture_dataset_collection

Gestures:
  '1': power
  '2': lateral
  '3': tripod
  '4': pointer
  '5': open
  '6': rest

To reproduce the code get the same seed (200) and feats/params invluded in config files (*.yaml).

Using ./datautil/preprocessing/aug_preprocessing/extract_aug_data.py extract features from the data. 
To generate 5-fold cross validation and then calculate params in each fold use ./datautil/fold_generator.py followed by ./datautil/great_dataset.py

To run experiments, check if you want to change ./experiment_configs/exp/{...} file for any experiment running from ./experiments
  
