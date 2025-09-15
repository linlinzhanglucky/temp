# Apr 24
## Trained:
- blob & beam, region, 10k, 60e, 1c
  - blob best: 0.0047 | 0.0049
  - beam best: 0.0037 | 0.0036

## Todo:
- Read code: beam.py & blob.py
  - multi-process works on Mac
  - data collection
  - Merged region & uni data collection
  - Try DATA_COLLECTION = False for running simulation window
  - Current version: uni works, region does not work
- Debug 
  - BEAM_TYPE = "region"
  - BLOB_TYPE = "region"
- Data collection
  - 2 random regions beam, 12 k data
  - 2 random regions blob, 12 k data
  - run data_preprocess.py, it will clean empty folders
  - upload zip to drive

# Apr 26
## Trained:
- beam, region, 10k, 100e, 1c, ViT Base
  - best: [Epoch 84] Train Loss: 0.003234 | Val Loss: 0.003301
  - learning cruve:
  - ![lc](./../images/vis_apr26/output_b2_base_loss_curve.png)
  - Results
  - ![r](./../images/vis_apr26/84.png)

# Apr 30
## Trained:
- beam, 2 region, 7k+, 100e, 1c, ViT Tiny
  - learning curves
  - ![log](./../images/vis_apr30/output_b1_tiny_loss_curve_log.png)
  - results
  - ![r](./../images/vis_apr30/72.png)
  - 3 channels
  - ![rgb](./../images/vis_apr30/72_channels_base.png)

## Visualization:
- Beam region, 10k ViT Base, 3 channels compare:
- ![rgb](./../images/vis_apr30/84_channels.png)
- Log plot:
- ![log](./../images/vis_apr30/output_b2_base_loss_curve_log.png)

# May 26
## Trained:
- beam, 1 region, 10k+, 100e, 1c, ViT, with initial vel
  - learning curves
  - ![log](./../images/vis_may26/output_b1_tiny_loss_curve_log.png)
  - results
  - ![r](./../images/vis_may26/1.png)
  - ![r](./../images/vis_may26/100.png)
  - 3 channels
  - ![rgb](./../images/vis_may26/1_channels_base.png)
  - ![rgb](./../images/vis_may26/1_channels_base.png)

# May 27
## Dataset: mpm_output_0527_decouple
- beam, 1 region, 10k, with initial vel
 -learning curves
  - learning curves
  - ![log](./../images/vis_may27/output_b1_tiny_loss_curve_log.png)
  - results
  - ![r](./../images/vis_may27/1.png)
  - ![r](./../images/vis_may27/100.png)
  - 3 channels
  - ![rgb](./../images/vis_may27/1_channels_base.png)
  - ![rgb](./../images/vis_may27/100_channels_base.png)
