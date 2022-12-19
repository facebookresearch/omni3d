# Results

1. [Cube R-CNN on Omni3D](#omni3d)
2. [Comparison with Competing Methods](#priormethods)
3. [Cube R-CNN on KITTI test](#kittitest)

## Cube R-CNN on Omni3D <a name="omni3d"></a>

The Cube R-CNN evaluation produces two tables which summarize performance on the evaluation set. The first is a performance analysis table and the second is the *main* Omni3D performance table. 

Below are the results of the `DLA34` Cube R-CNN model trained on the *full* Omni3D training set and evaluated on the test set.

1. Performance Analysis Table

|     Dataset      |  #iters  | AP2D    | AP3D    | AP3D@15   | AP3D@25   | AP3D@50   | AP3D-N   | AP3D-M   | AP3D-F   |
|------------------|:--------:|:-------:|:-------:|:---------:|:---------:|:---------:|:--------:|:--------:|:--------:|
|   SUNRGBD_test   |  final   | 16.0053 | 15.3321 | 21.6640   | 16.9483   | 5.13407   | 15.3321  | nan      | nan      |
|  Hypersim_test   |  final   | 12.2447 | 7.47746 | 10.0416   | 7.58102   | 2.25935   | 7.95103  | 0.586047 | 0        |
| ARKitScenes_test |  final   | 41.3007 | 41.7261 | 53.0945   | 45.4191   | 19.2622   | 41.7267  | 0        | nan      |
|  Objectron_test  |  final   | 56.4603 | 50.8374 | 65.6977   | 54.0398   | 22.4584   | 50.8374  | nan      | nan      |
|    KITTI_test    |  final   | 41.3125 | 32.5909 | 41.9073   | 34.5554   | 16.236    | 56.6344  | 36.0952  | 16.4902  |
|  nuScenes_test   |  final   | 36.31   | 30.059  | 39.1756   | 32.1927   | 14.5962   | 47.4605  | 34.8069  | 11.9634  |
|   **Concat**     |  final   | 27.6387 | 23.266  | 30.8423   | 24.865    | 9.51163   | 27.9432  | 12.0738  | 8.49733  |

2. Omni3D Performance Table -- To be used to compare with Cube R-CNN

|     Dataset      |  #iters  | AP2D    | AP3D    |
|------------------|:--------:|:-------:|:-------:|
|   SUNRGBD_test   |  final   | 16.0053 | 15.3321 |
|  Hypersim_test   |  final   | 12.2447 | 7.47746 |
| ARKitScenes_test |  final   | 41.3007 | 41.7261 |
|  Objectron_test  |  final   | 56.4603 | 50.8374 |
|    KITTI_test    |  final   | 41.3125 | 32.5909 |
|  nuScenes_test   |  final   | 36.31   | 30.059  |
|    Omni3D_Out    |  final   | 38.8662 | 33.0019 |
|    Omni3D_In     |  final   | 23.4123 | 20.0325 |
|    **Omni3D**    |  final   | 27.6387 | 23.266  |

The Omni3D entry (last row) gives performance on the *full Omni3D test set*. This is what we report in our paper and what should be used to compare to Cube R-CNN.
The tables also report performance on the outdoor and indoor subsets of the test set, in the Omni3D_Out and Omni3D_In entries respectively.

### Performance on Omni3D Indoor and Outdoor

Below we provide Cube R-CNN performance when trained and evaluated on the indoor (Omni3D_In) and outdoor (Omni3D_Out) splits.

<details><summary>Omni3D_In</summary>

Here we **train and evaluate** Cube R-CNN on the Omni3D_In split which consists of {Hypersim, SUN RGB-D, ARKitScenes}.

1. Performance Analysis Table

|     Dataset      |  #iters  | AP2D    | AP3D    | AP3D@15   | AP3D@25   | AP3D@50   | AP3D-N   | AP3D-M   | AP3D-F   |
|------------------|:--------:|:-------:|:-------:|:---------:|:---------:|:---------:|:--------:|:--------:|:--------:|
|   SUNRGBD_test   |  final   | 18.1246 | 16.7919 | 23.7455   | 18.8856   | 5.3282    | 16.792   | nan      | nan      |
|  Hypersim_test   |  final   | 13.3741 | 7.30641 | 9.75961   | 7.26068   | 2.61309   | 7.81662  | 0.690567 | 0        |
| ARKitScenes_test |  final   | 43.7697 | 43.5845 | 56.7073   | 47.2705   | 18.9982   | 43.5858  | 0        | nan      |
|    **Concat**    |  final   | 19.2801 | 15.0396 | 20.4719   | 16.1827   | 5.51773   | 15.5207  | 0.66831  | 0        |

2. Omni3D Performance Table

|     Dataset      |  #iters  | AP2D    | AP3D    |
|------------------|:--------:|:-------:|:-------:|
|   SUNRGBD_test   |  final   | 18.1246 | 16.7919 |
|  Hypersim_test   |  final   | 13.3741 | 7.30641 |
| ARKitScenes_test |  final   | 43.7697 | 43.5845 |
|    Omni3D_Out    |  final   | nan     | nan     |
|  **Omni3D_In**   |  final   | 19.2801 | 15.0396 |
|      Omni3D      |  final   | nan     | nan     |

</details>

<details><summary>Omni3D_Out</summary>

Here we **train and evaluate** Cube R-CNN on the Omni3D_Out split which consists of {KITTI, nuScenes}.

1. Performance Analysis Table

|    Dataset    |  #iters  | AP2D    | AP3D    | AP3D@15   | AP3D@25   | AP3D@50   | AP3D-N   | AP3D-M   | AP3D-F   |
|---------------|:--------:|:-------:|:-------:|:---------:|:---------:|:---------:|:--------:|:--------:|:--------:|
| nuScenes_test |  final   | 38.3153 | 32.6165 | 41.1472   | 33.8193   | 17.7208   | 43.6913  | 37.7967  | 15.0941  |
|  KITTI_test   |  final   | 43.7112 | 35.9988 | 44.8352   | 37.5502   | 19.7639   | 52.0115  | 40.0687  | 16.3201  |
|  **Concat**   |  final   | 39.1041 | 31.8352 | 40.2945   | 33.0542   | 16.809    | 45.3547  | 37.2141  | 13.9689  |

2. Omni3D Performance Table

|    Dataset    |  #iters  | AP2D    | AP3D    |
|---------------|:--------:|:-------:|:-------:|
| nuScenes_test |  final   | 38.3153 | 32.6165 |
|  KITTI_test   |  final   | 43.7112 | 35.9988 |
|**Omni3D_Out** |  final   | 39.1041 | 31.8352 |
|   Omni3D_In   |  final   | nan     | nan     |
|    Omni3D     |  final   | nan     | nan     |

</details>

## Comparison with Competing Methods on Omni3D <a name="priormethods"></a>

We compare Cube R-CNN to recent state-of-the-art 3D object detection methods [SMOKE][smoke], [FCOS3D][fcos3d], [PGD][pgd] and [ImVoxelNet][imvx]. We augment the first three with our *virtual camera (vc)* feature. ImVoxelNet by design uses camera intrinsics to unproject to 3D.

|    Method            |  AP3D |
|----------------------|:-----:|
| [ImVoxelNet][imvx]   |  9.4  |
| [SMOKE][smoke]       |  9.6  | 
|[SMOKE][smoke] + vc   | 10.4  |
| [FCOS3D][fcos3d]     |  9.8  | 
|[FCOS3D][fcos3d] + vc |  10.6 |
|   [PGD][PGD]         |  11.2 | 
|  [PGD][PGD]  + vc    | 15.4  |
| Cube R-CNN           | 23.3  |

## Cube R-CNN Performance KITTI test <a name="kittitest"></a>

We evaluate and compare Cube R-CNN on KITTI test using [KITTI's evaluation server][kittieval] which reports AP3D at a 70\% 3D IoU. Note that Cube R-CNN is not tuned for the KITTI benchmark. We train Cube R-CNN on KITTI only and do not perform any data augmentations or model ensembling at test time.

|      Method        | Easy  | Med   | Hard  |
|--------------------|:-----:|:-----:|:-----:|
| [SMOKE][smoke]     |14.03  |9.76   | 7.84  |
| [ImVoxelNet][imvx] | 17.15 |10.97  |9.15   |
|  [PGD][PGD]        | 19.05 |11.76  |9.39   |
| [GUPNet][gupnet]   |  22.26|15.02  |13.12  |
| Cube R-CNN         | 23.59  |15.01 |12.56  |


[m3d]: https://arxiv.org/abs/1907.06038
[gupnet]: https://arxiv.org/abs/2107.13774
[smoke]: https://arxiv.org/abs/2002.10111
[fcos3d]: https://arxiv.org/abs/2104.10956
[pgd]: https://arxiv.org/abs/2107.14160
[imvx]: https://arxiv.org/abs/2106.01178
[kittieval]: https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d