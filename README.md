# LightCTR

LightCTR aims to provide modularized layers and models for deep-learning based click-through rate prediction problem.

- Provide model functions for `tf.estimator.Estimator` interface for large scale training.
- Provide deep learning building blocks `tf.keras.layers.Layer` which can be used to easily build custom models.
- Provide multi optimizer supprot for fine-grained model optimization strategy

## Models List

|                 Model                  | Paper                                                                                                                                                           |
| :------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|                 FTRL                   | [Ad Click Prediction: a View from the Trenches](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41159.pdf)                      |
|             Wide & Deep                | [Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf)                                                                            |
|                 MLR                    | [Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction](https://arxiv.org/pdf/1704.05194.pdf)                                         |
|                 FM                     | [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)                                                                            |
|                 FwFM                   | [Field-weighted Factorization Machines for Click-Through Rate Prediction in Display Advertising](https://arxiv.org/pdf/1806.03514.pdf)                          |
|                 FFM                    | [Field-aware Factorization Machines for CTR Prediction](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf)                                                      |
|                 NFM                    | [Neural Factorization Machines for Sparse Predictive Analytics](https://arxiv.org/pdf/1708.05027.pdf)                                                           |
|                 AFM                    | [Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks](https://arxiv.org/pdf/1708.04617.pdf)                  |
|                DeepFM                  | [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/pdf/1703.04247.pdf)                                                 |
|                 MVM                    | [Multi-View Factorization Machines](https://arxiv.org/pdf/1506.01110.pdf)                                                                                       |
|              Deep Crossing             | [Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features](https://www.kdd.org/kdd2016/papers/files/adf0975-shanA.pdf)                 |
|          Deep & Cross Network          | [Deep & Cross Network for Ad Click Predictions](https://arxiv.org/pdf/1708.05123.pdf)                                                                           |
|                 PNN                    | [Product-based Neural Networks for User Response Prediction over Multi-field Categorical Data ](https://arxiv.org/pdf/1807.00311.pdf)                           |
|               xDeepFM                  | [xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/pdf/1803.05170.pdf)                                   |
|               AutoInt                  | [AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/pdf/1810.11921.pdf)                                      |
|               FiBiNET                  | [FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09433.pdf)                |
|               xDeepInt                 | [xDeepInt: a hybrid architecture for modeling the vector-wise and bit-wise feature interactions](https://dlp-kdd.github.io/assets/pdf/a2-yan.pdf)               |
