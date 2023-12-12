# Expression-Deciphered
Course Final Project

## Training Steps

1. To be added

## Explain the Model

1. **Train Your Own Model or Use the Provided [Model](https://drive.google.com/file/d/1Wqx9NfS51fGHNFW1JDjFCZ5ZOjGZ1Rev/view?usp=sharing)**
2. **Generte Heat Maps **
    - Place the trained model in the `models` folder.
    - Both `attentionMap_FER` and `gradcam_FER.py` can be executed directly; please modify the corresponding paths.

    Example command:
    ```bash
    python attentionMap_FER.py
    python gradcam_FER.py
    ```
## Acknowledgements

We would like to express our gratitude to these repos[Grad-CAM]([https://link-to-their-work.com](https://github.com/1Konny/gradcam_plus_plus-pytorch)https://github.com/1Konny/gradcam_plus_plus-pytorch)
and [Attention-transfer](https://github.com/szagoruyko/attention-transfer). We base our project on their codes.

## References:
[1] Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization, Selvaraju et al, ICCV, 2017 <br>
[2] Grad-CAM++: Generalized Gradient-based Visual Explanations for Deep Convolutional Networks, Chattopadhyay et al, WACV, 2018 <br>
[3] Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer, Sergey Zagoruyko and Nikos Komodakis, ICLR, 2017
