# trocr_mathwriting

## Overview
**trocr_mathwriting** is an innovative project that utilizes the TrOCR model by Microsoft to convert images of mathematical handwriting into machine-readable text. This project specifically focuses on adapting TrOCR to work effectively with the MathWriting dataset, a new and comprehensive dataset comprised of handwritten mathematical expressions.

### About the MathWriting Dataset
MathWriting is the largest online handwritten mathematical expression (HME) dataset currently available. It includes over 230,000 human-written samples along with an additional 400,000 synthetic samples, making it ideal for both online and offline HME recognition tasks. This dataset surpasses the size of all existing offline HME datasets like IM2LATEX-100K.

## Installation

To set up the **trocr_mathwriting** project, follow these steps:

```bash
git clone https://github.com/us/trocr_mathwriting.git
cd trocr_mathwriting
pip install -r requirements.txt
```

## Dataset Preparation

- **inkml_to_image.py**: This script converts .inkml files to images and creates metadata.csv for `file_name` and `label`, which are then used by the model for training.

  Usage:
  ```bash
  python inkml_to_image.py --help
  
  usage: inkml_to_image.py [-h] --input INPUT --output OUTPUT
                         [--size WIDTH HEIGHT]

    Process InkML files and generate images and annotations.

    options:
    -h, --help            show this help message and exit
    --input INPUT, -i INPUT
                            Directory containing InkML files
    --output OUTPUT, -o OUTPUT
                            Output directory for images and CSV
                            annotations
    --size WIDTH HEIGHT, -s WIDTH HEIGHT
                            Dimensions of output images
  ```


## Usage

The workflow for using this project is as follows:

1. **Prepare the Dataset**: Convert the MathWriting dataset (in INKML format) to images using the `inkml_to_image.py` script and save these images along with their annotations in the `mw_images` directory. I used default TROCR input size `384x384` as width and height !

   ```bash
   python inkml_to_image.py -i mw/train -o mw_images/train
   python inkml_to_image.py -i mw/test -o mw_images/test
   python inkml_to_image.py -i mw/symbols -o mw_images/symbols
   python inkml_to_image.py -i mw/synthetic -o mw_images/synthetic
   python inkml_to_image.py -i mw/valid -o mw_images/valid
   ```

2. **Fine-Tuning the TrOCR Model**: Once the images are prepared, use the `Fine_tune_TrOCR_on_MathWriting_Database_using_Seq2SeqTrainer.ipynb` notebook to fine-tune the TrOCR model on the MathWriting images.

   Note: Ensure that you have downloaded the MathWriting dataset, renamed it as `mw`, and have created the `mw_images` dataset according to the preparation step.


## Contributing

Contributions to **trocr_mathwriting** are welcome. Please ensure to follow the existing coding standards and include tests with your code if applicable.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

