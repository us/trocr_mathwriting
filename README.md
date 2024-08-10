# TrOCR on MathWriting

## Overview

**TrOCR on MathWriting** is an innovative project that leverages the power of the TrOCR model developed by Microsoft to convert images of mathematical handwriting into machine-readable LaTeX text. This project is uniquely tailored to work with the MathWriting dataset, offering a comprehensive solution for recognizing handwritten mathematical expressions. Explore more about the model on our [Hugging Face repository](https://huggingface.co/us4/trocr-mathwriting).

You can access the pretrained model from https://huggingface.co/us4/trocr-mathwriting .


### About the MathWriting Dataset

MathWriting is the largest online dataset of handwritten mathematical expressions (HME) available today. It contains over **230,000** human-written samples and an additional **400,000** synthetic samples, making it the ideal dataset for both online and offline HME recognition tasks. MathWriting surpasses the size of all existing offline HME datasets like IM2LATEX-100K, providing unparalleled resources for researchers and developers.

## Hugging Face Model Usage

Leverage the pre-trained models with Hugging Face:

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Load the processor and model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("us4/trocr-mathwriting")
```

## Installation

To set up the **TrOCR on MathWriting** project, follow these steps:

```bash
git clone https://github.com/us/trocr_mathwriting.git
cd trocr_mathwriting
pip install -r requirements.txt
```

## Dataset Preparation

Prepare your dataset using our conversion script:

- **inkml_to_image.py**: This script converts `.inkml` files into images and creates a `metadata.csv` file containing `file_name` and `label`, which are essential for model training.

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
                          Output directory for images and CSV annotations
    --size WIDTH HEIGHT, -s WIDTH HEIGHT
                          Dimensions of output images
  ```

## Usage

The workflow for using this project is as follows:

1. **Prepare the Dataset**: Convert the MathWriting dataset (in INKML format) to images using the `inkml_to_image.py` script, and save these images along with their annotations in the `mw_images` directory. We use the default TrOCR input size `384x384` for width and height.

   ```bash
   python inkml_to_image.py -i mw/train -o mw_images/train
   python inkml_to_image.py -i mw/test -o mw_images/test
   python inkml_to_image.py -i mw/symbols -o mw_images/symbols
   python inkml_to_image.py -i mw/synthetic -o mw_images/synthetic
   python inkml_to_image.py -i mw/valid -o mw_images/valid
   ```

2. **Fine-Tuning the TrOCR Model**: Once the images are prepared, use the `Fine_tune_TrOCR_on_MathWriting_Database_using_Seq2SeqTrainer.ipynb` notebook to fine-tune the TrOCR model on the MathWriting images.

   **Note**: Ensure that you have downloaded the MathWriting dataset, renamed it as `mw`, and have created the `mw_images` dataset according to the preparation step.

3. **Prediction**

   Use the `predict.py` script to process images and output results:

   ```bash
   python predict.py --help
   usage: predict.py [-h] --input_dir INPUT_DIR --output_file OUTPUT_FILE
                     --checkpoint_path CHECKPOINT_PATH

   Process images and output results to a CSV file.

   options:
     -h, --help            show this help message and exit
     --input_dir INPUT_DIR
                           Directory containing images to process
     --output_file OUTPUT_FILE
                           Output CSV file to write results
     --checkpoint_path CHECKPOINT_PATH
                           Path to the model checkpoint
   ```


## Contributing

Contributions to **TrOCR on MathWriting** are welcome. Please ensure to follow the existing coding standards and include tests with your code if applicable.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Discover the future of handwritten mathematical expression recognition with TrOCR on MathWriting. Visit our [Hugging Face repository](https://huggingface.co/us4/trocr-mathwriting) to explore more and get started today!
