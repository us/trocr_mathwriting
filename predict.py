import os
import csv
import argparse
import logging
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_models(checkpoint_path):
    logging.info(f"Loading models from checkpoint: {checkpoint_path}")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained(checkpoint_path, local_files_only=True)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 256
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4
    return processor, model

def process_image(image_path, processor, model):
    logging.info(f"Processing image: {image_path}")
    image = Image.open(image_path)
    
    # Padding settings
    padding = 10  # 10 pixels on each side
    pad_color = (255, 255, 255)  # pad with white color

    # Add padding
    image = ImageOps.expand(image, border=padding, fill=pad_color)

    # Resize image to 384x384
    image = image.resize((384, 384))
        
    pixel_values = processor(image, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values, output_logits=True, return_dict_in_generate=True)
    ids, scores = generated_ids['sequences'], generated_ids['logits']
    generated_text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    concat = torch.cat(scores, dim=0)
    confianza = (torch.mean(torch.max(F.softmax(concat, dim=1), dim=1).values)).item()

    logging.info(f"Generated text: {generated_text}, Confidence: {confianza}")
    return image_path, generated_text, confianza

def process_directory(input_dir, output_file, processor, model):
    results = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            result = process_image(image_path, processor, model)
            results.append(result)
    
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Generated Text", "Confidence"])
        writer.writerows(results)
    
    logging.info(f"Results written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images and output results to a CSV file.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing images to process')
    parser.add_argument('--output_file', type=str, required=True, help='Output CSV file to write results')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint')
    
    args = parser.parse_args()

    processor, model = load_models(args.checkpoint_path)
    
    process_directory(args.input_dir, args.output_file, processor, model)
