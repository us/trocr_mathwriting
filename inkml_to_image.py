import dataclasses
import os
from xml.etree.ElementTree import parse
import cairo
import math
import PIL.Image
import numpy as np
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm

@dataclasses.dataclass
class Ink:
    strokes: list[np.ndarray]
    annotations: dict[str, str]

def read_inkml_file(file_path: str) -> Ink:
    """Reads InkML file from path directly to save memory and processing time."""
    tree = parse(file_path)
    root = tree.getroot()

    strokes = []
    annotations = {}
    for element in root:
        tag_name = element.tag.removeprefix('{http://www.w3.org/2003/InkML}')
        if tag_name == 'annotation':
            annotations[element.attrib.get('type')] = element.text
        elif tag_name == 'trace':
            points = element.text.split(',')
            stroke_x, stroke_y, stroke_t = zip(*(map(float, point.split()) for point in points))
            strokes.append(np.array([stroke_x, stroke_y, stroke_t]))
    return Ink(strokes=strokes, annotations=annotations)

def cairo_to_pil(surface: cairo.ImageSurface) -> PIL.Image.Image:
    """Converts a Cairo surface into a PIL image."""
    size = (surface.get_width(), surface.get_height())
    stride = surface.get_stride()
    with surface.get_data() as memory:
        return PIL.Image.frombuffer('RGB', size, memory.tobytes(), 'raw', 'BGRX', stride)

def render_ink(ink: Ink, margin: int = 10, stroke_width: float = 1.5, stroke_color: tuple[float, float, float] = (0, 0, 0), background_color: tuple[float, float, float] = (1, 1, 1), resize_dims: tuple[int, int] = None) -> PIL.Image.Image:
    """Renders an ink as a PIL image using Cairo."""
    xmin, ymin = np.min([np.min(stroke[:2], axis=1) for stroke in ink.strokes], axis=0)
    xmax, ymax = np.max([np.max(stroke[:2], axis=1) for stroke in ink.strokes], axis=0)
    width, height = int(xmax - xmin + 2 * margin), int(ymax - ymin + 2 * margin)

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(*background_color)
    ctx.paint()
    ctx.set_source_rgb(*stroke_color)
    ctx.set_line_width(stroke_width)
    ctx.set_line_cap(cairo.LineCap.ROUND)
    ctx.set_line_join(cairo.LineJoin.ROUND)

    shift_x, shift_y = -xmin + margin, -ymin + margin
    for stroke in ink.strokes:
        points = np.array(stroke)
        if len(points[0]) == 1:
            x, y = points[0][0] + shift_x, points[1][0] + shift_y
            ctx.arc(x, y, stroke_width / 2, 0, 2 * math.pi)
            ctx.fill()
        else:
            x, y = points[0] + shift_x, points[1] + shift_y
            ctx.move_to(x[0], y[0])
            for xi, yi in zip(x[1:], y[1:]):
                ctx.line_to(xi, yi)
            ctx.stroke()

    pil_image = cairo_to_pil(surface)
    if resize_dims:
        pil_image = pil_image.resize(resize_dims, PIL.Image.Resampling.LANCZOS)
    return pil_image

def process_file(file_path, output_dir, size):
    """Process an individual InkML file to render and save its corresponding image and collect annotations."""
    output_path = os.path.join(output_dir, os.path.basename(file_path).replace('.inkml', '.png'))

    # Read and render the ink from the InkML file
    ink = read_inkml_file(file_path)
    image = render_ink(ink, resize_dims=size)

    # Save the image to disk
    image.save(output_path)

    # Collect relevant annotations for metadata
    return {
        'file_name': os.path.basename(output_path),
        'label': ink.annotations.get('normalizedLabel', ink.annotations.get('label', ""))
    }

def list_files(directory):
    """Efficiently list InkML files in a directory using os.scandir."""
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith('.inkml'):
                yield entry.path

def batch_process(input_dir, output_dir, size, num_workers):
    """Processes files in batches using parallel processing."""
    annotations = []
    files = list_files(input_dir)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_file, file, output_dir, size): file for file in files}
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                result = future.result()
                annotations.append(result)
            except Exception as e:
                print(f"Error processing {futures[future]}: {str(e)}")

    return annotations

def main():
    parser = argparse.ArgumentParser(description="Process InkML files and generate images and annotations.")
    parser.add_argument('--input', '-i', type=str, required=True, help='Directory containing InkML files')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output directory for images and CSV annotations')
    parser.add_argument('--size', '-s', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'), default=(384, 384), help='Dimensions of output images')

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)

    # Process files in parallel and collect annotations
    annotations = batch_process(args.input, args.output, args.size, num_workers=os.cpu_count())  # Adjust num_workers as needed

    # Save annotations to a CSV file
    pd.DataFrame(annotations).to_csv(os.path.join(args.output, 'metadata.csv'), index=False)

if __name__ == '__main__':
    main()
