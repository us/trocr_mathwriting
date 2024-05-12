import dataclasses
import os
from xml.etree import ElementTree
import cairo
import math
import PIL
import PIL.Image
import numpy as np
import argparse
import os
import pandas as pd
import argparse
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

@dataclasses.dataclass
class Ink:
  """Represents a single ink, as read from an InkML file."""
  # Every stroke in the ink.
  # Each stroke array has shape (3, number of points), where the first
  # dimensions are (x, y, timestamp), in that order.
  strokes: list[np.ndarray]
  # Metadata present in the InkML.
  annotations: dict[str, str]

def read_inkml_file(filename: str) -> Ink:
  """Simple reader for MathWriting's InkML files."""
  with open(filename, "r") as f:
    root = ElementTree.fromstring(f.read())

  strokes = []
  annotations = {}

  for element in root:
    tag_name = element.tag.removeprefix('{http://www.w3.org/2003/InkML}')
    if tag_name == 'annotation':
      annotations[element.attrib.get('type')] = element.text

    elif tag_name == 'trace':
      points = element.text.split(',')
      stroke_x, stroke_y, stroke_t = [], [], []
      for point in points:
        x, y, t = point.split(' ')
        stroke_x.append(float(x))
        stroke_y.append(float(y))
        stroke_t.append(float(t))
      strokes.append(np.array((stroke_x, stroke_y, stroke_t)))

  return Ink(strokes=strokes, annotations=annotations)



def cairo_to_pil(surface: cairo.ImageSurface) -> PIL.Image.Image:
  """Converts a ARGB Cairo surface into an RGB PIL image."""
  size = (surface.get_width(), surface.get_height())
  stride = surface.get_stride()
  with surface.get_data() as memory:
    return PIL.Image.frombuffer(
        'RGB', size, memory.tobytes(), 'raw', 'BGRX', stride
    )


def render_ink(
    ink: Ink,
    *,
    margin: int = 10,
    stroke_width: float = 1.5,
    stroke_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    background_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    resize_dims: tuple[int, int] = None
) -> PIL.Image.Image:
  """Renders an ink as a PIL image using Cairo.

  The image size is chosen to fit the entire ink while having one pixel per
  InkML unit.

  Args:
    margin: size of the blank margin around the image (pixels)
    stroke_width: width of each stroke (pixels)
    stroke_color: color to paint the strokes with
    background_color: color to fill the background with

  Returns:
    Rendered ink, as a PIL image.
  """

  # Compute transformation to fit the ink in the image.
  xmin, ymin = np.vstack([stroke[:2].min(axis=1) for stroke in ink.strokes]).min(axis=0)
  xmax, ymax = np.vstack([stroke[:2].max(axis=1) for stroke in ink.strokes]).max(axis=0)
  width = int(xmax - xmin + 2*margin)
  height = int(ymax - ymin + 2*margin)

  shift_x = - xmin + margin
  shift_y = - ymin + margin

  def apply_transform(ink_x: float, ink_y: float):
    return ink_x + shift_x, ink_y + shift_y

  # Create the canvas with the background color
  surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
  ctx = cairo.Context(surface)
  ctx.set_source_rgb(*background_color)
  ctx.paint()

  # Set pen parameters
  ctx.set_source_rgb(*stroke_color)
  ctx.set_line_width(stroke_width)
  ctx.set_line_cap(cairo.LineCap.ROUND)
  ctx.set_line_join(cairo.LineJoin.ROUND)

  for stroke in ink.strokes:
    if len(stroke[0]) == 1:
      # For isolated points we just draw a filled disk with a diameter equal
      # to the line width.
      x, y = apply_transform(stroke[0, 0], stroke[1, 0])
      ctx.arc(x, y, stroke_width / 2, 0, 2 * math.pi)
      ctx.fill()

    else:
      ctx.move_to(*apply_transform(stroke[0,0], stroke[1,0]))

      for ink_x, ink_y in stroke[:2, 1:].T:
        ctx.line_to(*apply_transform(ink_x, ink_y))
      ctx.stroke()
    pil_image = cairo_to_pil(surface)

  # Resize the image if resize dimensions are provided
  if resize_dims is not None:
      pil_image = pil_image.resize(resize_dims, PIL.Image.LANCZOS)
    
  return pil_image


def process_file(filename, input_dir, output_dir, size):
    """Process an individual InkML file to render and save its corresponding image and collect annotations."""
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename.replace('.inkml', '.png'))

    # Read the ink from the InkML file
    ink = read_inkml_file(input_path)

    # Render the ink to an image with specified size
    image = render_ink(ink, resize_dims=(size[0], size[1]))

    # Save the image
    image.save(output_path)
    # file_name,label,splitTagOriginal,inkCreationMethod,sampleId,normalizedLabel
    return {
       'file_name': filename.replace('.inkml', '.png'),
       'label': ink.annotations['normalizedLabel'],
      #  'normalized_label': ink.annotations['normalizedLabel']
    }
    # return {'file_name': filename.replace('.inkml', '.png'), **ink.annotations}

def main():
    parser = argparse.ArgumentParser(description="Process InkML files and generate images and annotations.")
    parser.add_argument('--input', '-i', type=str, required=True, help='Directory containing InkML files')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output directory for images and CSV annotations')
    parser.add_argument('--size', '-s', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'), default=(384, 384), help='Dimensions of output images')

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)

    # Prepare to collect annotations
    annotations = []
    files = [f for f in os.listdir(args.input) if f.endswith('.inkml')]

    # Process each file in the input directory using parallel processing
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_file, files, [args.input]*len(files), [args.output]*len(files), [args.size]*len(files)), total=len(files)))

    # Extend the annotations list with the results from each process
    annotations.extend(results)

    # Save annotations to a CSV file
    df = pd.DataFrame(annotations)
    df.to_csv(os.path.join(args.output, 'metadata.csv'), index=False)

if __name__ == '__main__':
    main()
