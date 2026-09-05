import glob
import os
from PIL import Image
from pathlib import Path
import argparse
import sys

def crop_to_alpha(img, padding=0, threshold=0):
    """
    Crop a PIL Image to bounding box of pixels with alpha > threshold
    """
    # Ensure image has alpha channel
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Get alpha channel
    alpha = img.getchannel('A')
    
    # Get bounding box of non-zero alpha pixels
    # Note: getbbox() uses threshold=1 by default, so we need to handle custom threshold
    if threshold > 0:
        # Convert to numpy for threshold control (simpler approach)
        import numpy as np
        alpha_array = np.array(alpha)
        
        # Find rows and columns with alpha > threshold
        rows = np.any(alpha_array > threshold, axis=1)
        cols = np.any(alpha_array > threshold, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return img
        
        # Find bounding box
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        
        bbox = (xmin, ymin, xmax + 1, ymax + 1)
    else:
        bbox = alpha.getbbox()
    
    if not bbox:
        return img
    
    # Apply padding if specified
    if padding > 0:
        left, upper, right, lower = bbox
        left = max(0, left - padding)
        upper = max(0, upper - padding)
        right = min(img.width, right + padding)
        lower = min(img.height, lower + padding)
        bbox = (left, upper, right, lower)
    
    # Crop image
    return img.crop(bbox)

def batch_crop_images(input_dir, output_dir=None, pattern='*.png', padding=0, threshold=0, 
                      recursive=False, overwrite=False, verbose=True):
    """
    Crop all PNG images in a directory with various options
    """
    input_path = Path(input_dir)
    
    # If output_dir is not specified, create a subdirectory
    if output_dir is None:
        output_path = input_path
    else:
        output_path = Path(output_dir)
    
     # Поддерживаемые форматы изображений
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff', '*.webp']
    
    # Находим все изображения рекурсивно
    image_files = []
    for extension in image_extensions:
        pattern = os.path.join(input_path, '**', extension)
        image_files.extend(glob.glob(pattern, recursive=True))
        
        pattern_upper = os.path.join(input_path, '**', extension.upper())
        image_files.extend(glob.glob(pattern_upper, recursive=True))
    
    # Убираем дубликаты (на случай если файлы с разным регистром совпадают)
    image_files = list(set(image_files))
    
    # Process all matching files
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for img_path in image_files:
        # Skip directories
        img_file = Path(img_path)
        if img_file.is_dir():
            continue
            
        output_file = img_file
        
        # Skip if file exists and overwrite is False
        
        try:
            # Open and process image
            img = Image.open(img_file)
            original_size = img.size
            
            cropped = crop_to_alpha(img, padding=padding, threshold=threshold)
            
            # Save with same format
            if img_file.suffix.lower() in ['.png', '.webp', '.tiff', '.tif']:
                # Preserve alpha for formats that support it
                cropped.save(output_file)
            else:
                # For formats without alpha support, convert to RGB
                cropped.convert('RGB').save(output_file)
            
            processed_count += 1
            if verbose:
                print(f"Processed: {img_file.name} "
                      f"({original_size[0]}x{original_size[1]} → "
                      f"{cropped.size[0]}x{cropped.size[1]})")
                
        except Exception as e:
            error_count += 1
            if verbose:
                print(f"Error processing {img_file.name}: {e}")
    
    # Summary
    if verbose:
        print(f"\nSummary:")
        print(f"  Processed: {processed_count}")
        print(f"  Skipped: {skipped_count}")
        print(f"  Errors: {error_count}")
        print(f"  Output directory: {output_path}")
    
    return processed_count, skipped_count, error_count

def main():
    parser = argparse.ArgumentParser(
        description='Batch crop PNG images to visible content (alpha > 0)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input_folder
  %(prog)s input_folder -o output_folder
  %(prog)s input_folder -p 10 -t 10
  %(prog)s input_folder -r -p 5 -v
  %(prog)s input_folder --pattern "*.webp" --overwrite
        """
    )
    
    # Required arguments
    
    # Optional arguments
    parser.add_argument('-o', '--output', help='Output directory (default: input/cropped)')
    parser.add_argument('-p', '--padding', type=int, default=0,
                       help='Padding around cropped area (pixels, default: 0)')
    parser.add_argument('-t', '--threshold', type=int, default=0,
                       help='Alpha threshold (0-255, default: 0)')
    parser.add_argument('--pattern', default='*.png',
                       help='File pattern to match (default: *.png)')
    parser.add_argument('-r', '--recursive', action='store_true',
                       help='Search subdirectories recursively')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing files in output directory')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Show detailed processing information')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Suppress all output (except errors)')
    
    args = parser.parse_args()
    
    # Check if input directory exists
    input_path = Path(os.path.dirname(os.path.abspath(__file__)))

    if not input_path.exists():
        print(f"Error: Input directory '{args.input}' does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Set verbose based on quiet flag
    verbose = not args.quiet if args.quiet else args.verbose
    
    # Process images
    try:
        processed, skipped, errors = batch_crop_images(
            input_dir=input_path,
            output_dir=input_path,
            pattern=args.pattern,
            padding=args.padding,
            threshold=args.threshold,
            recursive=args.recursive,
            overwrite=args.overwrite,
            verbose=verbose
        )
        
        if not args.quiet:
            if errors > 0:
                sys.exit(1)  # Exit with error code if there were errors
                
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()