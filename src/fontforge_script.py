"""
Fixed FontForge Script for Your Filename Format
"""

import fontforge
from pathlib import Path
import logging
import re
import sys
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_unicode_from_filename(filename: str) -> int:
    """
    Extract Unicode value from your specific filename formats:
    - 000_A_000.svg (character format)
    - 052_0_000.svg (numeric format)
    Returns the first number group as Unicode value
    """
    match = re.search(r'^(\d+)', filename)
    if not match:
        raise ValueError(f"Cannot parse Unicode from filename: {filename}")
    return int(match.group(1))

def create_font_from_svg() -> None:
    try:
        svg_dir = Path("data/generated_characters")
        output_path = Path("data/fonts/MyHandwriting.otf")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Verify SVG files exist
        svg_files = list(svg_dir.glob("*.svg"))
        if not svg_files:
            raise FileNotFoundError(f"No SVG files found in {svg_dir}")

        logger.info(f"Found {len(svg_files)} SVG files")
        
        # Create font
        font = fontforge.font()
        font.fontname = "MyHandwriting"
        font.familyname = "My Handwriting"
        font.fullname = "My Custom Handwriting Font"
        
        # Process each SVG
        for svg_file in svg_files:
            try:
                unicode_val = parse_unicode_from_filename(svg_file.name)
                char_name = f"uni{unicode_val:04X}"
                
                logger.debug(f"Processing {svg_file.name} as U+{unicode_val:04X}")
                
                # Create glyph
                glyph = font.createChar(unicode_val, char_name)
                glyph.importOutlines(str(svg_file))
                glyph.correctDirection()
                
                # Validate
                if not glyph.isWorthOutputting():
                    logger.warning(f"Glyph empty in {svg_file.name} - skipping")
                    font.removeGlyph(glyph)
                    continue
                    
                logger.info(f"Added glyph for U+{unicode_val:04X} from {svg_file.name}")
                
            except Exception as e:
                logger.error(f"Error processing {svg_file.name}: {str(e)}")
                continue

        # Final checks
        if len(font) == 0:
            raise ValueError("No valid glyphs were added to the font")
        
        # Set font metrics
        font.ascent = 800
        font.descent = 200
        font.em = 1000
        
        # Generate font
        logger.info(f"Generating font with {len(font)} glyphs...")
        font.generate(output_path.as_posix())
        logger.info(f"Successfully created font at {output_path}")

    except Exception as e:
        logger.error(f"Font creation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if "--debug" in sys.argv:
        logger.setLevel(logging.DEBUG)
    
    create_font_from_svg()