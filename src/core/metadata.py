from PIL import Image

class MetadataEngine:
    @staticmethod
    def strip_metadata(image_path: str, output_path: str) -> bool:
        """
        Removes all EXIF, GPS, and other metadata from the image.
        This roughly corresponds to Feature 1.2 (Metadata Nuke).
        
        Note: The 'Quality Guard' (Feature 1.3) is implicitly handled here by saving as a new PNG,
        which typically drops original EXIF data unless explicitly copied.
        """
        try:
            img = Image.open(image_path)
            
            # Creating a new image and saving it without copying info dict effectively strips metadata
            data = list(img.getdata())
            image_without_exif = Image.new(img.mode, img.size)
            image_without_exif.putdata(data)
            
            image_without_exif.save(output_path, "PNG")
            return True
        except Exception as e:
            print(f"Error stripping metadata: {e}")
            return False
