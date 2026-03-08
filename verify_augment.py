import os
import cv2
from webapp.app.helpers import augment_image_variants
from PIL import Image

def test_augmentation():
    print("Testing augmentation variants...")
    
    # 1. Create a dummy image
    img_path = "test_augment.png"
    img = Image.new('RGBA', (200, 300), color = (255, 0, 0, 255))
    img.save(img_path)
    print(f"Dummy image created at {img_path}")
    
    output_dir = "test_variants"
    
    try:
        # 2. Run augmentation
        print("Running augment_image_variants...")
        variant_paths = augment_image_variants(img_path, output_dir)
        
        print(f"Generated {len(variant_paths)} variants:")
        for v in variant_paths:
            print(f"  - {v}")
            if not os.path.exists(v):
                raise Exception(f"Variant file not found: {v}")
        
        if len(variant_paths) == 4:
            print("Test passed!")
        else:
            print(f"Test failed: expected 4 variants, got {len(variant_paths)}")
            
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if os.path.exists(img_path):
            os.remove(img_path)
        # We'll leave the variants for manual inspection if needed, or clean up
        # for folder in [output_dir]:
        #     if os.path.exists(folder):
        #         for f in os.listdir(folder):
        #             os.remove(os.path.join(folder, f))
        #         os.rmdir(folder)

if __name__ == "__main__":
    test_augmentation()
