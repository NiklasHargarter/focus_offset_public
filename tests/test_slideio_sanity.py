import slideio
from src import config


def test_slideio():
    dataset_name = "ZStack_HE"
    raw_dir = config.get_vsi_raw_dir(dataset_name)
    print(f"Checking raw directory: {raw_dir}")

    if not raw_dir.exists():
        print("Raw directory does not exist.")
        return

    slides = list(raw_dir.glob("*.vsi"))
    if not slides:
        print("No .vsi files found.")
        return

    slide_path = slides[0]
    print(f"Attempting to open slide: {slide_path}")

    try:
        slide = slideio.open_slide(str(slide_path), "VSI")
        scene = slide.get_scene(0)
        print(f"Successfully opened slide. Scene size: {scene.size}")

        # Try a small read
        rect = (0, 0, 1000, 1000)
        block = scene.read_block(rect=rect, size=(224, 224), slices=(0, 1))
        print(f"Successfully read block. Shape: {block.shape}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_slideio()
