# main_program.py
from yolo_object_detection import perform_object_detection
from yolo_initialization import initialize_yolo

def main():
    initialize_yolo()
    perform_object_detection("path/to/your/image.jpg")

if __name__ == "__main__":
    main()