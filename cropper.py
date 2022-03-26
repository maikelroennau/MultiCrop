import argparse
from pathlib import Path

import cv2
import numpy as np


class MultiCrop:
    def __init__(self, base_image, images, mode=None, width=None, height=None, output_path="."):
        self.DYNAMIC_MODE = 0
        self.FIXED_MODE = 1
        self.MAX_HEIGHT = 960
        self.MAX_WIDTH = 1280

        self.name = base_image
        self.original_image = cv2.imread(base_image)
        self.original_height = self.original_image.shape[0]
        self.original_width = self.original_image.shape[1]

        self.new_height = None
        self.new_width = None

        self.base_image = self.original_image.copy()

        if self.original_width > self.MAX_WIDTH:
            self.new_width = self.MAX_WIDTH

            ratio = self.MAX_HEIGHT / self.original_height
            self.new_height = int(self.original_height * ratio)
            self.base_image = cv2.resize(self.base_image, (self.MAX_WIDTH, self.new_height,))

        self.image = self.base_image.copy()
        self.images = images
        self.mode = self.parse_mode(mode)
        self.output_path = output_path

        self.x = 0
        self.y = 0
        if self.mode == self.DYNAMIC_MODE:
            self.h = 0
            self.w = 0
        elif self.mode == self.FIXED_MODE:
            if height is None or width is None:
                raise ValueError("`size` can't be `None` when `mode` is `fixed`.")
            if self.original_width != self.base_image.shape[1]:
                height = int(height * (self.MAX_HEIGHT / self.original_height))
                width = int(width * (self.MAX_WIDTH / self.original_width))
            self.h = height
            self.w = width

        self.drawing = False


    def __del__(self):
        cv2.destroyAllWindows()


    def parse_mode(self, mode):
        if mode.lower() == "dynamic":
            return self.DYNAMIC_MODE
        elif mode.lower() == "fixed":
            return self.FIXED_MODE


    def update(self, x, y):
        self.h = abs(self.y - y)
        self.w = abs(self.x - x)
        self.x = min(self.x, x)
        self.y = min(self.y, y)


    def integrate_dynamic(self):
        if self.h != 0 and self.w != 0:
            self.image = self.base_image.copy()
            roi = self.image[self.y:self.y+self.h, self.x:self.x+self.w, :]
            rect = np.ones_like(roi, dtype=np.uint8) * 255
            self.image[self.y:self.y+self.h, self.x:self.x+self.w, :] = cv2.addWeighted(roi, 1.0, rect, 0.2, 1.0)


    def integrate_fixed(self, x, y):
        width = x
        height = y

        width = min(max(0, width - self.w // 2), self.MAX_WIDTH - self.h)
        height = min(max(0, height - self.h // 2), self.MAX_HEIGHT - self.w)

        self.image = self.base_image.copy()
        roi = self.image[height:height+self.w, width:width+self.h, :]
        rect = np.ones_like(roi, dtype=np.uint8) * 255
        self.image[height:height+self.w, width:width+self.h, :] = cv2.addWeighted(roi, 1.0, rect, 0.2, 1.0)


    def draw_rectangle(self, event, x, y, flags, param):
        if self.mode == self.DYNAMIC_MODE:
            if event == cv2.EVENT_LBUTTONDOWN:
                    self.drawing = True
                    self.x = x
                    self.y = y
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.drawing:
                    self.update(x, y)
                    self.integrate_dynamic()
            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                self.integrate_dynamic()

        elif self.mode == self.FIXED_MODE:
            if event == cv2.EVENT_LBUTTONDOWN:
                    self.drawing = True
                    self.x = x
                    self.y = y
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.drawing:
                    self.integrate_fixed(x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                self.x = x
                self.y = y

        cv2.imshow("Input", self.image)


    def fixed_crop(self):
        width = self.x
        height = self.y

        width = min(max(0, width - self.w // 2), self.MAX_WIDTH - self.h)
        height = min(max(0, height - self.h // 2), self.MAX_HEIGHT - self.w)

        width = int(width * (self.original_width / self.MAX_WIDTH))
        height = int(height * (self.original_height / self.MAX_HEIGHT))

        w = int(self.w * (self.original_width / self.MAX_WIDTH))
        h = int(self.h * (self.original_height / self.MAX_HEIGHT))

        print(f"[{height}:{height+w}, {width}:{width+h}]")
        crop = self.original_image[height:height+w, width:width+h, :]
        cv2.imwrite(str(Path(self.output_path).joinpath(f"{Path(self.name).name}_crop.png")), crop)

        for image_path in self.images:
            image = cv2.imread(image_path)
            crop =  image[height:height+w, width:width+h, :]
            cv2.imwrite(str(Path(self.output_path).joinpath(f"{Path(image_path).name}_crop.png")), crop)


    def run(self):
        # Draw rectangle
        cv2.namedWindow("Input")
        cv2.setMouseCallback("Input", self.draw_rectangle)
        cv2.imshow("Input", self.image)

        print("\nMultiCrop")
        print("\nCommands:")
        print("  - Esc        ==> Exit.")
        print("  - Left mouse ==> Draw or drag rectangle.")
        print("  - Space      ==> Crop")

        output_path = Path(self.output_path)
        output_path.mkdir(exist_ok=True, parents=True)

        while True:
            key = cv2.waitKey(0)
            if key == 32:
                if self.mode == self.DYNAMIC_MODE:
                    pass
                elif self.mode == self.FIXED_MODE:
                    self.fixed_crop()
            if key == 27 or key == 13:
                break


def main():
    parser = argparse.ArgumentParser(description="Crop multiple images based on a main image.")

    parser.add_argument(
        "-b",
        "--base-image",
        help="Base image.",
        required=True,
        type=str)

    parser.add_argument(
        "-i",
        "--images",
        help="Images to crop.",
        required=True,
        nargs="+",
        type=str)

    parser.add_argument(
        "-o",
        "--output",
        help="Path where to save the cropped images.",
        default="cropped",
        type=str)

    parser.add_argument(
        "-m",
        "--mode",
        help="Either `dynamic` or `fixed`. In dynamic mode (default) crop size is defined on the fly. In fixed mode crop size is a box with fixed with that must be placed at the desired location.",
        default="dynamic"
    )

    parser.add_argument(
        "-s",
        "--size",
        help="Dimensions of the fixed size crop in the format `HEIGHTxWIDTH`. Only valid if `mode` is `fixed`."
    )

    args = parser.parse_args()

    if args.mode == "fixed":
        size = args.size.split("x")
        width = int(size[0])
        height = int(size[1])
    else:
        width = None
        height = None

    grab_cut = MultiCrop(args.base_image, args.images, args.mode, width, height, args.output)
    grab_cut.run()


if __name__ == "__main__":
    main()
