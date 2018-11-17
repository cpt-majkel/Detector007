# FILENAME: main.py
# AUTHORS: Jalocha, Noga, Piekarski, Tancula

from create_dataset import CreateDataset
import cv2


def main():

    base_image_dir = 'input_data'
    dataset = CreateDataset(base_image_dir)
    dataset.read_data()
    # print(f'Test data: {dataset.test_data}')
    # print(f'Test data: {dataset.train_dataset}')
    cv2.imshow(dataset.test_data[0][1], dataset.test_data[0][0])
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
