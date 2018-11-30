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
    dataset.crop_face()
    cv2.imshow(dataset.train_dataset[5][1], dataset.train_dataset[5][0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
