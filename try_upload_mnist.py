import hub
from keras.datasets import mnist
from tqdm import tqdm

(x_train, y_train), (x_test, y_test) = mnist.load_data()

BASE_URL = "."


def main():
    ds = hub.Dataset(f"{BASE_URL}/mnist_hub")
    ds.delete()

    print(x_train.shape)
    print(y_train.shape)

    with ds:
        ds.create_tensor("image", htype="image")
        ds.create_tensor("label", htype="class_label")

        for x, y in tqdm(
            zip(x_train, y_train), desc="uploading mnist", total=x_train.shape[0]
        ):
            print(y, type(y))
            exit()
            ds.image.append(x)
            ds.label.append(y)


if __name__ == "__main__":
    main()
