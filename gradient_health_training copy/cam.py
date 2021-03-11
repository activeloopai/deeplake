import cv2
import numpy as np
import os
import pandas as pd
from configparser import ConfigParser
from generator import AugmentedImageSequence
from models.keras import ModelFactory
from keras import backend as kb


def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer


def create_cam(df_g, output_dir, image_source_dir, model, generator, class_names):
    """
    Create a CAM overlay image for the input image

    :param df_g: pandas.DataFrame, bboxes on the same image
    :param output_dir: str
    :param image_source_dir: str
    :param model: keras model
    :param generator: generator.AugmentedImageSequence
    :param class_names: list of str
    """
    file_name = df_g["file_name"]
    print(f"process image: {file_name}")

    # draw bbox with labels
    img_ori = cv2.imread(filename=os.path.join(image_source_dir, file_name))

    label = df_g["label"]
    if label == "Infiltrate":
        label = "Infiltration"
    index = class_names.index(label)

    output_path = os.path.join(output_dir, f"{label}.{file_name}")

    img_transformed = generator.load_image(file_name)

    # CAM overlay
    # Get the 512 input weights to the softmax.
    class_weights = model.layers[-1].get_weights()[0]
    final_conv_layer = get_output_layer(model, "bn")
    get_output = kb.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
    [conv_outputs, predictions] = get_output([np.array([img_transformed])])
    conv_outputs = conv_outputs[0, :, :, :]

    # Create the class activation map.
    cam = np.zeros(dtype=np.float32, shape=(conv_outputs.shape[:2]))
    for i, w in enumerate(class_weights[index]):
        cam += w * conv_outputs[:, :, i]
    # print(f"predictions: {predictions}")
    cam /= np.max(cam)
    cam = cv2.resize(cam, img_ori.shape[:2])
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap[np.where(cam < 0.2)] = 0
    img = heatmap * 0.5 + img_ori

    # add label & rectangle
    # ratio = output dimension / 1024
    ratio = 1
    x1 = int(df_g["x"] * ratio)
    y1 = int(df_g["y"] * ratio)
    x2 = int((df_g["x"] + df_g["w"]) * ratio)
    y2 = int((df_g["y"] + df_g["h"]) * ratio)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(img, text=label, org=(5, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8, color=(0, 0, 255), thickness=1)
    cv2.imwrite(output_path, img)


def main():
    # parser config
    config_file = "./config.ini"
    cp = ConfigParser()
    cp.read(config_file)

    # default config
    output_dir = cp["DEFAULT"].get("output_dir")
    base_model_name = cp["DEFAULT"].get("base_model_name")
    class_names = cp["DEFAULT"].get("class_names").split(",")
    image_source_dir = cp["DEFAULT"].get("image_source_dir")
    image_dimension = cp["TRAIN"].getint("image_dimension")

    # parse weights file path
    output_weights_name = cp["TRAIN"].get("output_weights_name")
    weights_path = os.path.join(output_dir, output_weights_name)
    best_weights_path = os.path.join(output_dir, f"best_{output_weights_name}")

    # CAM config
    bbox_list_file = cp["CAM"].get("bbox_list_file")
    use_best_weights = cp["CAM"].getboolean("use_best_weights")

    print("** load model **")
    if use_best_weights:
        print("** use best weights **")
        model_weights_path = best_weights_path
    else:
        print("** use last weights **")
        model_weights_path = weights_path
    model_factory = ModelFactory()
    model = model_factory.get_model(
        class_names,
        model_name=base_model_name,
        use_base_weights=False,
        weights_path=model_weights_path)

    print("read bbox list file")
    df_images = pd.read_csv(bbox_list_file, header=None, skiprows=1)
    df_images.columns = ["file_name", "label", "x", "y", "w", "h"]

    print("create a generator for loading transformed images")
    cam_sequence = AugmentedImageSequence(
        dataset_csv_file=os.path.join(output_dir, "test.csv"),
        class_names=class_names,
        source_image_dir=image_source_dir,
        batch_size=1,
        target_size=(image_dimension, image_dimension),
        augmenter=None,
        steps=1,
        shuffle_on_epoch_end=False,
    )

    image_output_dir = os.path.join(output_dir, "cam")
    if not os.path.isdir(image_output_dir):
        os.makedirs(image_output_dir)

    print("create CAM")
    df_images.apply(
        lambda g: create_cam(
            df_g=g,
            output_dir=image_output_dir,
            image_source_dir=image_source_dir,
            model=model,
            generator=cam_sequence,
            class_names=class_names,
        ),
        axis=1,
    )


if __name__ == "__main__":
    main()
