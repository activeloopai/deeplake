from hub.util.exceptions import ModuleNotInstalledException


def dataset_to_tensorflow(dataset):
    """Converts the dataset into a tensorflow compatible format"""
    global tf
    try:
        import tensorflow as tf
    except ModuleNotFoundError:
        raise ModuleNotInstalledException(
            "'tensorflow' should be installed to convert the Dataset into tensorflow format"
        )

    def tf_gen():
        for index in range(len(dataset)):
            sample = {}
            for key in dataset.tensors:
                value = dataset[key][index].numpy()
                sample[key] = value
            yield sample

    def generate_signature():
        signature = {}
        for key in dataset.tensors:
            dtype = dataset[key].meta["dtype"]
            shape = dataset[key].shape
            dynamic_shape = [None for dim in shape]
            signature[key] = tf.RaggedTensorSpec(
                shape=dynamic_shape, dtype=tf.dtypes.DType(dtype)
            )
        return signature

    signature = generate_signature()
    return tf.data.Dataset.from_generator(tf_gen, output_signature=signature)
