from PIL import Image
import urllib.request
import numpy as np


def bbox_converter_(obj, converter, tensor_name, context, generate_labels):
    ds = context["ds"]
    try:
        ds.create_tensor(
            tensor_name,
            htype="bbox",
            dtype="int32",
            coords={"type": "pixel", "mode": "LTWH"},
        )
    except:
        pass

    if generate_labels:
        print("bbox converter does not support generating labels")

    converter.register_feature_id_for_kind("tool", "bounding_box", obj, tensor_name)

    def bbox_converter(row, obj):
        if tensor_name not in converter.values_cache:
            converter.values_cache[tensor_name] = dict()
        if row not in converter.values_cache[tensor_name]:
            converter.values_cache[tensor_name][row] = []

        converter.values_cache[tensor_name][row].append(
            [
                int(v)
                for v in [
                    obj["bounding_box"]["left"],
                    obj["bounding_box"]["top"],
                    obj["bounding_box"]["width"],
                    obj["bounding_box"]["height"],
                ]
            ]
        )
    converter.regsistered_actions[obj.feature_schema_id] = bbox_converter


def radio_converter_(obj, converter, tensor_name, context, generate_labels):
    ds = context["ds"]

    converter.label_mappings[tensor_name] = {
        options.value: i for i, options in enumerate(obj.options)
    }

    if generate_labels:
        print("radio converter does not support generating labels")

    try:
        ds.create_tensor(
            tensor_name,
            htype="class_label",
            class_names=list(converter.label_mappings[tensor_name].keys()),
            chunk_compression="lz4",
        )
    except:
        pass

    converter.register_feature_id_for_kind(
        "annotation", "radio_answer", obj, tensor_name
    )

    def radio_converter(row, o):
        if tensor_name not in converter.values_cache:
            converter.values_cache[tensor_name] = dict()
        if row not in converter.values_cache[tensor_name]:
            converter.values_cache[tensor_name][row] = []
        converter.values_cache[tensor_name][row] = [converter.label_mappings[tensor_name][o["value"]]]

    for option in obj.options:
        converter.regsistered_actions[option.feature_schema_id] = radio_converter

    def radio_converter_nested(row, obj):
        radio_converter(row, obj["radio_answer"])

    converter.regsistered_actions[obj.feature_schema_id] = radio_converter_nested


def checkbox_converter_(obj, converter, tensor_name, context, generate_labels):
    ds = context["ds"]

    converter.label_mappings[tensor_name] = {
        options.value: i for i, options in enumerate(obj.options)
    }

    if generate_labels:
        print("checkbox converter does not support generating labels")

    try:
        ds.create_tensor(
            tensor_name,
            htype="class_label",
            class_names=list(converter.label_mappings[tensor_name].keys()),
            chunk_compression="lz4",
        )
    except:
        pass

    converter.register_feature_id_for_kind(
        "annotation", "checklist_answers", obj, tensor_name
    )

    def checkbox_converter(row, obj):
        if tensor_name not in converter.values_cache:
            converter.values_cache[tensor_name] = dict()
        if row not in converter.values_cache[tensor_name]:
            converter.values_cache[tensor_name][row] = []

        converter.values_cache[tensor_name][row].append(converter.label_mappings[tensor_name][obj["value"]])

    for option in obj.options:
        converter.regsistered_actions[option.feature_schema_id] = checkbox_converter

    def checkbox_converter_nested(row, obj):
        for o in obj["checklist_answers"]:
            checkbox_converter(row, o)

    converter.regsistered_actions[obj.feature_schema_id] = checkbox_converter_nested


def point_converter_(obj, converter, tensor_name, context, generate_labels):
    ds = context["ds"]
    try:
        ds.create_tensor(tensor_name, htype="point", dtype="int32")
    except:
        pass

    converter.register_feature_id_for_kind("annotation", "point", obj, tensor_name)

    if generate_labels:
        print("point converter does not support generating labels")

    def point_converter(row, obj):
        if tensor_name not in converter.values_cache:
            converter.values_cache[tensor_name] = dict()
        if row not in converter.values_cache[tensor_name]:
            converter.values_cache[tensor_name][row] = []
        
        converter.values_cache[tensor_name][row].append([int(obj["point"]["x"]), int(obj["point"]["y"])])

    converter.regsistered_actions[obj.feature_schema_id] = point_converter


def line_converter_(obj, converter, tensor_name, context, generate_labels):
    ds = context["ds"]
    try:
        ds.create_tensor(tensor_name, htype="polygon", dtype="int32")
    except:
        pass

    converter.register_feature_id_for_kind("annotation", "line", obj, tensor_name)

    if generate_labels:
        print("line converter does not support generating labels")

    def polygon_converter(row, obj):
        if tensor_name not in converter.values_cache:
            converter.values_cache[tensor_name] = dict()
        if row not in converter.values_cache[tensor_name]:
            converter.values_cache[tensor_name][row] = []
        
        converter.values_cache[tensor_name][row].append([[int(l["x"]), int(l["y"])] for l in obj["line"]])

    converter.regsistered_actions[obj.feature_schema_id] = polygon_converter


def raster_segmentation_converter_(
    obj, converter, tensor_name, context, generate_labels
):
    ds = context["ds"]
    try:
        ds.create_tensor(
            tensor_name, htype="binary_mask", dtype="bool", sample_compression="lz4"
        )
        if generate_labels:
            ds.create_tensor(
                f"{tensor_name}_labels",
                htype="class_label",
                dtype="int32",
                class_names=[],
                chunk_compression="lz4",
            )
            converter.label_mappings[f"{tensor_name}_labels"] = dict()
    except:
        pass

    converter.register_feature_id_for_kind(
        "annotation", "raster-segmentation", obj, tensor_name
    )

    tool_name = obj.name

    def mask_converter(row, obj):
        try:
            r = urllib.request.Request(
                obj["mask"]["url"],
                headers={"Authorization": f'Bearer {context["lb_api_key"]}'},
            )
            with urllib.request.urlopen(r) as response:
                if generate_labels:
                    if (
                        tool_name
                        not in converter.label_mappings[f"{tensor_name}_labels"]
                    ):
                        converter.label_mappings[f"{tensor_name}_labels"][tool_name] = (
                            len(converter.label_mappings[f"{tensor_name}_labels"])
                        )
                        ds[f"{tensor_name}_labels"].info.update(
                            class_names=list(
                                converter.label_mappings[f"{tensor_name}_labels"].keys()
                            )
                        )
                    val = []
                    try:
                        val = (
                            ds[f"{tensor_name}_labels"][row].numpy(aslist=True).tolist()
                        )
                    except (KeyError, IndexError):
                        pass

                    val.append(
                        converter.label_mappings[f"{tensor_name}_labels"][tool_name]
                    )
                    ds[f"{tensor_name}_labels"][row] = val

                mask = np.array(Image.open(response)).astype(np.bool_)
                mask = mask[..., np.newaxis]
                try:
                    if generate_labels:
                        val = ds[tensor_name][row].numpy()
                        labels = ds[f"{tensor_name}_labels"].info['class_names']
                        if len(labels) != val.shape[-1]:
                            val = np.concatenate([ds[tensor_name][row].numpy(), np.zeros_like(mask)], axis=-1)
                        idx = labels.index(tool_name)
                        val[:,:,idx] = np.logical_or(val[:,:,idx], mask[:,:,0])
                    else:
                        val = np.logical_or(ds[tensor_name][row].numpy(), mask)
                except (KeyError, IndexError):
                    val = mask

                ds[tensor_name][row] = val
        except Exception as e:
            print(f"Error downloading mask: {e}")

    converter.regsistered_actions[obj.feature_schema_id] = mask_converter


def text_converter_(obj, converter, tensor_name, context, generate_labels):
    ds = context["ds"]
    try:
        ds.create_tensor(tensor_name, htype="text", dtype="str")
    except:
        pass

    converter.register_feature_id_for_kind("annotation", "text", obj, tensor_name)

    if generate_labels:
        print("text converter does not support generating labels")

    def text_converter(row, obj):
        if tensor_name not in converter.values_cache:
            converter.values_cache[tensor_name] = dict()
        if row not in converter.values_cache[tensor_name]:
            converter.values_cache[tensor_name][row] = []
        converter.values_cache[tensor_name][row] = obj["text_answer"]["content"]

    converter.regsistered_actions[obj.feature_schema_id] = text_converter
