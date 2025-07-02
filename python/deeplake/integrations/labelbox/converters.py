from PIL import Image
import urllib.request
import numpy as np
import copy
from deeplake.integrations.labelbox.deeplake_utils import *


def bbox_converter_(obj, converter, tensor_name, context, generate_labels):
    ds = context["ds"]
    try:
        ds.create_tensor(tensor_name, **bbox_tensor_create_kwargs_())
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

    def interpolator(start, end, progress):
        start_box = start["bounding_box"]
        end_box = end["bounding_box"]
        bbox = copy.deepcopy(start)
        bbox["bounding_box"] = {
            "top": start_box["top"] + (end_box["top"] - start_box["top"]) * progress,
            "left": start_box["left"]
            + (end_box["left"] - start_box["left"]) * progress,
            "width": start_box["width"]
            + (end_box["width"] - start_box["width"]) * progress,
            "height": start_box["height"]
            + (end_box["height"] - start_box["height"]) * progress,
        }

        return bbox

    converter.registered_interpolators[obj.feature_schema_id] = interpolator

def polygon_converter_(obj, converter, tensor_name, context, generate_labels):
    ds = context["ds"]
    try:
        ds.create_tensor(tensor_name, **polygon_tensor_create_kwargs_())
    except:
        pass

    if generate_labels:
        print("polygon converter does not support generating labels")

    converter.register_feature_id_for_kind("tool", "polygon", obj, tensor_name)

    def polygon_converter(row, obj):
        if tensor_name not in converter.values_cache:
            converter.values_cache[tensor_name] = dict()
        if row not in converter.values_cache[tensor_name]:
            converter.values_cache[tensor_name][row] = []
        polygon = obj["polygon"]
        if len(polygon) != 0 and not isinstance(polygon[0], dict):
            # if polygon is a list of points, convert it to a list of dicts
            polygon = [{"x": float(p[0]), "y": float(p[1])} for p in polygon]
        converter.values_cache[tensor_name][row].append(
            np.array([[float(p["x"]), float(p["y"])] for p in polygon])
        )

    converter.regsistered_actions[obj.feature_schema_id] = polygon_converter

    def interpolator(start, end, progress):
        start_polygon = start["polygon"]
        end_polygon = end["polygon"]
        polygon = copy.deepcopy(start)
        polygon["polygon"] = [
            [
                start_polygon[i]["x"]
                + (end_polygon[i]["x"] - start_polygon[i]["x"]) * progress,
                start_polygon[i]["y"]
                + (end_polygon[i]["y"] - start_polygon[i]["y"]) * progress,
            ]
            for i in range(len(start_polygon))
        ]

        return polygon

    converter.registered_interpolators[obj.feature_schema_id] = interpolator

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
            **class_label_tensor_create_kwargs_(),
        )
        ds[tensor_name].update_metadata(
            {"class_names": list(converter.label_mappings[tensor_name].keys())}
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
        converter.values_cache[tensor_name][row] = [
            converter.label_mappings[tensor_name][o["value"]]
        ]

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
            **class_label_tensor_create_kwargs_(),
        )
        ds[tensor_name].update_metadata(
            {"class_names": list(converter.label_mappings[tensor_name].keys())}
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

        converter.values_cache[tensor_name][row].append(
            converter.label_mappings[tensor_name][obj["value"]]
        )

    for option in obj.options:
        converter.regsistered_actions[option.feature_schema_id] = checkbox_converter

    def checkbox_converter_nested(row, obj):
        for o in obj["checklist_answers"]:
            checkbox_converter(row, o)

    converter.regsistered_actions[obj.feature_schema_id] = checkbox_converter_nested


def point_converter_(obj, converter, tensor_name, context, generate_labels):
    ds = context["ds"]
    try:
        ds.create_tensor(tensor_name, **point_tensor_create_kwargs_())
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

        converter.values_cache[tensor_name][row].append(
            [int(obj["point"]["x"]), int(obj["point"]["y"])]
        )

    converter.regsistered_actions[obj.feature_schema_id] = point_converter

    def interpolator(start, end, progress):
        start_point = start["point"]
        end_point = end["point"]
        point = copy.deepcopy(start)
        point["point"] = {
            "x": start_point["x"] + (end_point["x"] - start_point["x"]) * progress,
            "y": start_point["y"] + (end_point["y"] - start_point["y"]) * progress,
        }

        return point

    converter.registered_interpolators[obj.feature_schema_id] = interpolator


def line_converter_(obj, converter, tensor_name, context, generate_labels):
    ds = context["ds"]
    try:
        ds.create_tensor(tensor_name, **polygon_tensor_create_kwargs_())
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
        line = obj["line"]
        if len(line) != 0 and not isinstance(line[0], dict):
            # if line is a list of points, convert it to a list of dicts
            line = [{"x": int(l[0]), "y": int(l[1])} for l in line]
        converter.values_cache[tensor_name][row].append(
            [[int(l["x"]), int(l["y"])] for l in line]
        )

    converter.regsistered_actions[obj.feature_schema_id] = polygon_converter

    def interpolator(start, end, progress):
        start_line = start["line"]
        end_line = end["line"]
        line = copy.deepcopy(start)
        line["line"] = [
            [
                start_line[i]["x"] + (end_line[i]["x"] - start_line[i]["x"]) * progress,
                start_line[i]["y"] + (end_line[i]["y"] - start_line[i]["y"]) * progress,
            ]
            for i in range(len(start_line))
        ]

        return line

    converter.registered_interpolators[obj.feature_schema_id] = interpolator


def raster_segmentation_converter_(
    obj, converter, tensor_name, context, generate_labels
):
    ds = context["ds"]
    try:
        ds.create_tensor(tensor_name, **binary_mask_tensor_create_kwargs_())
    except:
        pass

    try:
        if generate_labels:
            ds.create_tensor(
                f"{tensor_name}_labels", **class_label_tensor_create_kwargs_()
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
                        ds[f"{tensor_name}_labels"].update_metadata(
                            {
                                "class_names": list(
                                    converter.label_mappings[
                                        f"{tensor_name}_labels"
                                    ].keys()
                                )
                            }
                        )
                    val = []
                    try:
                        val = ds[f"{tensor_name}_labels"].value(row, aslist=True)
                    except (KeyError, IndexError):
                        pass
                    val.append(
                        converter.label_mappings[f"{tensor_name}_labels"][tool_name]
                    )
                    ds[f"{tensor_name}_labels"].set_value(row, val)

                mask = np.array(Image.open(response)).astype(np.bool_)
                mask = mask[..., np.newaxis]
                try:
                    if generate_labels:
                        val = ds[tensor_name].value(row)
                        labels = ds[f"{tensor_name}_labels"].info["class_names"]
                        if val is None:
                            raise IndexError()
                        if len(labels) != val.shape[-1]:
                            val = np.concatenate(
                                [val, np.zeros_like(mask)],
                                axis=-1,
                            )
                        idx = labels.index(tool_name)
                        val[:, :, idx] = np.logical_or(val[:, :, idx], mask[:, :, 0])
                    else:
                        val = np.logical_or(ds[tensor_name].value(row), mask)
                except (KeyError, IndexError):
                    val = mask

                ds[tensor_name].set_value(row, val)
        except Exception as e:
            print(f"Error downloading mask: {e}")

    converter.regsistered_actions[obj.feature_schema_id] = mask_converter


def text_converter_(obj, converter, tensor_name, context, generate_labels):
    ds = context["ds"]
    try:
        ds.create_tensor(tensor_name, **text_tensor_create_kwargs_())
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
