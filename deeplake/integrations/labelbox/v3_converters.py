
from PIL import Image
import urllib.request
import numpy as np

def bbox_converter_(obj, converter, context):
    ds = context['ds']
    obj_name = obj.name
    try:
        ds.create_tensor(obj.name, htype='bbox', dtype='int32', coords={"type": "pixel", "mode": "LTWH"})
    except:
        pass

    converter.register_feature_id_for_kind('tool', 'bounding_box', obj)

    def bbox_converter(row, obj):
        vals = []
        try:
            vals = ds[obj_name][row].numpy(aslist=True).tolist()
        except (KeyError, IndexError):
            pass

        vals.append([int(v) for v in [obj['bounding_box']['left'], obj['bounding_box']['top'], obj['bounding_box']['width'], obj['bounding_box']['height']]])
        ds[obj_name][row] = vals
    converter.regsistered_actions[obj.feature_schema_id] = bbox_converter

def radio_converter_(obj, converter, context):
    ds = context['ds']

    obj_name = obj.name
    converter.label_mappings[obj_name] = {options.value: i for i, options in enumerate(obj.options)}

    try:
        ds.create_tensor(obj.name, htype='class_label', class_names=list(converter.label_mappings[obj_name].keys()), chunk_compression="lz4")
    except:
        pass

    converter.register_feature_id_for_kind('annotation', 'radio_answer', obj)

    def radio_converter(row, o):
        ds[obj_name][row] = converter.label_mappings[obj_name][o['value']]

    for option in obj.options:
        converter.regsistered_actions[option.feature_schema_id] = radio_converter

    def radio_converter_nested(row, obj):
        radio_converter(row, obj['radio_answer'])
    converter.regsistered_actions[obj.feature_schema_id] = radio_converter_nested


def checkbox_converter_(obj, converter, context):
    ds = context['ds']
    obj_name = obj.name
    converter.label_mappings[obj_name] = {options.value: i for i, options in enumerate(obj.options)}

    try:
        ds.create_tensor(obj.name, htype='class_label', class_names=list(converter.label_mappings[obj_name].keys()), chunk_compression="lz4")
    except:
        pass

    converter.register_feature_id_for_kind('annotation', 'checklist_answers', obj)

    def checkbox_converter(row, obj):
        vals = []
        try:
            vals = ds[obj_name][row].numpy(aslist=True).tolist()
        except (KeyError, IndexError):
            pass
        vals.append(converter.label_mappings[obj_name][obj['value']])

        ds[obj_name][row] = vals

    for option in obj.options:
        converter.regsistered_actions[option.feature_schema_id] = checkbox_converter

    def checkbox_converter_nested(row, obj):
        for o in obj['checklist_answers']:
            checkbox_converter(row, o)
    converter.regsistered_actions[obj.feature_schema_id] = checkbox_converter_nested


def point_converter_(obj, converter, context):
    ds = context['ds']
    obj_name = obj.name
    try:
      ds.create_tensor(obj.name, htype='point', dtype='int32')
    except:
      pass

    converter.register_feature_id_for_kind('annotation', 'point', obj)

    def point_converter(row, obj):
        vals = []
        try:
            vals = ds[obj_name][row].numpy(aslist=True).tolist()
        except (KeyError, IndexError):
            pass
        vals.append([int(obj['point']['x']), int(obj['point']['y'])])
        ds[obj_name][row] = vals
    converter.regsistered_actions[obj.feature_schema_id] = point_converter


def line_converter_(obj, converter, context):
    ds = context['ds']
    obj_name = obj.name
    try:
      ds.create_tensor(obj.name, htype='polygon', dtype='int32')
    except:
      pass

    converter.register_feature_id_for_kind('annotation', 'line', obj)

    def polygon_converter(row, obj):
        vals = []
        try:
            vals = ds[obj_name][row].numpy(aslist=True)
        except (KeyError, IndexError):
            pass
        vals.append([[int(l['x']), int(l['y'])] for l in obj['line']])
        ds[obj_name][row] = vals
    
    converter.regsistered_actions[obj.feature_schema_id] = polygon_converter

def raster_segmentation_converter_(obj, converter, context):
    ds = context['ds']
    obj_name = obj.name
    try:
        ds.create_tensor(obj.name, htype='segment_mask', dtype='uint8', sample_compression="lz4")
    except:
        pass

    converter.register_feature_id_for_kind('annotation', 'raster-segmentation', obj)

    def mask_converter(row, obj):
        try:
            r = urllib.request.Request(obj['mask']['url'], headers={'Authorization': f'Bearer {context["lb_api_key"]}'})
            with urllib.request.urlopen(r) as response:
                mask = np.array(Image.open(response)).astype(np.uint8)

                ds[obj_name][row] = mask[..., np.newaxis]
        except Exception as e:
            print(f"Error downloading mask: {e}")


    converter.regsistered_actions[obj.feature_schema_id] = mask_converter

def text_converter_(obj, converter, context):
    ds = context['ds']
    obj_name = obj.name
    try:
        ds.create_tensor(obj.name, htype='text', dtype='str')
    except:
        pass

    converter.register_feature_id_for_kind('annotation', 'text', obj)

    def text_converter(row, obj):
        ds[obj_name][row] = obj['text_answer']['content']
    converter.regsistered_actions[obj.feature_schema_id] = text_converter
