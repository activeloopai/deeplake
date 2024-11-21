
class labelbox_type_converter:
    def __init__(self, ontology, converters, project, project_id, dataset, context):
        self.labelbox_feature_id_to_type_mapping = dict()
        self.regsistered_actions = dict()
        self.label_mappings = dict()
        self.registered_interpolators = dict()

        self.project = self.fix_project_order_(project, dataset)
        self.project_id = project_id
        self.dataset = dataset

        self.labelbox_type_converters_ = converters
        self.key_frames_to_ignore_ = ['checklist_answers', 'radio_answers', 'bounding_box']
        
        self.register_ontology_(ontology, context)
        
    def register_feature_id_for_kind(self, kind, key, obj):
        self.labelbox_feature_id_to_type_mapping[obj.feature_schema_id] = {
            'kind': kind,
            'key': key,
            'name': obj.name
        }

    def dataset_with_applied_annotations(self):
        print("start parsing annotations")
        idx_offset = 0
        for p in self.project:
            for lbl_idx, labels in enumerate(p["projects"][self.project_id]["labels"]):
                print("parse project with video url", p["data_row"]["external_id"], "label idx", lbl_idx)
                segments = labels["annotations"]["segments"]
                frames = labels["annotations"]["frames"]
                key_frame_feature_map = labels["annotations"]["key_frame_feature_map"]

                for feature_id, ranges in segments.items():
                    for r in ranges:
                        self.process_range_(r[0], r[1], idx_offset, frames, feature_id)

                for feature_id, indices in key_frame_feature_map.items():
                    for i in indices:
                        self.process_key_(str(i), idx_offset + i, frames, feature_id, {}, True)

            idx_offset += p['media_attributes']['frame_count']

        return self.dataset

    def register_tool_(self, tool, context):
        if tool.tool.value not in self.labelbox_type_converters_:
            print('skip tool:', tool.tool.value)
            return
        self.labelbox_type_converters_[tool.tool.value](tool, self, context)

    def register_classification_(self, tool, context):
        if tool.class_type.value not in self.labelbox_type_converters_:
            print('skip classification:', tool.class_type.value)
            return
        self.labelbox_type_converters_[tool.class_type.value](tool, self, context)

    def register_ontology_(self, ontology, context):
        for tool in ontology.tools():
            self.register_tool_(tool, context)

        for classification in ontology.classifications():
            if classification.scope.value != 'index':
                print('skip global classification:', classification.name)
                continue
            self.register_classification_(classification, context)

    def find_annotation_json_(self, feature_id, frame, is_key_frame):
        if 'objects' in frame:
            if feature_id in frame['objects']:
                return frame['objects'][feature_id], False
            
        if 'classifications' in frame:
            for c in frame['classifications']:
                if c['feature_id'] == feature_id:
                    if is_key_frame and self.labelbox_feature_id_to_type_mapping[c['feature_schema_id']]['key'] in self.key_frames_to_ignore_:
                        return None, True
                    return c, False
                nested_key = self.labelbox_feature_id_to_type_mapping[c['feature_schema_id']]['key']
                if nested_key in c:
                    if isinstance(c[nested_key], list):
                        for i in c[nested_key]:
                            if i['feature_id'] == feature_id:
                                return i, False
                    else:
                        if c[nested_key]['feature_id'] == feature_id:
                            return c[nested_key], False
                        
        return None, False
    
    def process_key_(self, key, i, frames, feature_id, last_objects_cache, is_key_frame):
        if key in frames:
            obj, skip = self.find_annotation_json_(feature_id, frames[key], is_key_frame)
            if skip:
                assert (is_key_frame)
                return
            if obj is not None:
                last_objects_cache[feature_id] = obj
        assert(feature_id in last_objects_cache)
        obj = last_objects_cache[feature_id]
        assert(obj is not None)
        self.regsistered_actions[obj['feature_schema_id']](i - 1, obj)

    def process_range_(self, start, end, offset, frames, feature_id):
        last_objects = {}
        for i in range(start, end + 1):
            self.process_key_(str(i), offset + i, frames, feature_id, last_objects, False)

    def fix_project_order_(self, project_j, ds):
        return sorted(project_j, key=lambda x: ds.info['labelbox_video_sources'].index(x["data_row"]["external_id"]))
