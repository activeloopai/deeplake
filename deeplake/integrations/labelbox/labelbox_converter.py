class labelbox_type_converter:
    def __init__(self, ontology, converters, project, project_id, dataset, context):
        self.labelbox_feature_id_to_type_mapping = dict()
        self.regsistered_actions = dict()
        self.label_mappings = dict()
        self.registered_interpolators = dict()

        self.project = project
        self.project_id = project_id
        self.dataset = dataset

        self.labelbox_type_converters_ = converters
        
        self.register_ontology_(ontology, context)
        
    def register_feature_id_for_kind(self, kind, key, obj, tensor_name):
        self.labelbox_feature_id_to_type_mapping[obj.feature_schema_id] = {
            'kind': kind,
            'key': key,
            'name': obj.name,
            'tensor_name': tensor_name
        }

    def dataset_with_applied_annotations(self):
        idx_offset = 0
        for p in self.fixed_project_order_(self.project, self.dataset):
            print("parse project with video url", p["data_row"]["external_id"])
            if not 'labels' in p["projects"][self.project_id]:
                continue
            for lbl_idx, labels in enumerate(p["projects"][self.project_id]["labels"]):
                if 'frames' not in labels["annotations"]:
                    continue
                frames = labels["annotations"]["frames"]
                if not len(frames):
                    print('skip project:', p["data_row"]["external_id"], 'with label idx', lbl_idx, 'as it has no frames')
                    continue

                assert(len(frames) == p['media_attributes']['frame_count'])

                for i in range(p['media_attributes']['frame_count']):
                    if str(i + 1) not in frames:
                        print('skip frame:', i + 1)
                    self.parse_frame_(frames[str(i + 1)], idx_offset + i)

                if 'segments' not in labels["annotations"]:
                    continue
                segments = labels["annotations"]["segments"]
                # the frames contain only the interpolated values
                # iterate over segments and assign same value to all frames in the segment
                self.parse_segments_(segments, frames, idx_offset)

            idx_offset += p['media_attributes']['frame_count']

        return self.dataset

    def register_tool_(self, tool, context):
        if tool.tool.value not in self.labelbox_type_converters_:
            print('skip tool:', tool.tool.value)
            return
        
        should_group_with_classifications = len(tool.classifications) > 0
        self.labelbox_type_converters_[tool.tool.value](tool, self, tool.name + "/" + tool.name if should_group_with_classifications else tool.name, context)

        for classification in tool.classifications:
            self.register_classification_(classification, context, parent=tool.name)


    def register_classification_(self, tool, context, parent=''):
        if tool.class_type.value not in self.labelbox_type_converters_:
            return
        
        tool_name = parent + '/' + tool.name if len(parent) else tool.name
        self.labelbox_type_converters_[tool.class_type.value](tool, self, tool_name, context)


    def register_ontology_(self, ontology, context):
        for tool in ontology.tools():
            self.register_tool_(tool, context)

        for classification in ontology.classifications():
            if classification.scope.value != 'index':
                print('skip global classification:', classification.name)
                continue
            self.register_classification_(classification, context)


    def parse_frame_(self, frame, idx):
        if 'objects' in frame:
            for _, obj  in frame['objects'].items():
                self.parse_object_(obj, idx)

        if 'classifications' in frame:
            for obj in frame['classifications']:
                self.parse_classification_(obj, idx)

    def parse_object_(self, obj, idx):
        if obj['feature_schema_id'] not in self.regsistered_actions:
            print('skip object:', obj['feature_schema_id'])
            return

        self.regsistered_actions[obj['feature_schema_id']](idx, obj)

        if 'classifications' in obj:
            for obj in obj['classifications']:
                self.parse_classification_(obj, idx)

    def parse_classification_(self, obj, idx):
        if obj['feature_schema_id'] not in self.regsistered_actions:
            print('skip classification:', obj['feature_schema_id'])
            return

        self.regsistered_actions[obj['feature_schema_id']](idx, obj)

        if 'classifications' in obj:
            for obj in obj['classifications']:
                self.parse_classification_(obj, idx)

    def find_object_with_feature_id_(self, frame, feature_id):
        if isinstance(frame, list):
            for f in frame:
                if ret := self.find_object_with_feature_id_(f, feature_id):
                    return ret
        
        if 'objects' in frame:
            if feature_id in frame['objects']:
                return frame['objects'][feature_id]
            for _, obj in frame['objects'].items():
                if ret := self.find_object_with_feature_id_(obj, feature_id):
                    return ret
                
        if 'classifications' in frame:
            for obj in frame['classifications']:
                if ret := self.find_object_with_feature_id_(obj, feature_id):
                    return ret
                k = self.labelbox_feature_id_to_type_mapping[obj['feature_schema_id']]['key']
                if k in obj:
                    if ret := self.find_object_with_feature_id_(obj[k], feature_id):
                        return ret
                        
        if 'feature_id' in frame and frame['feature_id'] == feature_id:
            return frame

        return None

    def parse_segments_(self, segments, frames, offset):
        for feature_id, ranges in segments.items():
            for r in ranges:
                obj = self.find_object_with_feature_id_(frames[str(r[0])], feature_id)
                assert(obj is not None)
                for i in range(r[0] + 1, r[1]):
                    new_obj = self.find_object_with_feature_id_(frames[str(i)], feature_id)
                    if new_obj:
                        obj = new_obj
                        continue
                    # update the frame if the object was not present in the frame
                    self.regsistered_actions[obj['feature_schema_id']](offset + i - 1, obj)

    def fixed_project_order_(self, project_j, ds):
        order = [ds.info['labelbox_video_sources'].index(x["data_row"]["external_id"]) for x in project_j]
        for idx in order:
            yield project_j[idx]
