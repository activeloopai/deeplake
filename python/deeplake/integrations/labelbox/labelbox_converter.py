from deeplake.integrations.labelbox.labelbox_utils import *
from collections import defaultdict
from deeplake.integrations.labelbox.deeplake_utils import dataset_wrapper

try:
    from tqdm import tqdm as progress_bar
except ImportError:

    def progress_bar(iterable, *args, **kwargs):
        return iterable


class labelbox_type_converter:
    def __init__(
        self,
        ontology,
        converters,
        project,
        project_id,
        dataset,
        context,
        metadata_generators=None,
        group_mapping=None,
    ):
        self.labelbox_feature_id_to_type_mapping = dict()
        self.regsistered_actions = dict()
        self.label_mappings = dict()
        self.values_cache = dict()
        self.registered_interpolators = dict()

        self.metadata_generators_ = metadata_generators

        self.project = project
        self.project_id = project_id

        self.dataset = dataset_wrapper(dataset)

        self.group_mapping = group_mapping if group_mapping is not None else dict()
        self.groupped_tensor_overrides = dict()

        self.labelbox_type_converters_ = converters

        self.register_ontology_(ontology, context)

    def register_feature_id_for_kind(self, kind, key, obj, tensor_name):
        self.labelbox_feature_id_to_type_mapping[obj.feature_schema_id] = {
            "kind": kind,
            "key": key,
            "name": obj.name,
            "tensor_name": tensor_name,
        }

    def dataset_with_applied_annotations(self):
        idx_offset = 0
        print("total annotations projects count: ", len(self.project))

        if self.metadata_generators_:
            self.generate_metadata_tensors_(self.metadata_generators_, self.dataset)

        for p_idx, p in enumerate(self.yield_projects_(self.project, self.dataset)):
            if "labels" not in p["projects"][self.project_id]:
                print("no labels for project with index: ", p_idx)
                continue
            print("parsing annotations for project with index: ", p_idx)
            for lbl_idx, labels in enumerate(p["projects"][self.project_id]["labels"]):
                self.values_cache = dict()
                if "frames" not in labels["annotations"]:
                    continue
                frames = labels["annotations"]["frames"]
                if not len(frames):
                    print(
                        "skip",
                        external_url_from_media_project_(p),
                        "with label idx",
                        lbl_idx,
                        "as it has no frames",
                    )
                    continue

                assert len(frames) <= p["media_attributes"]["frame_count"]

                print("parsing frames for label index: ", lbl_idx)
                for i in progress_bar(range(p["media_attributes"]["frame_count"])):
                    if str(i + 1) not in frames:
                        continue
                    self.parse_frame_(frames[str(i + 1)], idx_offset + i)

                if "segments" not in labels["annotations"]:
                    continue
                segments = labels["annotations"]["segments"]
                key_frame_feature_map = labels["annotations"]["key_frame_feature_map"]
                # the frames contain only the interpolated values
                # iterate over segments and assign same value to all frames in the segment
                self.parse_segments_(
                    segments, frames, key_frame_feature_map, idx_offset
                )

                self.apply_cached_values_(self.values_cache, idx_offset)
                if self.metadata_generators_:
                    print("filling metadata for project with index: ", p_idx)
                    self.fill_metadata_(
                        self.metadata_generators_,
                        self.dataset,
                        p,
                        self.project_id,
                        p["media_attributes"]["frame_count"],
                        idx_offset,
                    )

            idx_offset += p["media_attributes"]["frame_count"]

        self.pad_all_tensors(self.dataset)

        return self.dataset.ds

    def register_tool_(self, tool, context, fix_grouping_only):
        if tool.tool.value not in self.labelbox_type_converters_:
            print("skip tool:", tool.tool.value)
            return

        prefered_name = tool.name

        if tool.tool.value in self.group_mapping:
            prefered_name = self.group_mapping[tool.tool.value]
        else:
            prefered_name = tool.name

        should_group_with_classifications = len(tool.classifications) > 0
        if should_group_with_classifications:
            tool_name = prefered_name + "/" + prefered_name
            if fix_grouping_only:
                if tool.tool.value in self.group_mapping:
                    self.groupped_tensor_overrides[tool.tool.value] = tool_name
        else:
            tool_name = prefered_name

        for classification in tool.classifications:
            self.register_classification_(
                classification,
                context,
                fix_grouping_only=fix_grouping_only,
                parent=prefered_name,
            )

        if fix_grouping_only:
            return

        if tool.tool.value in self.groupped_tensor_overrides:
            tool_name = self.groupped_tensor_overrides[tool.tool.value]

        self.labelbox_type_converters_[tool.tool.value](
            tool, self, tool_name, context, tool.tool.value in self.group_mapping
        )

    def register_classification_(self, tool, context, fix_grouping_only, parent=""):
        if tool.class_type.value not in self.labelbox_type_converters_:
            return

        if tool.class_type.value in self.group_mapping:
            prefered_name = (parent + "/" if parent else "") + self.group_mapping[
                tool.class_type.value
            ]
        else:
            prefered_name = (parent + "/" if parent else "") + tool.name

        if fix_grouping_only:
            return

        self.labelbox_type_converters_[tool.class_type.value](
            tool,
            self,
            prefered_name,
            context,
            tool.class_type.value in self.group_mapping,
        )

    def register_ontology_(self, ontology, context, fix_grouping_only=True):
        for tool in ontology.tools():
            self.register_tool_(tool, context, fix_grouping_only=fix_grouping_only)

        for classification in ontology.classifications():
            if classification.scope.value != "index":
                print("skip global classification:", classification.name)
                continue
            self.register_classification_(
                classification, context, fix_grouping_only=fix_grouping_only
            )

        if fix_grouping_only:
            self.register_ontology_(ontology, context, fix_grouping_only=False)

    def parse_frame_(self, frame, idx):
        if "objects" in frame:
            for _, obj in frame["objects"].items():
                self.parse_object_(obj, idx)

        for obj in frame.get("classifications", []):
            self.parse_classification_(obj, idx)

    def parse_object_(self, obj, idx):
        if obj["feature_schema_id"] not in self.regsistered_actions:
            print("skip object:", obj["feature_schema_id"])
            return

        self.regsistered_actions[obj["feature_schema_id"]](idx, obj)

        for obj in obj.get("classifications", []):
            self.parse_classification_(obj, idx)

    def parse_classification_(self, obj, idx):
        if obj["feature_schema_id"] not in self.regsistered_actions:
            print("skip classification:", obj["feature_schema_id"])
            return

        self.regsistered_actions[obj["feature_schema_id"]](idx, obj)

        for obj in obj.get("classifications", []):
            self.parse_classification_(obj, idx)

    def find_object_with_feature_id_(self, frame, feature_id):
        if isinstance(frame, list):
            for f in frame:
                if ret := self.find_object_with_feature_id_(f, feature_id):
                    return ret

        if "objects" in frame:
            if feature_id in frame["objects"]:
                return frame["objects"][feature_id]
            for _, obj in frame["objects"].items():
                if ret := self.find_object_with_feature_id_(obj, feature_id):
                    return ret

        if "classifications" in frame:
            for obj in frame["classifications"]:
                if ret := self.find_object_with_feature_id_(obj, feature_id):
                    return ret
                k = self.labelbox_feature_id_to_type_mapping[obj["feature_schema_id"]][
                    "key"
                ]
                if k in obj:
                    if ret := self.find_object_with_feature_id_(obj[k], feature_id):
                        return ret

        if "feature_id" in frame and frame["feature_id"] == feature_id:
            return frame

        return None

    def existing_sub_ranges_(self, frames, r, key_frames):
        end = r[1]
        sub_ranges = [(r[0], end)]
        for i in key_frames:
            if str(i) not in frames:
                continue
            if i <= r[0] or i >= end:
                continue
            sub_ranges[-1] = (sub_ranges[-1][0], i)
            sub_ranges.append((i, end))
        return sub_ranges

    def _interpolate_frames(self, start, end, st, en, frames, feature_id, offset):
        """Handle interpolation for frames between start and end."""
        for i in range(st + 1, en + 1):
            # skip if the frame already has the object
            if (
                str(i) in frames
                and self.find_object_with_feature_id_(frames[str(i)], feature_id)
                is not None
            ):
                continue

            if start["feature_schema_id"] in self.registered_interpolators:
                obj = self.registered_interpolators[start["feature_schema_id"]](
                    start, end, (i - st) / (en - st)
                )
            else:
                obj = end

            self.regsistered_actions[obj["feature_schema_id"]](offset + i - 1, obj)
            # nested classifications are not in the segments
            for o in obj.get("classifications", []):
                self.regsistered_actions[o["feature_schema_id"]](offset + i - 1, o)

    def _process_sub_ranges(self, sub_ranges, frames, feature_id, offset):
        """Process individual sub-ranges for a given feature_id."""
        for st, en in sub_ranges:
            if not st or str(st) not in frames:
                print(
                    f"Warning: Could not find start object with feature_id {feature_id} in frame {st}"
                )
                continue
            start = self.find_object_with_feature_id_(frames[str(st)], feature_id)
            if str(en) in frames:
                end = self.find_object_with_feature_id_(frames[str(en)], feature_id)
            else:
                end = start

            if not start:
                print(
                    f"Warning: Could not find start object with feature_id {feature_id} in frame {st}"
                )
                continue
            if not end:
                print(
                    f"Warning: Could not find end object with feature_id {feature_id} in frame {en}"
                )
                continue
            if start["feature_schema_id"] != end["feature_schema_id"]:
                print(
                    f"Warning: Feature schema ID mismatch between start ({start['feature_schema_id']}) and end ({end['feature_schema_id']})"
                )
                continue

            self._interpolate_frames(start, end, st, en, frames, feature_id, offset)

    def parse_segments_(self, segments, frames, key_frame_feature_map, offset):
        print("total segments count to parse:", len(segments))
        for feature_id, ranges in segments.items():
            print("parsing segments with feature id: ", feature_id)
            for r in progress_bar(ranges):
                sub_ranges = self.existing_sub_ranges_(
                    frames, r, key_frame_feature_map[feature_id]
                )
                self._process_sub_ranges(sub_ranges, frames, feature_id, offset)

    def apply_cached_values_(self, cache, offset):
        print("applying cached values")
        for tensor_name, row_map in cache.items():
            print("applying cached values for tensor: ", tensor_name)
            max_val = max(row_map.keys()) - offset
            values = []
            for i in progress_bar(range(max_val + 1)):
                key = i + offset
                if key in row_map:
                    values.append(row_map[key])
                else:
                    values.append(None)

            self.dataset.fill_data(tensor_name, values, offset)

    def yield_projects_(self, project_j, ds):
        raise NotImplementedError("fixed_project_order_ is not implemented")

    def generate_metadata_tensors_(self, generators, ds):
        for tensor_name, v in generators.items():
            try:
                self.dataset.add_column(tensor_name, **v["create_tensor_kwargs"])
            except:
                pass

    def fill_metadata_(
        self, generators, dataset, project, project_id, frames_count, offset
    ):
        metadata_dict = defaultdict(list)
        context = {"project_id": project_id}
        for tensor_name, v in generators.items():
            for i in range(frames_count):
                context["frame_idx"] = i
                metadata_dict[tensor_name].append(v["generator"](project, context))

        for tensor_name, values in metadata_dict.items():
            dataset.fill_data(tensor_name, values, offset)

    def pad_all_tensors(self, dataset):
        dataset.pad_all_tensors()


# if changes are made to the labelbox_video_converter class, check if ontology_for_debug works correctly
class labelbox_video_converter(labelbox_type_converter):
    def __init__(
        self,
        ontology,
        converters,
        project,
        project_id,
        dataset,
        context,
        metadata_generators=None,
        group_mapping=None,
    ):
        super().__init__(
            ontology,
            converters,
            project,
            project_id,
            dataset,
            context,
            metadata_generators,
            group_mapping,
        )

    def yield_projects_(self, project_j, ds):
        if "labelbox_meta" not in ds.metadata:
            raise ValueError("No labelbox meta data in dataset")
        info = ds.metadata["labelbox_meta"]

        def sorter(p):
            url = external_url_from_media_project_(p)
            return info["sources"].index(url)

        ordered_values = sorted(project_j, key=sorter)
        for p in ordered_values:
            yield p


class labelbox_image_converter(labelbox_type_converter):
    def __init__(
        self,
        ontology,
        converters,
        project,
        project_id,
        dataset,
        context,
        metadata_generators=None,
        group_mapping=None,
    ):
        super().__init__(
            ontology,
            converters,
            project,
            project_id,
            dataset,
            context,
            metadata_generators,
            group_mapping,
        )

    def yield_projects_(self, project_j, ds):
        if "labelbox_meta" not in ds.metadata:
            raise ValueError("No labelbox meta data in dataset")
        info = ds.metadata["labelbox_meta"]

        def sorter(p):
            url = external_url_from_media_project_(p)
            return info["sources"].index(url)

        ordered_values = sorted(project_j, key=sorter)
        for p in ordered_values:
            yield p

    def dataset_with_applied_annotations(self):
        idx_offset = 0
        print("total annotations projects count: ", len(self.project))

        if self.metadata_generators_:
            self.generate_metadata_tensors_(self.metadata_generators_, self.dataset)
        for p_idx, p in enumerate(self.yield_projects_(self.project, self.dataset)):
            if "labels" not in p["projects"][self.project_id]:
                print("no labels for project with index: ", p_idx)
                continue
            print("parsing annotations for project with index: ", p_idx)
            for lbl_idx, labels in enumerate(p["projects"][self.project_id]["labels"]):
                self.values_cache = dict()
                if "objects" not in labels["annotations"]:
                    continue
                objects = labels["annotations"]["objects"]
                if not len(objects):
                    print(
                        "skip",
                        external_url_from_media_project_(p),
                        "with label idx",
                        lbl_idx,
                        "as it has no objects",
                    )
                    continue

                print("parsing objects for label index: ", lbl_idx)
                for obj in progress_bar(objects):
                    self.parse_object_(obj, idx_offset)

            self.apply_cached_values_(self.values_cache, idx_offset)
            if self.metadata_generators_:
                print("filling metadata for project with index: ", p_idx)
                self.fill_metadata_(
                    self.metadata_generators_,
                    self.dataset,
                    p,
                    self.project_id,
                    1,
                    idx_offset,
                )

            idx_offset += 1

        self.pad_all_tensors(self.dataset)

        return self.dataset.ds
