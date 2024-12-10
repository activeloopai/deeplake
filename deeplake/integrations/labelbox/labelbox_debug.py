import json
from deeplake.integrations.labelbox.labelbox_converter import labelbox_video_converter


class ontology_for_debug:
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ontology_for_debug(value))
            elif isinstance(value, list):
                setattr(
                    self,
                    key,
                    [
                        ontology_for_debug(item) if isinstance(item, dict) else item
                        for item in value
                    ],
                )
            else:
                setattr(self, key, value)


def ontology_for_debug_from_json(projects, project_id):

    global_objects = {}

    classifications = set()
    tools = {}

    # handle the rest of the tools if needed
    annotation_kind_map = {
        "VideoBoundingBox": "rectangle",
    }

    def parse_classification_(classification):
        d = {
            "feature_schema_id": classification["feature_schema_id"],
            "name": classification["name"],
            "options": [],
        }

        option = None

        # handle the rest of the tools if needed
        if "radio_answer" in classification:
            d["class_type"] = "radio"
            option = {
                "name": classification["radio_answer"]["name"],
                "value": classification["radio_answer"]["value"],
                "feature_schema_id": classification["radio_answer"][
                    "feature_schema_id"
                ],
            }

        if "checkbox_answers" in classification:
            d["class_type"] = "checkbox"
            option = {
                "name": classification["checkbox_answers"]["name"],
                "value": classification["checkbox_answers"]["value"],
                "feature_schema_id": classification["checkbox_answers"][
                    "feature_schema_id"
                ],
            }

        assert option is not None

        if classification["feature_schema_id"] not in global_objects:
            global_objects[classification["feature_schema_id"]] = d

        d = global_objects[classification["feature_schema_id"]]

        if option not in d["options"]:
            d["options"].append(option)

        return d

    def parse_tool(tool):
        tools[tool["feature_schema_id"]] = {
            "feature_schema_id": tool["feature_schema_id"],
            "name": tool["name"],
            "tool": annotation_kind_map[tool["annotation_kind"]],
        }

        classifications = []
        for c in tool.get("classifications", []):
            parse_classification_(c)
            classifications.append(c["feature_schema_id"])

        tools[tool["feature_schema_id"]]["classifications"] = classifications

    for p in projects:
        for label in p["projects"][project_id]["labels"]:
            for _, frame in label["annotations"]["frames"].items():
                for f_id, tool in frame["objects"].items():
                    parse_tool(tool)

                for classification in frame["classifications"]:
                    d = parse_classification_(classification)
                    classifications.add(d["feature_schema_id"])

    final_tools = list(tools.values())

    for tool in final_tools:
        for idx in range(len(tool["classifications"])):
            tool["classifications"][idx] = global_objects[tool["classifications"][idx]]

    final_classifications = []

    for classification in classifications:
        final_classifications.append(global_objects[classification])

    return ontology_for_debug(
        {"classifications": final_classifications, "tools": final_tools}
    )


class labelbox_video_converter_debug(labelbox_video_converter):
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

    def register_tool_(self, tool, context, fix_grouping_only):
        if tool.tool not in self.labelbox_type_converters_:
            print("skip tool:", tool.tool)
            return

        prefered_name = tool.name

        if tool.tool in self.group_mapping:
            prefered_name = self.group_mapping[tool.tool]
        else:
            prefered_name = tool.name

        should_group_with_classifications = len(tool.classifications) > 0
        if should_group_with_classifications:
            tool_name = prefered_name + "/" + prefered_name
            if fix_grouping_only:
                if tool.tool in self.group_mapping:
                    self.groupped_tensor_overrides[tool.tool] = tool_name
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

        if tool.tool in self.groupped_tensor_overrides:
            tool_name = self.groupped_tensor_overrides[tool.tool]

        self.labelbox_type_converters_[tool.tool](
            tool, self, tool_name, context, tool.tool in self.group_mapping
        )

    def register_classification_(self, tool, context, fix_grouping_only, parent=""):
        if tool.class_type not in self.labelbox_type_converters_:
            return

        if tool.class_type in self.group_mapping:
            prefered_name = (parent + "/" if parent else "") + self.group_mapping[
                tool.class_type
            ]
        else:
            prefered_name = (parent + "/" if parent else "") + tool.name

        if fix_grouping_only:
            return

        self.labelbox_type_converters_[tool.class_type](
            tool,
            self,
            prefered_name,
            context,
            tool.class_type in self.group_mapping,
        )

    def register_ontology_(self, ontology, context, fix_grouping_only=True):
        for tool in ontology.tools():
            self.register_tool_(tool, context, fix_grouping_only=fix_grouping_only)

        for classification in ontology.classifications():
            self.register_classification_(
                classification, context, fix_grouping_only=fix_grouping_only
            )

        if fix_grouping_only:
            self.register_ontology_(ontology, context, fix_grouping_only=False)
