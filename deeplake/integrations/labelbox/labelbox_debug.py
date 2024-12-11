# classes in this file are needed to debug the ontology and labelbox projects when there's no access to the labelbox workspace


# helper classes to support same accessors as labelbox instances for accessing the ontology
class ontology_list_for_debug(list):
    def __call__(self):
        return self


class ontology_for_debug:
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ontology_for_debug(value))
            elif isinstance(value, list):
                setattr(
                    self,
                    key,
                    ontology_list_for_debug(
                        [
                            ontology_for_debug(item) if isinstance(item, dict) else item
                            for item in value
                        ]
                    ),
                )
            else:
                setattr(self, key, value)

    def __call__(self):
        return self


# Creates ontology object from the final exported labelbox project.
# This function shall replace `client.get_ontology(ontology_id)` in the converter script.
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
            "scope": {"value": "index"},
            "options": [],
        }

        option = None

        # handle the rest of the tools if needed
        if "radio_answer" in classification:
            d["class_type"] = {"value": "radio"}
            option = {
                "name": classification["radio_answer"]["name"],
                "value": classification["radio_answer"]["value"],
                "feature_schema_id": classification["radio_answer"][
                    "feature_schema_id"
                ],
            }

        if "checkbox_answers" in classification:
            d["class_type"] = {"value": "checkbox"}
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
            "tool": {"value": annotation_kind_map[tool["annotation_kind"]]},
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
