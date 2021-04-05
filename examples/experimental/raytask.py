import os
import yaml
import ray
from task import Task
from logger import logger
from utils import exec


YAML_DIR = "."
HUB_PATH = "../../../Hub"


class RayTask(Task):
    """A class to execute python scripts on a ray cluster created from generated .yaml file"""

    def __init__(
        self,
        name: str,
        min_workers: int = 0,
        max_workers: int = 0,
        cloud: str = "aws",
        head_machine_type: str = "m4.xlarge",
        workers_machine_type: str = "m4.xlarge",
        creds: dict = {},
        region: str = "us-east-1",
        docker: str = "",
        image_id: str = "ami-02e86b825fe559330",
        file_mounts: dict = {"Hub": HUB_PATH},
        setup_commands: list = [
            "cd Hub && pip3 install -e .",
            "cd Hub && pip3 install -r requirements-dev.txt",
            "pip3 install ray==1.0.0",
            "sudo apt-get install -y tmux",
        ],
    ):
        super(RayTask, self).__init__(
            name=name,
            docker=docker,
            region=region,
            creds=creds,
            head_machine_type=head_machine_type,
            workers_machine_type=workers_machine_type,
            cloud=cloud,
            workers=max_workers,
            image_id=image_id,
        )
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.setup_commands = setup_commands
        self.file_mounts = file_mounts
        self.config_path = self._configure_yaml()

    def _configure_yaml(self):
        """ Constructs yaml based on inputs """
        print(YAML_DIR)
        with open(os.path.join(YAML_DIR, "full.yaml"), "r") as f:
            yaml_dict = yaml.load(f)

        yaml_dict["cluster_name"] = self.name
        yaml_dict["min_workers"] = self.min_workers
        yaml_dict["max_workers"] = self.max_workers
        yaml_dict["provider"]["type"] = self.cloud
        yaml_dict["provider"]["region"] = self.region
        yaml_dict["head_node"]["InstanceType"] = self.head_machine_type
        yaml_dict["head_node"]["ImageId"] = self.image_id
        yaml_dict["worker_nodes"]["InstanceType"] = self.workers_machine_type
        yaml_dict["worker_nodes"]["ImageId"] = self.image_id
        yaml_dict["file_mounts"] = self.file_mounts
        yaml_dict["setup_commands"] = self.setup_commands
        if self.docker:
            yaml_dict["docker"]["image"] = self.docker

        output_path = os.path.join(YAML_DIR, f"{self.name}.yaml")
        with open(output_path, "w") as f:
            yaml.dump(yaml_dict, f, default_flow_style=False, sort_keys=False)
        return output_path

    def get_pids(self, pgrep_output: str):
        "Get pids of main running script"
        pids = [
            line.strip()
            for line in pgrep_output.split("\n")
            if len(line.strip()) < 6 and line.strip()
        ]
        return pids

    def get_process_pid(self):
        """Read pid stored in file"""
        with open("ray_pid.txt", "r") as f:
            pid = f.readline().strip()
        return pid

    def launch_cluster(self) -> bool:
        """Start a cluster with given configurations"""
        try:
            self._state = self.LAUNCHED
            logs = exec(f"ray up -y {self.config_path}", sync=True)
            logger.info(logs)
            print(logs)
        except Exception as e:
            logger.error(e, exc_info=e, stack_info=True)
            self._state = self.ERROR
            return False
        return True

    def stop_cluster(self) -> bool:
        """ Shut down the cluster """
        self._state = self.DONE
        try:
            logs = exec(f"ray down -y {self.config_path}", sync=True)
            logger.info(logs)
        except Exception as e:
            logger.error(e, exc_info=e, stack_info=True)
            self._state = self.ERROR
            return False
        return True

    def submit(self, script) -> bool:
        """ Execute the task and write the process pid to the file"""
        try:
            exec(f"rm -f {self.log_path}", sync=True)
            if "submit" in script or "exec" in script:
                p_script = f"""{script} 2>&1 | tee {self.log_path} && ray stop"""
            else:
                p_script = (
                    f"""ray exec {os.path.join(YAML_DIR, self.name)}.yaml {script}"""
                )
            self._state = self.RUNNING
            self.tasks[script] = exec(p_script, sync=False)
            pyt_script = (
                script.replace(
                    f"ray submit {os.path.join(YAML_DIR, self.name)}.yaml ", ""
                )
                .replace("--stop", "")
                .replace("--start", "")
                .strip()
                .split(" ")
            )
            pyt_script[0] = pyt_script[0].split("/")[-1]
            pyt_script = " ".join(pyt_script)
            _, pgrep_output = exec(
                f"""ray exec {os.path.join(YAML_DIR, self.name)}.yaml 'pgrep -f "{pyt_script}"'""",
                sync=True,
            )
            with open("ray_pid.txt", "w") as f:
                pid = self.get_pids(pgrep_output)[0].strip()
                f.write(pid)
        except Exception as e:
            logger.error(e, exc_info=e, stack_info=True)
            self._state = self.ERROR
            return False
        return True

    def resubmit(self, script):
        """Stop executing the script if it is running and restart execution"""
        if self.is_running(script):
            self.stop(script)
        self.submit(script)
        return self.is_running(script)

    def is_running(self, script) -> bool:
        """ Check if the script is running"""
        if script in self.tasks and self.tasks[script].poll() is None:
            return True
        pids = self.get_process_pid()
        return len(pids) > 0

    def get_logs(self) -> str:
        """ Receive logs """
        try:
            with open(self.log_path, "r") as log_file:
                logs = log_file.readlines()
                return "".join(logs)
        except Exception as e:
            logger.error(e, exc_info=e, stack_info=True)
            return ""

    def get_usage(self) -> dict:
        """ Get CPU, GPU, Memory utilization metrics"""
        ray.init(address="auto", _redis_password="5241590000000000")
        total_resources = ray.cluster_resources()
        available_resources = ray.available_resources()

        res_dict = {}
        for param in total_resources.keys():
            if "node" not in param:
                res_dict[
                    param
                ] = f"{available_resources[param]}:{total_resources[param]}"
        return res_dict

    def stop(self, script) -> bool:
        """ Stop the process """
        try:
            init_script = script
            if "ray" in script:
                script = script.split(" ")[2]
            with open("ray_pid.txt", "r") as f:
                pid = f.readline().strip()
            exec(
                f"ray exec {YAML_DIR}{self.name}.yaml 'pkill -s {pid}'",
                sync=True,
            )
            self._state = self.TERMINATED
            del self.tasks[init_script]
            open("ray_pid.txt", "w").close()
        except Exception as e:
            logger.error(e, exc_info=e, stack_info=True)
            self._state = self.ERROR
            return False
        return True
