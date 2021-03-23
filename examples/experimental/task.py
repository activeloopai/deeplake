import psutil
from utils import exec
from logger import logger


class Task(object):
    CREATED = "Created"
    LAUNCHED = "Launched"
    RUNNING = "Running"
    TERMINATED = "Terminated"
    DONE = "Done"
    ERROR = "Error"

    def __init__(
        self,
        name: str,
        workers: int = 0,
        cloud: str = "local",
        head_machine_type: str = "local",
        workers_machine_type: str = "local",
        creds: dict = {},
        region: str = "local",
        docker: str = None,
        image_id: str = None,
    ):
        super().__init__()
        self.name = name
        self.workers = workers
        self.cloud = cloud
        self.head_machine_type = head_machine_type
        self.workers_machine_type = workers_machine_type
        self.creds = creds
        self.region = region
        self.docker = docker
        self.image_id = image_id
        self.log_path = f"/tmp/{self.name}_log.txt"
        self._state = self.CREATED
        self.tasks = {}

    def get_state(self):
        return self._state

    def launch_cluster(self) -> bool:
        """ Launch the cluster """
        self._state = self.LAUNCHED
        return True

    def stop_cluster(self) -> bool:
        self._state = self.DONE
        return True

    def submit(self, script) -> bool:
        """ Execute the task """
        try:
            exec(f"rm -f {self.log_path}", sync=True)
            p_script = f"{script} >> {self.log_path}"
            self._state = self.RUNNING
            self.tasks[script] = exec(p_script, sync=False)
        except Exception as e:
            logger.error(e, exc_info=e, stack_info=True)
            self._state = self.ERROR
            return False
        return True

    def is_running(self, script) -> bool:
        """ Check if the script is running"""
        if script in self.tasks and self.tasks[script].poll() is None:
            return True

        cmd = f'pgrep -f "/bin/sh -c {script}"'
        _, output = exec(cmd, sync=True)
        return len(output.replace("\n", "")) > 1

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
        return {
            "cpu": psutil.cpu_percent(),
            "ram": psutil.virtual_memory().percent,
            "ram_full": dict(psutil.virtual_memory()._asdict()),
        }

    def stop(self, script) -> bool:
        """ Stop the process """
        try:
            exec(f'pkill -9 -f "{script}"', sync=True)
            self._state = self.TERMINATED
            del self.tasks[script]
        except Exception as e:
            logger.error(e, exc_info=e, stack_info=True)
            self._state = self.ERROR
            return False
        return True
