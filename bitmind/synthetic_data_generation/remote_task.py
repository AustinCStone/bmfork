from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum
from datetime import datetime

class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class GenerationTask:
    task_id: str
    provider: str
    prompt: str
    start_time: datetime
    status: TaskStatus = TaskStatus.PENDING
    video_url: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class TaskTracker:
    def __init__(self):
        self.tasks: Dict[str, GenerationTask] = {}

    def add_task(self, task_id: str, provider: str, prompt: str) -> GenerationTask:
        task = GenerationTask(
            task_id=task_id,
            provider=provider,
            prompt=prompt,
            start_time=datetime.now()
        )
        self.tasks[task_id] = task
        return task

    def update_task(self, task_id: str, status: TaskStatus, **kwargs):
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.status = status
            for key, value in kwargs.items():
                setattr(task, key, value)

    def get_pending_tasks(self) -> Dict[str, GenerationTask]:
        return {
            task_id: task 
            for task_id, task in self.tasks.items() 
            if task.status in [TaskStatus.PENDING, TaskStatus.PROCESSING]
        }

    def get_task(self, task_id: str) -> Optional[GenerationTask]:
        return self.tasks.get(task_id)

    def get_tasks_by_provider(self, provider: str) -> Dict[str, GenerationTask]:
        return {
            task_id: task 
            for task_id, task in self.tasks.items() 
            if task.provider == provider
        }