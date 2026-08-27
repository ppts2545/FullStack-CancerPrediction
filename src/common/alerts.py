"""Shared Airflow failure callback. Wire a real Slack/email backend here."""
import logging

logger = logging.getLogger(__name__)


def on_failure_alert(context):
    task_instance = context["task_instance"]
    logger.error(
        "task failed dag=%s task=%s run_id=%s - wire this into Slack/PagerDuty for prod",
        task_instance.dag_id,
        task_instance.task_id,
        context["run_id"],
    )
