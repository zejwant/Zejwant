# ingest_scheduler.py
"""
Enterprise-grade ingestion scheduler.

Features:
- Cron-style scheduling for ingestion pipelines
- Supports API, DB, streaming, and file upload triggers
- Logging, retry, and error handling
- Email/Slack notifications on critical failures
- Configuration via YAML or JSON
- Async and concurrent execution
"""

from typing import Any, Callable, Dict, List, Optional
import asyncio
import logging
import time
import json
import smtplib
from pathlib import Path
from email.message import EmailMessage
import yaml
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_ERROR

# Logging setup
logger = logging.getLogger("ingest_scheduler")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter(
    '{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
ch.setFormatter(formatter)
logger.addHandler(ch)


class IngestScheduler:
    def __init__(
        self,
        config_file: Optional[Path] = None,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        alert_email: Optional[str] = None,
        alert_slack_webhook: Optional[str] = None
    ):
        """
        Initialize ingestion scheduler.

        Args:
            config_file (Path, optional): YAML or JSON configuration file
            max_retries (int): Max retries per ingestion job
            retry_backoff (float): Exponential backoff factor for retries
            alert_email (str, optional): Email for critical alerts
            alert_slack_webhook (str, optional): Slack webhook URL for alerts
        """
        self.scheduler = AsyncIOScheduler()
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.alert_email = alert_email
        self.alert_slack_webhook = alert_slack_webhook
        self.jobs_config = self._load_config(config_file) if config_file else []

        self.scheduler.add_listener(self._job_error_listener, EVENT_JOB_ERROR)

    def _load_config(self, config_file: Path) -> List[Dict[str, Any]]:
        """
        Load YAML or JSON config for scheduled jobs.

        Args:
            config_file (Path): Path to YAML/JSON config file

        Returns:
            List[Dict]: List of job configurations
        """
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                if config_file.suffix in [".yaml", ".yml"]:
                    return yaml.safe_load(f)
                elif config_file.suffix == ".json":
                    return json.load(f)
                else:
                    raise ValueError("Unsupported config format")
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            return []

    def add_job(
        self,
        job_func: Callable,
        cron: str,
        job_name: str,
        *args,
        **kwargs
    ):
        """
        Add a scheduled ingestion job.

        Args:
            job_func (Callable): Async function to execute
            cron (str): Cron expression string
            job_name (str): Job identifier
        """
        trigger = CronTrigger.from_crontab(cron)
        self.scheduler.add_job(
            self._job_wrapper,
            trigger,
            args=(job_func, job_name) + args,
            kwargs=kwargs,
            id=job_name,
            name=job_name,
            replace_existing=True
        )
        logger.info(f"Scheduled job '{job_name}' with cron '{cron}'")

    async def _job_wrapper(self, job_func: Callable, job_name: str, *args, **kwargs):
        """
        Wrapper for retries, logging, and error handling.

        Args:
            job_func (Callable): The ingestion function
            job_name (str): Name of the job
        """
        retries = 0
        while retries <= self.max_retries:
            try:
                start_time = time.time()
                if asyncio.iscoroutinefunction(job_func):
                    await job_func(*args, **kwargs)
                else:
                    # Run sync function in threadpool
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, job_func, *args, **kwargs)
                end_time = time.time()
                logger.info(json.dumps({
                    "job_name": job_name,
                    "status": "success",
                    "runtime_seconds": end_time - start_time
                }))
                break
            except Exception as e:
                wait = self.retry_backoff * (2 ** retries)
                logger.warning(json.dumps({
                    "job_name": job_name,
                    "status": "failed",
                    "retry": retries,
                    "error": str(e),
                    "retry_in_seconds": wait
                }))
                retries += 1
                await asyncio.sleep(wait)
                if retries > self.max_retries:
                    logger.error(json.dumps({
                        "job_name": job_name,
                        "status": "failed_permanently",
                        "error": str(e)
                    }))
                    await self._send_alert(job_name, str(e))

    async def _send_alert(self, job_name: str, message: str):
        """
        Send critical failure alert via email or Slack.

        Args:
            job_name (str): Job identifier
            message (str): Error message
        """
        if self.alert_email:
            try:
                msg = EmailMessage()
                msg.set_content(f"Ingestion job '{job_name}' failed.\n\nError:\n{message}")
                msg["Subject"] = f"[ALERT] Ingestion Job Failed: {job_name}"
                msg["From"] = "noreply@ingest-system.com"
                msg["To"] = self.alert_email
                # Example using local SMTP server
                with smtplib.SMTP("localhost") as server:
                    server.send_message(msg)
                logger.info(f"Sent email alert for job '{job_name}'")
            except Exception as e:
                logger.warning(f"Failed to send email alert: {e}")

        if self.alert_slack_webhook:
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    await session.post(
                        self.alert_slack_webhook,
                        json={"text": f":red_circle: Job '{job_name}' failed\n```{message}```"}
                    )
                logger.info(f"Sent Slack alert for job '{job_name}'")
            except Exception as e:
                logger.warning(f"Failed to send Slack alert: {e}")

    def _job_error_listener(self, event):
        """
        APScheduler job error listener
        """
        logger.error(f"Job {event.job_id} raised an error: {event.exception}")

    def start(self):
        """
        Start the scheduler and schedule jobs from config (if any)
        """
        for job in self.jobs_config:
            self.add_job(
                job_func=job["job_func"],
                cron=job["cron"],
                job_name=job.get("job_name", f"job_{job['cron']}")
            )
        self.scheduler.start()
        logger.info("IngestScheduler started")

    def stop(self):
        """
        Stop the scheduler
        """
        self.scheduler.shutdown()
        logger.info("IngestScheduler stopped")
      
