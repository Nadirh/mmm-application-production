"""
Data retention and compliance module for MMM application.
Implements automated data lifecycle management and regulatory compliance.
"""
import os
from typing import Dict, List, Optional, Any
from datetime import datetime, UTC, timedelta
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json
import structlog
import asyncio
from sqlalchemy import select, delete, and_

logger = structlog.get_logger()


class ComplianceFramework(Enum):
    """Regulatory compliance frameworks."""
    GDPR = "gdpr"           # EU General Data Protection Regulation
    CCPA = "ccpa"           # California Consumer Privacy Act
    HIPAA = "hipaa"         # Health Insurance Portability and Accountability Act
    SOC2 = "soc2"           # Service Organization Control 2
    ISO27001 = "iso27001"   # International Security Standard
    CUSTOM = "custom"       # Custom retention policy


class DataRetentionAction(Enum):
    """Actions for data retention policy."""
    ARCHIVE = "archive"         # Move to cold storage
    ANONYMIZE = "anonymize"     # Remove PII, keep aggregated data
    DELETE = "delete"           # Permanently delete
    EXPORT = "export"           # Export to client before deletion
    NOTIFY = "notify"           # Notify client of pending action


@dataclass
class RetentionPolicy:
    """Data retention policy configuration."""
    policy_id: str
    client_id: str
    data_type: str  # uploads, models, results, audit_logs
    retention_days: int
    action: DataRetentionAction
    compliance_framework: ComplianceFramework
    notify_before_days: int = 30
    auto_execute: bool = True
    created_at: datetime = None
    updated_at: datetime = None


@dataclass
class RetentionSchedule:
    """Scheduled retention action."""
    schedule_id: str
    policy_id: str
    client_id: str
    data_identifier: str
    scheduled_date: datetime
    action: DataRetentionAction
    status: str  # pending, notified, executed, failed
    notification_sent: bool = False
    execution_date: Optional[datetime] = None
    error_message: Optional[str] = None


class DataRetentionManager:
    """Manages data retention policies and compliance."""

    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        self.policies: Dict[str, RetentionPolicy] = {}
        self.schedules: Dict[str, RetentionSchedule] = {}

        # Default retention periods by compliance framework
        self.compliance_defaults = {
            ComplianceFramework.GDPR: {
                "uploads": 365,      # 1 year
                "models": 730,       # 2 years
                "results": 730,      # 2 years
                "audit_logs": 2555   # 7 years
            },
            ComplianceFramework.CCPA: {
                "uploads": 365,
                "models": 365,
                "results": 365,
                "audit_logs": 1095   # 3 years
            },
            ComplianceFramework.HIPAA: {
                "uploads": 2190,     # 6 years
                "models": 2190,
                "results": 2190,
                "audit_logs": 2190
            },
            ComplianceFramework.SOC2: {
                "uploads": 365,
                "models": 365,
                "results": 365,
                "audit_logs": 2555
            },
            ComplianceFramework.ISO27001: {
                "uploads": 1095,
                "models": 1095,
                "results": 1095,
                "audit_logs": 1095
            }
        }

    def create_retention_policy(self,
                              client_id: str,
                              data_type: str,
                              retention_days: Optional[int] = None,
                              action: DataRetentionAction = DataRetentionAction.DELETE,
                              compliance_framework: ComplianceFramework = ComplianceFramework.GDPR,
                              notify_before_days: int = 30,
                              auto_execute: bool = True) -> RetentionPolicy:
        """Create a retention policy for a client."""
        # Use compliance default if retention_days not specified
        if retention_days is None:
            retention_days = self.compliance_defaults[compliance_framework].get(
                data_type, 365
            )

        policy_id = f"policy_{client_id}_{data_type}"

        policy = RetentionPolicy(
            policy_id=policy_id,
            client_id=client_id,
            data_type=data_type,
            retention_days=retention_days,
            action=action,
            compliance_framework=compliance_framework,
            notify_before_days=notify_before_days,
            auto_execute=auto_execute,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC)
        )

        self.policies[policy_id] = policy

        logger.info("Created retention policy",
                   policy_id=policy_id,
                   client_id=client_id,
                   data_type=data_type,
                   retention_days=retention_days,
                   framework=compliance_framework.value)

        return policy

    async def scan_for_expired_data(self) -> List[RetentionSchedule]:
        """Scan for data that needs retention action."""
        schedules_created = []

        if not self.db_manager:
            logger.warning("No database manager configured for retention scanning")
            return schedules_created

        async with self.db_manager.get_session() as session:
            for policy in self.policies.values():
                # Calculate expiry date
                expiry_date = datetime.now(UTC) - timedelta(days=policy.retention_days)

                # Query based on data type
                if policy.data_type == "uploads":
                    # Find expired upload sessions
                    from mmm.database.models import UploadSession

                    result = await session.execute(
                        select(UploadSession).where(
                            and_(
                                UploadSession.upload_time < expiry_date,
                                # Add client filtering when client_id is added to model
                            )
                        )
                    )
                    expired_items = result.scalars().all()

                    for item in expired_items:
                        schedule = self._create_retention_schedule(
                            policy, item.id, item.upload_time
                        )
                        schedules_created.append(schedule)

                elif policy.data_type == "models":
                    # Find expired training runs
                    from mmm.database.models import TrainingRun

                    result = await session.execute(
                        select(TrainingRun).where(
                            TrainingRun.completion_time < expiry_date
                        )
                    )
                    expired_items = result.scalars().all()

                    for item in expired_items:
                        if item.completion_time:
                            schedule = self._create_retention_schedule(
                                policy, item.id, item.completion_time
                            )
                            schedules_created.append(schedule)

                elif policy.data_type == "results":
                    # Find expired optimization runs
                    from mmm.database.models import OptimizationRun

                    result = await session.execute(
                        select(OptimizationRun).where(
                            OptimizationRun.created_time < expiry_date
                        )
                    )
                    expired_items = result.scalars().all()

                    for item in expired_items:
                        schedule = self._create_retention_schedule(
                            policy, item.id, item.created_time
                        )
                        schedules_created.append(schedule)

        logger.info(f"Found {len(schedules_created)} items requiring retention action")
        return schedules_created

    def _create_retention_schedule(self,
                                  policy: RetentionPolicy,
                                  data_identifier: str,
                                  data_date: datetime) -> RetentionSchedule:
        """Create a retention schedule for a specific data item."""
        schedule_id = f"schedule_{policy.policy_id}_{data_identifier}"

        # Calculate scheduled action date
        scheduled_date = data_date + timedelta(days=policy.retention_days)

        schedule = RetentionSchedule(
            schedule_id=schedule_id,
            policy_id=policy.policy_id,
            client_id=policy.client_id,
            data_identifier=data_identifier,
            scheduled_date=scheduled_date,
            action=policy.action,
            status="pending"
        )

        self.schedules[schedule_id] = schedule

        # Check if notification needed
        notification_date = scheduled_date - timedelta(days=policy.notify_before_days)
        if datetime.now(UTC) >= notification_date and not schedule.notification_sent:
            self._send_retention_notification(schedule, policy)

        return schedule

    def _send_retention_notification(self,
                                    schedule: RetentionSchedule,
                                    policy: RetentionPolicy):
        """Send notification about upcoming retention action."""
        # In production, integrate with email/notification service
        logger.info("Sending retention notification",
                   client_id=schedule.client_id,
                   data_identifier=schedule.data_identifier,
                   action=schedule.action.value,
                   scheduled_date=schedule.scheduled_date.isoformat())

        schedule.notification_sent = True
        schedule.status = "notified"

    async def execute_retention_action(self,
                                      schedule: RetentionSchedule,
                                      force: bool = False) -> bool:
        """Execute a scheduled retention action."""
        try:
            policy = self.policies.get(schedule.policy_id)
            if not policy:
                logger.error("Policy not found for schedule",
                           schedule_id=schedule.schedule_id)
                return False

            # Check if execution is due
            if not force and datetime.now(UTC) < schedule.scheduled_date:
                logger.info("Retention action not yet due",
                          schedule_id=schedule.schedule_id,
                          scheduled_date=schedule.scheduled_date.isoformat())
                return False

            # Check auto-execute flag
            if not force and not policy.auto_execute:
                logger.info("Manual execution required for retention action",
                          schedule_id=schedule.schedule_id)
                return False

            # Execute action based on type
            success = False
            if schedule.action == DataRetentionAction.DELETE:
                success = await self._execute_deletion(schedule)
            elif schedule.action == DataRetentionAction.ARCHIVE:
                success = await self._execute_archival(schedule)
            elif schedule.action == DataRetentionAction.ANONYMIZE:
                success = await self._execute_anonymization(schedule)
            elif schedule.action == DataRetentionAction.EXPORT:
                success = await self._execute_export(schedule)

            if success:
                schedule.status = "executed"
                schedule.execution_date = datetime.now(UTC)
                logger.info("Retention action executed successfully",
                          schedule_id=schedule.schedule_id,
                          action=schedule.action.value)
            else:
                schedule.status = "failed"
                logger.error("Retention action failed",
                           schedule_id=schedule.schedule_id,
                           action=schedule.action.value)

            return success

        except Exception as e:
            schedule.status = "failed"
            schedule.error_message = str(e)
            logger.error("Error executing retention action",
                        schedule_id=schedule.schedule_id,
                        error=str(e))
            return False

    async def _execute_deletion(self, schedule: RetentionSchedule) -> bool:
        """Execute data deletion."""
        from mmm.security.data_isolation import data_isolation_manager

        # Use secure deletion from data isolation manager
        success = data_isolation_manager.delete_client_data(
            client_id=schedule.client_id,
            file_id=schedule.data_identifier,
            user_id="retention_policy",
            permanent=True  # Secure overwrite
        )

        if success and self.db_manager:
            # Also delete from database
            async with self.db_manager.get_session() as session:
                policy = self.policies.get(schedule.policy_id)

                if policy.data_type == "uploads":
                    from mmm.database.models import UploadSession
                    await session.execute(
                        delete(UploadSession).where(
                            UploadSession.id == schedule.data_identifier
                        )
                    )
                elif policy.data_type == "models":
                    from mmm.database.models import TrainingRun
                    await session.execute(
                        delete(TrainingRun).where(
                            TrainingRun.id == schedule.data_identifier
                        )
                    )
                elif policy.data_type == "results":
                    from mmm.database.models import OptimizationRun
                    await session.execute(
                        delete(OptimizationRun).where(
                            OptimizationRun.id == schedule.data_identifier
                        )
                    )

                await session.commit()

        return success

    async def _execute_archival(self, schedule: RetentionSchedule) -> bool:
        """Execute data archival to cold storage."""
        # In production, integrate with cloud storage (S3 Glacier, etc.)
        logger.info("Archiving data (mock)",
                   client_id=schedule.client_id,
                   data_identifier=schedule.data_identifier)

        # Move to archive storage with reduced access
        # Implementation depends on cloud provider
        return True

    async def _execute_anonymization(self, schedule: RetentionSchedule) -> bool:
        """Execute data anonymization."""
        if not self.db_manager:
            return False

        async with self.db_manager.get_session() as session:
            policy = self.policies.get(schedule.policy_id)

            if policy.data_type == "uploads":
                # Anonymize upload session
                from mmm.database.models import UploadSession

                result = await session.execute(
                    select(UploadSession).where(
                        UploadSession.id == schedule.data_identifier
                    )
                )
                upload = result.scalar_one_or_none()

                if upload:
                    # Remove identifiable information
                    upload.filename = f"anonymized_{upload.id[:8]}.csv"
                    # Keep aggregated statistics but remove raw data link
                    upload.file_path = None
                    await session.commit()
                    return True

        return False

    async def _execute_export(self, schedule: RetentionSchedule) -> bool:
        """Execute data export before deletion."""
        from mmm.security.data_isolation import data_isolation_manager

        try:
            # Retrieve data
            data, metadata = data_isolation_manager.retrieve_client_file(
                client_id=schedule.client_id,
                file_id=schedule.data_identifier,
                user_id="retention_policy"
            )

            # Create export package
            export_path = Path(f"/tmp/exports/{schedule.client_id}")
            export_path.mkdir(parents=True, exist_ok=True)

            export_file = export_path / f"export_{schedule.data_identifier}.json"
            export_data = {
                "export_date": datetime.now(UTC).isoformat(),
                "data_identifier": schedule.data_identifier,
                "metadata": metadata,
                # In production, include encrypted data or download link
            }

            with open(export_file, 'w') as f:
                json.dump(export_data, f)

            # In production, send to client or upload to secure transfer location
            logger.info("Data exported for retention",
                       client_id=schedule.client_id,
                       export_path=str(export_file))

            return True

        except Exception as e:
            logger.error("Failed to export data",
                        error=str(e),
                        schedule_id=schedule.schedule_id)
            return False

    def generate_compliance_report(self, client_id: str) -> Dict[str, Any]:
        """Generate compliance report for a client."""
        client_policies = [p for p in self.policies.values() if p.client_id == client_id]
        client_schedules = [s for s in self.schedules.values() if s.client_id == client_id]

        report = {
            "client_id": client_id,
            "report_date": datetime.now(UTC).isoformat(),
            "compliance_frameworks": list(set(p.compliance_framework.value for p in client_policies)),
            "policies": {
                "total": len(client_policies),
                "by_data_type": {},
                "by_action": {}
            },
            "scheduled_actions": {
                "total": len(client_schedules),
                "pending": sum(1 for s in client_schedules if s.status == "pending"),
                "notified": sum(1 for s in client_schedules if s.status == "notified"),
                "executed": sum(1 for s in client_schedules if s.status == "executed"),
                "failed": sum(1 for s in client_schedules if s.status == "failed")
            },
            "upcoming_actions": []
        }

        # Breakdown by data type
        for policy in client_policies:
            if policy.data_type not in report["policies"]["by_data_type"]:
                report["policies"]["by_data_type"][policy.data_type] = {
                    "retention_days": policy.retention_days,
                    "action": policy.action.value,
                    "framework": policy.compliance_framework.value
                }

            if policy.action.value not in report["policies"]["by_action"]:
                report["policies"]["by_action"][policy.action.value] = 0
            report["policies"]["by_action"][policy.action.value] += 1

        # List upcoming actions (next 30 days)
        upcoming_date = datetime.now(UTC) + timedelta(days=30)
        for schedule in client_schedules:
            if schedule.status == "pending" and schedule.scheduled_date <= upcoming_date:
                report["upcoming_actions"].append({
                    "data_identifier": schedule.data_identifier,
                    "scheduled_date": schedule.scheduled_date.isoformat(),
                    "action": schedule.action.value,
                    "notification_sent": schedule.notification_sent
                })

        return report

    async def run_retention_scheduler(self, interval_hours: int = 24):
        """Run retention scheduler as a background task."""
        while True:
            try:
                logger.info("Running retention scheduler")

                # Scan for expired data
                schedules = await self.scan_for_expired_data()

                # Process pending schedules
                for schedule in self.schedules.values():
                    if schedule.status in ["pending", "notified"]:
                        await self.execute_retention_action(schedule)

                logger.info(f"Retention scheduler completed, processed {len(schedules)} items")

            except Exception as e:
                logger.error("Error in retention scheduler", error=str(e))

            # Wait for next run
            await asyncio.sleep(interval_hours * 3600)


# Singleton instance
data_retention_manager = DataRetentionManager()