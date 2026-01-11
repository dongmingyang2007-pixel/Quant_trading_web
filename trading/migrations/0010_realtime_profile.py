from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
import uuid


class Migration(migrations.Migration):
    dependencies = [
        ("trading", "0009_userprofile_api_credentials"),
    ]

    operations = [
        migrations.CreateModel(
            name="RealtimeProfile",
            fields=[
                ("profile_id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ("name", models.CharField(max_length=80)),
                ("description", models.CharField(blank=True, default="", max_length=200)),
                ("payload", models.JSONField(default=dict)),
                ("is_active", models.BooleanField(default=False)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="realtime_profiles",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "ordering": ["-updated_at"],
            },
        ),
        migrations.AddIndex(
            model_name="realtimeprofile",
            index=models.Index(fields=["user", "-updated_at"], name="trading_rea_user_id_9a5b4e_idx"),
        ),
        migrations.AddIndex(
            model_name="realtimeprofile",
            index=models.Index(fields=["user", "is_active"], name="trading_rea_user_id_68d5e6_idx"),
        ),
    ]
