from django.db import migrations, models
import uuid
import paper.models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ("auth", "0012_alter_user_first_name_max_length"),
    ]

    operations = [
        migrations.CreateModel(
            name="PaperTradingSession",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("session_id", models.UUIDField(default=uuid.uuid4, editable=False, unique=True)),
                ("name", models.CharField(blank=True, default="", max_length=120)),
                ("ticker", models.CharField(max_length=16)),
                ("benchmark", models.CharField(blank=True, default="", max_length=16)),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("draft", "Draft"),
                            ("running", "Running"),
                            ("paused", "Paused"),
                            ("stopped", "Stopped"),
                            ("error", "Error"),
                        ],
                        default="running",
                        max_length=16,
                    ),
                ),
                ("config", models.JSONField(blank=True, default=dict)),
                ("current_cash", models.DecimalField(decimal_places=2, default=0, max_digits=18)),
                ("initial_cash", models.DecimalField(decimal_places=2, default=0, max_digits=18)),
                ("current_positions", models.JSONField(blank=True, default=paper.models.default_positions)),
                ("equity_curve", models.JSONField(blank=True, default=paper.models.default_curve)),
                ("last_equity", models.DecimalField(decimal_places=2, default=0, max_digits=18)),
                ("interval_seconds", models.PositiveIntegerField(default=300)),
                ("last_run_at", models.DateTimeField(blank=True, null=True)),
                ("next_run_at", models.DateTimeField(blank=True, null=True)),
                ("started_at", models.DateTimeField(auto_now_add=True)),
                ("ended_at", models.DateTimeField(blank=True, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=models.CASCADE, related_name="paper_sessions", to="auth.user"
                    ),
                ),
            ],
            options={
                "ordering": ["-updated_at"],
            },
        ),
        migrations.CreateModel(
            name="PaperTrade",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                (
                    "side",
                    models.CharField(choices=[("buy", "Buy"), ("sell", "Sell")], max_length=8),
                ),
                ("symbol", models.CharField(max_length=16)),
                ("quantity", models.DecimalField(decimal_places=6, max_digits=18)),
                ("price", models.DecimalField(decimal_places=6, max_digits=18)),
                ("notional", models.DecimalField(decimal_places=2, max_digits=18)),
                ("executed_at", models.DateTimeField(auto_now_add=True)),
                ("metadata", models.JSONField(blank=True, default=dict)),
                (
                    "session",
                    models.ForeignKey(
                        on_delete=models.CASCADE, related_name="trades", to="paper.papertradingsession"
                    ),
                ),
            ],
            options={
                "ordering": ["-executed_at"],
            },
        ),
        migrations.AddIndex(
            model_name="papertradingsession",
            index=models.Index(fields=["user", "-updated_at"], name="paper_paper_user_id_a1109b_idx"),
        ),
        migrations.AddIndex(
            model_name="papertradingsession",
            index=models.Index(fields=["status", "next_run_at"], name="paper_paper_status__1cd6ac_idx"),
        ),
        migrations.AddIndex(
            model_name="papertrade",
            index=models.Index(fields=["session", "-executed_at"], name="paper_paper_session__015048_idx"),
        ),
    ]
