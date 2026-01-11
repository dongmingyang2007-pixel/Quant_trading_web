from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("trading", "0008_taskexecution"),
    ]

    operations = [
        migrations.AddField(
            model_name="userprofile",
            name="api_credentials",
            field=models.JSONField(blank=True, default=dict),
        ),
    ]
