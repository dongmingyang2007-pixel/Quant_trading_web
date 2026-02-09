from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("trading", "0014_rename_trading_not_recipie_3ea86d_idx_trading_not_recipie_945806_idx_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="userprofile",
            name="api_credentials_encrypted",
            field=models.TextField(blank=True, default=""),
        ),
    ]
