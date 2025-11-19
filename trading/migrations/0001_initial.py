from django.db import migrations, models
from django.conf import settings


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='BacktestRecord',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('record_id', models.CharField(max_length=64, unique=True)),
                ('timestamp', models.DateTimeField()),
                ('ticker', models.CharField(max_length=32)),
                ('benchmark', models.CharField(blank=True, default='', max_length=32)),
                ('engine', models.CharField(blank=True, default='', max_length=64)),
                ('start_date', models.CharField(blank=True, default='', max_length=16)),
                ('end_date', models.CharField(blank=True, default='', max_length=16)),
                ('metrics', models.JSONField(default=list)),
                ('stats', models.JSONField(default=dict)),
                ('params', models.JSONField(default=dict)),
                ('warnings', models.JSONField(default=list)),
                ('snapshot', models.JSONField(default=dict)),
                ('user', models.ForeignKey(on_delete=models.deletion.CASCADE, related_name='backtests', to=settings.AUTH_USER_MODEL)),
            ],
            options={'ordering': ['-timestamp']},
        ),
        migrations.AddIndex(
            model_name='backtestrecord',
            index=models.Index(fields=['user', 'timestamp'], name='trading_backtest_user_time_idx'),
        ),
    ]

