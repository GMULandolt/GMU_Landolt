# Generated by Django 5.1 on 2024-08-16 21:28

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('web', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='script',
            name='file',
            field=models.CharField(default=None, max_length=100),
            preserve_default=False,
        ),
    ]
