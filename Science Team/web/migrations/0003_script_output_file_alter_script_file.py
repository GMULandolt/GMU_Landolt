# Generated by Django 5.1 on 2024-08-16 21:29

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('web', '0002_script_file'),
    ]

    operations = [
        migrations.AddField(
            model_name='script',
            name='output_file',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='script',
            name='file',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
    ]
