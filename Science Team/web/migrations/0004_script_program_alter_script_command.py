# Generated by Django 5.1 on 2024-08-16 21:33

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('web', '0003_script_output_file_alter_script_file'),
    ]

    operations = [
        migrations.AddField(
            model_name='script',
            name='program',
            field=models.CharField(blank=True, choices=[('python', 'Python'), ('bash', 'Bash')], max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='script',
            name='command',
            field=models.CharField(blank=True, help_text='Raw command to run', max_length=100, null=True),
        ),
    ]
