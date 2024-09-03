from django.db import models
from django.conf import settings
import subprocess
import os


PROGRAM_CHOICES = [
    ('python', 'Python'),
    ('bash', 'Bash'),
]

class Script(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()

    program = models.CharField(max_length=100, choices=PROGRAM_CHOICES, blank=True, null=True)
    file = models.CharField(max_length=100, blank=True, null=True)
    output_file = models.CharField(max_length=100, blank=True, null=True)
    command = models.CharField(max_length=100, blank=True, null=True, help_text="Advanced: raw command to run")

    def run(self):
        if self.command:
            return subprocess.run(self.command, shell=True, stdout=subprocess.PIPE)
        filepath = os.path.join(settings.SCRIPTS_DIR, self.file)
        directory = os.path.dirname(filepath)
        return subprocess.run([self.program, filepath], stdout=subprocess.PIPE, cwd=directory)

    def __str__(self):
        return self.name

class Table(models.Model):
    epoch = models.FloatField(blank=True)
    bstar = models.FloatField(blank=True)
    ndot = models.FloatField(blank=True)
    nddot = models.FloatField(blank=True)
    ecco = models.FloatField(blank=True)
    argpo = models.FloatField(blank=True)
    inclo = models.FloatField(blank=True)
    mo = models.FloatField(blank=True)
    no_kozai = models.FloatField(blank=True)
    nodeo = models.FloatField(blank=True)
    timezone = models.CharField(max_length=50, blank=True)
    start = models.CharField(max_length=50, blank=True)
    end = models.CharField(max_length=50, blank=True)
    lat = models.FloatField(blank=True)
    lon = models.FloatField(blank=True)
    elev = models.FloatField(blank=True)
    tdelta = models.FloatField(blank=True)
    chunks = models.FloatField(blank=True)
    tle1 = models.CharField(max_length=69, blank=True)
    tle2 = models.CharField(max_length=69, blank=True)
    t_eff = models.FloatField(blank=True)
    ccd_eff = models.FloatField(blank=True)
    t_diam = models.FloatField(blank=True)
    beta = models.FloatField(blank=True)
    n = models.FloatField(blank=True)
    humidity = models.CharField(max_length=69, blank=True)

    def __str__(self):
        return self.name
