from django import forms
from .models import Table


class TableSettings(forms.ModelForm):
    class Meta:
            model = Table
            fields = ["epoch",
                 "bstar",
                 "ndot",
                 "nddot",
                 "ecco",
                 "argpo",
                 "inclo",
                 "mo",
                 "no_kozai",
                 "nodeo",
                 "timezone",
                 "start",
                 "end",
                 "lat",
                 "lon",
                 "elev",
                 "tdelta",
                 "chunks",
                 "tle1",
                 "tle2",
                 "t_eff",
                 "ccd_eff",
                 "t_diam",
                 "beta",
                 "n",
                 "humidity"]