from django.contrib import admin
from .models import Script

admin.site.site_header = 'Landolt Website Administration'

class ScriptAdmin(admin.ModelAdmin):
    list_display = ('name', 'program', 'file', 'output_file', 'command')
    list_filter = ('program', 'file', 'output_file')
    search_fields = ('name', 'description', 'program', 'file', 'command')

admin.site.register(Script, ScriptAdmin)