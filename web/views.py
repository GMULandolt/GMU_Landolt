from django.shortcuts import render, get_object_or_404
from django.conf import settings
from django.http import HttpResponse
from .models import Script
from .forms import TableSettings
from django.http import JsonResponse
import json

import magic
import os


def home(request):
    form = TableSettings(request.POST)
    if form.is_valid():
        data = form.cleaned_data
        save_file = open(os.path.join(settings.SCRIPTS_DIR, "Working_Files/settings.json"), "w")
        jsondict = JsonResponse(data).content.decode("utf-8")
        json.dump(jsondict, save_file, indent = 6)  
        save_file.close()
    else:
        data = form.errors.as_json()
        save_file = open(os.path.join(settings.SCRIPTS_DIR, "Working_Files/settings.json"), "w")
        jsondict = JsonResponse(data, status=400).content.content.decode("utf-8")
        json.dump(jsondict, save_file, indent = 6)  
        save_file.close()
    return render(request, 'home.html', {'form': form})


def scripts(request):
    context = {
        "scripts": Script.objects.all()
    }
    return render(request, 'scripts.html', context)


def run_script(request, pk):
    script = get_object_or_404(Script, pk=pk)
    output = script.run()
    context = {
        "script": script,
        "output": output.stdout.decode(),
    }
    return render(request, 'script_output.html', context)


def download_output_file(request, pk):
    script = get_object_or_404(Script, pk=pk)
    file = os.path.join(settings.SCRIPTS_DIR, script.output_file)
    mime = magic.Magic(mime=True)
    mime_type = mime.from_file(file)
    with open(file, 'rb') as data:
        response = HttpResponse(data, content_type=mime_type)
    response['Content-Disposition'] = f'attachment; filename="{os.path.basename(file)}"'
    return response
