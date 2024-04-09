from django.contrib import admin
from .models import Transcription, Classification, StatementType
# Register your models here.
@admin.register(Transcription)
class TranscriptionAdmin(admin.ModelAdmin):
    list_display = ('id', 'text', 'created_at')
    search_fields = ('id', 'text')

@admin.register(Classification)
class ClassificationAdmin(admin.ModelAdmin):
    list_display = ('id', 'transcription_id', 'category', 'statement', 'level')
    list_filter = ('category',)
    search_fields = ('category', 'transcription__id')

@admin.register(StatementType)
class StatementTypeAdmin(admin.ModelAdmin):
    list_display = ('id', 'category', 'definition', 'instruction', 'prompt')
    search_fields = ('id', 'category')
