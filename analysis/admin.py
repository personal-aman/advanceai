from django.contrib import admin

from .aiService.weaviateDb import storeData
from .models import Transcription, Classification, StatementType, llmModel
# Register your models here.


@admin.action(description="correct statement push to db")
def push_to_right_classification(modeladmin, request, queryset):
    classified_objects = {}
    classified_objects['QUESTIONING_CORRECT_EXAMPLE'] = queryset.filter(category="QUESTIONING").all()
    classified_objects['OPENING_CORRECT_EXAMPLE'] = queryset.filter(category="OPENING").all()
    classified_objects['PRESENTING_CORRECT_EXAMPLE'] = queryset.filter(category="PRESENTING").all()
    for name in classified_objects.keys():
        if len(classified_objects[name]) > 0:
            print(name, classified_objects[name].values_list('statement', flat=True))
            storeData(name, classified_objects[name].values_list('statement', flat=True))
@admin.action(description="incorrect statement push to db")
def push_to_wrong_classification(modeladmin, request, queryset):
    classified_objects = {}
    classified_objects['QUESTIONING_INCORRECT_EXAMPLE'] = queryset.filter(category="QUESTIONING").all()
    classified_objects['OPENING_INCORRECT_EXAMPLE'] = queryset.filter(category="OPENING").all()
    classified_objects['PRESENTING_INCORRECT_EXAMPLE'] = queryset.filter(category="PRESENTING").all()
    for name in classified_objects.keys():
        if len(classified_objects[name]) > 0:
            print(name, classified_objects[name].values_list('statement', flat=True))
            storeData(name, classified_objects[name].values_list('statement', flat=True))
    # print(objects[0].category)
    # print(request)

@admin.register(Transcription)
class TranscriptionAdmin(admin.ModelAdmin):
    list_display = ('id', 'text', 'created_at')
    search_fields = ('id', 'text')

@admin.register(Classification)
class ClassificationAdmin(admin.ModelAdmin):
    list_display = ('id', 'transcription_id', 'segment_number', 'category', 'statement', 'model_llm', 'level')
    list_filter = ('category', 'model_llm', 'segment_number')
    search_fields = ('category', 'transcription__id')
    actions = [push_to_right_classification, push_to_wrong_classification]


@admin.register(StatementType)
class StatementTypeAdmin(admin.ModelAdmin):
    list_display = ('id', 'category', 'definition', 'instruction', 'prompt', 'active')
    search_fields = ('id', 'category')


@admin.register(llmModel)
class llmModelAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'ibm_name', 'active')