from django.contrib.postgres.fields import ArrayField
from django.db import models


CATEGORY_CHOICES = [
        ('OPENING', 'Opening'),
        ('QUESTIONING', 'Questioning'),
        ('PRESENTING', 'Presenting'),
        ('CLOSING_OUTCOME', 'Closing & Outcome'),
    ]
class Transcription(models.Model):
    text = models.TextField()
    segments = ArrayField(
        models.TextField(),  # Define the type of data in the array
        size=None,  # If you do not want to limit the number of segments, keep size=None
        default=list  # Default to an empty list if no segments are provided
    )
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return str(self.id)

class Classification(models.Model):
    transcription = models.ForeignKey(Transcription, on_delete=models.CASCADE)
    segment_number = models.IntegerField(default=-1, blank=True, null=True)
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES)
    level = models.IntegerField(blank=True, null=True)
    statement = models.TextField()
    model_llm = models.TextField()

    def transcription_id(self):
        return self.transcription.id

    transcription_id.short_description = 'transcription ID'


class StatementType(models.Model):
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES, unique=True)
    definition = models.TextField()
    prompt = models.TextField()
    instruction = models.TextField()
    active = models.BooleanField(default=True)

class llmModel(models.Model):
    name = models.CharField(max_length=100)
    ibm_name = models.CharField(max_length=100)
    active = models.BooleanField(default=False)
