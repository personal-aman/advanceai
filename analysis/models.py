from django.db import models


CATEGORY_CHOICES = [
        ('OPENING', 'Opening'),
        ('QUESTIONING', 'Questioning'),
        ('PRESENTING', 'Presenting'),
        ('CLOSING_OUTCOME', 'Closing & Outcome'),
    ]
class Transcription(models.Model):
    text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return str(self.id)

class Classification(models.Model):
    transcription = models.ForeignKey(Transcription, on_delete=models.CASCADE)
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES)
    level = models.IntegerField(blank=True, null=True)
    statement = models.TextField()

    def transcription_id(self):
        return self.transcription.id

    transcription_id.short_description = 'transcription ID'


class StatementType(models.Model):
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES, unique=True)
    definition = models.TextField()
    prompt = models.TextField()
    instruction = models.TextField()

