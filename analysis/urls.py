from django.urls import path
from .views import ClassificationView, TranscriptionView, VectorDataView

urlpatterns = [
    path('classification/', ClassificationView.as_view(), name='classification'),
    path('transcription/', TranscriptionView.as_view(), name='transcript'),
    path('vectorDb/', VectorDataView.as_view(), name='vectorDb'),
]