from django.urls import path
from .views import ClassificationView, TranscriptionView, VectorDataView, LevellingDataView

urlpatterns = [
    path('classification/', ClassificationView.as_view(), name='classification'),
    path('transcription/', TranscriptionView.as_view(), name='transcript'),
    path('vectorDb/', VectorDataView.as_view(), name='vectorDb'),
    path('rating/', LevellingDataView.as_view(), name='rating'),
]