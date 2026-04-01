from django.db import models
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError

class BasePrediction(models.Model):
    user = models.ForeignKey(User,on_delete=models.CASCADE,related_name="predictions")

    input_data = models.JSONField(null=True, blank=True)
    input_file = models.FileField(upload_to='inputs/', null=True, blank=True)

    predicted_class = models.CharField(max_length=255)
    confidence = models.FloatField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    def clean(self):
        if not self.input_data and not self.input_file:
            raise ValidationError("Either input_data or input_file must be provided.")

    def __str__(self):
        return f"{self.user.username} - {self.predicted_class} - {self.created_at}"

    class Meta:
        abstract = True
        ordering = ['-created_at']

class Prediction(BasePrediction):
    pass