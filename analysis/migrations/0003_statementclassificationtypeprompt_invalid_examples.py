# Generated by Django 5.0.4 on 2024-05-19 22:37

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('analysis', '0002_llmmodel_statementclassificationtypeprompt_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='statementclassificationtypeprompt',
            name='invalid_examples',
            field=models.TextField(default=''),
            preserve_default=False,
        ),
    ]
