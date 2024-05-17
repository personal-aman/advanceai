# Generated by Django 5.0.4 on 2024-05-17 00:04

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('analysis', '0017_rename_finalstatement_finalstatementwithlevel_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='statementclassification',
            name='confidence_score',
        ),
        migrations.RemoveField(
            model_name='statementclassification',
            name='level',
        ),
        migrations.RemoveField(
            model_name='statementclassification',
            name='model_llm',
        ),
        migrations.RemoveField(
            model_name='statementclassification',
            name='reason_for_level',
        ),
        migrations.AddField(
            model_name='statementclassification',
            name='levelDone',
            field=models.BooleanField(default=False),
        ),
    ]