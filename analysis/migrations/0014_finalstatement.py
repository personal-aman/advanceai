# Generated by Django 5.0.4 on 2024-05-16 22:53

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('analysis', '0013_rename_instruction_statementtype_examples_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='FinalStatement',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('category', models.CharField(choices=[('OPENING', 'Opening'), ('QUESTIONING', 'Questioning'), ('PRESENTING', 'Presenting'), ('CLOSING', 'Closing'), ('OUTCOME', 'Outcome')], max_length=20, unique=True)),
                ('level', models.IntegerField(blank=True, null=True)),
                ('confidence_score', models.TextField(blank=True, null=True)),
                ('reason_for_level', models.TextField(blank=True, null=True)),
                ('statement', models.TextField()),
                ('transcription', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='analysis.transcription')),
            ],
        ),
    ]
