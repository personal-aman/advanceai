from __future__ import absolute_import, unicode_literals

import re

from celery import shared_task
from langchain.chains.llm import LLMChain
from langchain_community.chat_models.azure_openai import AzureChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from .models import Transcription, StatementClassification, StatementLevelPrompt, FinalStatementWithLevel
from .serializers import TranscriptionSerializer
from .utils import create_overlapping_segments, get_statements_data, get_important_note_and_parser
from .outputParser import EvaluationResult


# Include the necessary modules and functions such as
# get_statements_data, get_important_note_and_parser,
# classify_segments function, statement classification objects creation, etc.
@shared_task
def save_transcription(transcription_data):
    segment_length = 6000  # Adjust as needed
    overlap = 100  # Adjust as needed
    transcription_data['segments'] = create_overlapping_segments(transcription_data['text'], segment_length, overlap)
    serializer = TranscriptionSerializer(data=transcription_data)
    if serializer.is_valid():
        instance = serializer.save()
        classify_segments.delay(instance.id)
        return instance.id
    else:
        return serializer.errors

@shared_task
def classify_segments(transcription_id):
    # Same logic as we have built in the API code earlier
    transcription = Transcription.objects.get(id=transcription_id)
    total_segments = len(transcription.segments)

    for index, segment in enumerate(transcription.segments):
        # Here goes the detail classification logic that you build
        # Obtaining statement type, creating a prompt, running classification_chain, etc.

        # INITIAL_PROCESSING
        # Get all the statement category types
        statement_type_considered, different_statement_definition = get_statements_data(index + 1, total_segments)
        important_note, parser = get_important_note_and_parser(statement_type_considered, index + 1, total_segments)

        prompt_template = (
                "READ THE " + str(index + 1) + " PART OF THE TRANSCRIPT (WHOLE TRANSCRIPT HAS TOTAL " + str(
            total_segments)
                + " PARTS, with overlaps of 200 words in each parts). \n"
                  "The conversation is between a representative (REP) and one or more healthcare professionals (HCPs). "
                  "\nRead the transcript's part line  by line: \n###\n{transcript}\n### \n\n"
                  "Objective: Review and classify dialogues from interactions between pharmaceutical sales representatives (REPs) and healthcare professionals (HCPs)."
                  " Each dialogue may belong to MORE THAN ONE OR NONE of the following categories( based on its content, intention and definition of the category):\n"
                + ", ".join(statement_type_considered) + "\n"
                                                         " This requires discerning the REP's strategic approach towards initiating the conversation,"
                                                         " engaging in inquiry, presenting information, and steering towards a productive conclusion."
                                                         "\n\nDetailed Definition for Classification::\n"
                                                         "\n{different_statement_definition}\n\n"
                                                         "Important Notes:\n"
                                                         "###{important_note}\n\n"
                                                         "Maintain the original transcript form for authenticity, "
                                                         "focusing on the accuracy and relevance of each classification.###\n\n"
                                                         "Assignment Execution: ### Thoroughly proceed through the transcripts,"
                                                         " identifying and classifying each of REP dialogues according to the category's definition, category's examples and category's invalid examples."
                                                         "Also take care of the guidelines and notes provided. Utilize a nuanced approach to determine the strategic intent "
                                                         "and outcome of the dialogue within the structured interaction framework.###"
                                                         "Instruction for your output format:\n"
                                                         "\n{format_instructions}\n\n"
                                                         "Output: "
        )

        prompt = PromptTemplate(
            input_variables=[
                'transcript',
                'different_statement_definition',
                'important_note'
            ],
            template=prompt_template,
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        llm = AzureChatOpenAI(azure_deployment='test3', temperature=0.4, max_tokens=4000)
        model_name = 'test3'

        classification_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

        response = classification_chain.run(
            transcript=segment,
            different_statement_definition=different_statement_definition,
            important_note=important_note
        )
        print("response before")
        print(response)
        print("response next")
        if "```" in response:
            response = re.findall(r'```(.*?)```', response, re.DOTALL)
        statements = eval(response)
        for key in statements.keys():
            if 'opening' in key:
                category = 'OPENING'
            elif 'questioning' in key:
                category = 'QUESTIONING'
            elif 'presenting' in key:
                category = 'PRESENTING'
            elif 'closing_outcome_sentences' in key:
                category = 'CLOSING_OUTCOME'

            print("Creating objects for : ", category)
            for statement in statements[key]:
                classifcation_obj = StatementClassification(
                    category=category,
                    transcription=transcription,
                    statement=str(statement),
                    segment_number=index + 1
                )
                classifcation_obj.save()
            print("Finished creating the Classification objects for the statement type: ", category)
        # After classification, the next task in the sequence is initiated:
    level_statements.delay(transcription.id)

@shared_task
def level_statements(transcription_id):
    # Existing logic for leveling statements, similar to what we've discussed before
    # This is the logic that involves interacting with another LLMChain to
    # get the level of statements and then save the leveling results
    transcript = Transcription.objects.get(id=transcription_id)

    sentences = transcript.statementclassification_set.all()
    category_map = {
        "OPENING": "OPENING",
        "QUESTIONING": "QUESTIONING",
        "PRESENTING": "PRESENTING",
        "CLOSING": "CLOSING_OUTCOME",
        "OUTCOME": "CLOSING_OUTCOME",
    }
    for category in ['OPENING', 'QUESTIONING', 'PRESENTING', 'CLOSING', 'OUTCOME']:
        statements = list(
            sentences.filter(category=category_map[category]).filter(levelDone=False).values('id', 'statement'))
        classification_statements = sentences.filter(category=category_map[category]).filter(levelDone=False)
        # statements = sentences.filter(category=category).filter(level=0).values('id', 'statement')
        # print(statements)

        statement_level_obj = StatementLevelPrompt.objects.filter(active=True).filter(category=category).first()
        if len(statements) == 0 or not statement_level_obj:
            continue
        prompt_template = (
                category + " statements: \n{statements}\n\n"
                           "Objective: !!!{objective}!!!\n\n"
                           "Evaluation Criteria:\n ###{evaluation_criteria}###\n\n"
                           "Score assignment criteria:\n ###{score_assignment_criteria}###\n\n"
                           "Instruction:\n ###{instruction}###\n\n"
                           "Examples:\n ###{examples}###\n\n"
                           "notes:\n ###{notes}###\n\n"
                           "Instruction for your output format:\n"
                           "\n{format_instructions}\n\n"
                           "Output: "
        )
        parser = PydanticOutputParser(pydantic_object=EvaluationResult)
        prompt = PromptTemplate(
            input_variables=[
                'statements',
                'objective',
                'evaluation_criteria',
                'score_assignment_criteria',
                'instruction',
                'examples',
                'notes'
            ],
            template=prompt_template,
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        # llm = AzureOpenAI(azure_deployment='test', temperature=0, max_tokens=4000)
        # model_name = 'test'

        llm = AzureChatOpenAI(azure_deployment='test3', temperature=0.4, max_tokens=4000)
        model_name = 'test3'
        rating_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
        response = rating_chain.run(
            statements=statements,
            objective=statement_level_obj.objective,
            evaluation_criteria=statement_level_obj.evaluation_criteria,
            score_assignment_criteria=statement_level_obj.score_assignment_criteria,
            instruction=statement_level_obj.instruction,
            examples=statement_level_obj.examples,
            notes=statement_level_obj.notes
        )
        print(response)
        print("response next")
        if "```" in response:
            response = re.findall(r'```(.*?)```', response, re.DOTALL)
        scored_statements = eval(response)
        for scored_statement in scored_statements["statements"]:
            print(scored_statement)
            statement_obj = StatementClassification.objects.get(id=scored_statement['id'])
            final_statement = FinalStatementWithLevel(
                transcription=statement_obj.transcription,
                category=category,
                statement=statement_obj.statement,
                level=scored_statement['level'],
                confidence_score=scored_statement['confidence_score'],
                reason_for_level=scored_statement['reason']
            )
            final_statement.save()

            print(final_statement)
        if category != "CLOSING":
            classification_statements.update(levelDone=True)



# Invoke this from the APIView as before